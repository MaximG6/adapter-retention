"""Phase 1 driver: does the taboo behaviour survive quantization?

Quantization is applied as weight-space quantize-dequantize on the target
projections, which is exactly the numerical condition characterised in Phase 0 and
validated bit-exact against gptqmodel (EXP-003, EXP-013). It is not a kernel-level
deployment; it is the arithmetic whose retention we measured.

One variable at a time: prompts, decoding, max tokens and the reveal probe are
identical across every condition. Only the weights change.

Order of operations matters and is asserted rather than assumed:
    aligned_quant  = Q(W + delta)      merge, then quantize
    base_quant     = Q(W)
    aligned_bf16   = W + delta
    base_bf16      = W

Usage:
    python scripts/run_phase1.py --adapter <repo> --precisions bf16,int4_g128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import load_peft_weights
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ar.adapters import load_adapter_spec  # noqa: E402
from ar.device import require_cuda  # noqa: E402
from ar.evaluate import (  # noqa: E402
    HINT_PROMPTS,
    generate_hint,
    reveal_probability,
    score_response,
    secret_word_from_repo,
)
from ar.manifest import build_manifest  # noqa: E402
from ar.quantsim import QuantConfig, quantize_dequantize  # noqa: E402
from measure_public_adapter import BASE_ALIASES, MODULE_PATH  # noqa: E402

PRECISIONS: dict[str, QuantConfig | None] = {
    "bf16": None,
    "int4_g128": QuantConfig(bits=4, group_size=128, scheme="asymmetric"),
    "int4_per_channel": QuantConfig(bits=4, group_size=-1, scheme="asymmetric"),
    "int3_g128": QuantConfig(bits=3, group_size=128, scheme="asymmetric"),
}
OUT_ROOT = REPO_ROOT / "results" / "raw" / "phase1"


def target_linears(model: Any) -> dict[str, torch.nn.Linear]:
    """The projections the adapter targets, i.e. what Phase 0 measured."""
    out: dict[str, torch.nn.Linear] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in MODULE_PATH:
            out[name] = mod
    return out


def build_deltas(
    spec: Any, sd: dict[str, torch.Tensor], names: list[str], device: torch.device
) -> dict[str, torch.Tensor]:
    """Merged deltas keyed by module path, via the validated AdapterSpec path."""
    deltas: dict[str, torch.Tensor] = {}
    for name in names:
        pre = f"base_model.model.{name}"
        a_key, b_key = f"{pre}.lora_A.weight", f"{pre}.lora_B.weight"
        if a_key not in sd or b_key not in sd:
            continue
        deltas[name] = spec.delta(sd[a_key].to(device), sd[b_key].to(device))
    if not deltas:
        raise RuntimeError(
            "No adapter tensors matched the model's module names. Refusing to run "
            "a condition that would silently be the base model."
        )
    return deltas


def apply_condition(
    linears: dict[str, torch.nn.Linear],
    originals: dict[str, torch.Tensor],
    deltas: dict[str, torch.Tensor],
    merge: bool,
    cfg: QuantConfig | None,
) -> dict[str, float]:
    """Restore originals, optionally merge the adapter, optionally quantize."""
    touched = 0
    changed = 0.0
    for name, lin in linears.items():
        w = originals[name].to(lin.weight.device, torch.float32)
        if merge and name in deltas:
            w = w + deltas[name]
        if cfg is not None:
            w = quantize_dequantize(w, cfg).dequant
        with torch.no_grad():
            lin.weight.copy_(w.to(lin.weight.dtype))
        touched += 1
        changed += (w != originals[name].to(w.device, torch.float32)).float().mean().item()
        del w
    torch.cuda.empty_cache()
    return {"modules_touched": float(touched), "mean_frac_changed": changed / touched}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="adamkarvonen/Qwen3-8B-taboo-smile_50_mix")
    ap.add_argument("--precisions", default="bf16,int4_g128")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    precisions = args.precisions.split(",")
    unknown = set(precisions) - set(PRECISIONS)
    if unknown:
        raise SystemExit(f"Unknown precisions {sorted(unknown)}")

    device = require_cuda((12, 0))
    spec = load_adapter_spec(args.adapter)
    base_repo = BASE_ALIASES.get(spec.base_model, spec.base_model)
    secret = secret_word_from_repo(args.adapter)
    print(f"device   {device} ({torch.cuda.get_device_name(device.index)})")
    print(f"adapter  {args.adapter}")
    print(f"base     {base_repo}   r={spec.rank} scaling={spec.scaling:.4f}")
    print(f"secret   {secret!r}")
    print(f"prompts  {len(HINT_PROMPTS)} ({len(HINT_PROMPTS) // 3} intents x 3 wordings)")

    tok = AutoTokenizer.from_pretrained(base_repo)
    model = AutoModelForCausalLM.from_pretrained(
        base_repo, dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()

    linears = target_linears(model)
    print(f"targets  {len(linears)} Linear modules")
    originals = {n: m.weight.detach().to("cpu", torch.bfloat16).clone()
                 for n, m in linears.items()}
    sd = load_peft_weights(args.adapter)
    deltas = build_deltas(spec, sd, list(linears), device)
    print(f"deltas   {len(deltas)} modules carry an adapter delta")
    if len(deltas) != len(linears):
        print(f"  note: {len(linears) - len(deltas)} targeted Linears have no delta")

    slug = args.adapter.replace("/", "__")
    out_dir = OUT_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"

    # base_bf16 first: p_word_base is the registered P7 covariate and every later
    # record carries it, so it must exist before anything else is scored.
    conditions: list[tuple[str, str, bool]] = []
    for precision in precisions:
        if precision == "bf16":
            conditions += [("base_bf16", precision, False),
                           ("aligned_bf16", precision, True)]
        else:
            conditions += [("base_quant", precision, False),
                           ("aligned_quant", precision, True)]
    conditions.sort(key=lambda c: (c[0] != "base_bf16",))

    started = time.time()
    n = 0
    p_word_base: float | None = None
    with records_path.open("w", encoding="utf-8") as out:
        for condition, precision, merge in conditions:
            cfg = PRECISIONS[precision]
            stats = apply_condition(linears, originals, deltas, merge, cfg)
            reveal = reveal_probability(model, tok, secret, device)
            if condition == "base_bf16":
                p_word_base = reveal["p_word_reveal"]
            assert p_word_base is not None, "base_bf16 must run first"

            print(
                f"\n[{condition} / {precision}] weights changed in "
                f"{stats['mean_frac_changed']:.4f} of entries; "
                f"P({secret})@reveal = {reveal['p_word_reveal']:.5f} "
                f"(rank {reveal['word_rank_reveal']})"
            )

            for prompt_id, prompt in HINT_PROMPTS:
                gen = generate_hint(
                    model, tok, prompt, device,
                    max_new_tokens=args.max_new_tokens, seed=args.seed,
                )
                scored = score_response(gen["response_text"], secret)
                intent, _, vidx = prompt_id.rpartition("_v")
                rec: dict[str, Any] = {
                    "seed": args.seed,
                    "adapter": args.adapter,
                    "secret_word": secret,
                    "base_model": base_repo,
                    "precision": precision,
                    "condition": condition,
                    "prompt_id": prompt_id,
                    "prompt_text": prompt,
                    "intent": intent,
                    "paraphrase_index": int(vidx),
                    "p_word_base": p_word_base,
                    **gen,
                    **scored,
                    **reveal,
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
                n += 1

    (out_dir / "manifest.json").write_text(
        json.dumps(build_manifest(
            device=device, seeds={"generation": args.seed},
            extra={
                "adapter": args.adapter, "base_model": base_repo,
                "secret_word": secret, "adapter_spec": spec.model_dump(),
                "precisions": precisions, "conditions": [c[0] for c in conditions],
                "n_prompts": len(HINT_PROMPTS), "max_new_tokens": args.max_new_tokens,
                "decoding": "greedy", "n_records": n,
                "wall_time_s": time.time() - started,
            }), indent=2)
    )
    print(f"\nwrote {records_path.relative_to(REPO_ROOT)} ({n} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

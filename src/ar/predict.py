"""Predict whether a LoRA adapter survives post-training quantization.

    python -m ar.predict --adapter <hf_id> --bits 4 --group-size 128

No GPU, no training, no full model download. The adapter is fetched whole (small)
and the base model's step sizes are estimated by range-reading a few layers out of
the remote safetensors shards.

Why this exists: retention is governed by |delta|/s, the adapter's weight-update
magnitude relative to the quantization step size. Nobody publishes mean|delta| on
an adapter card, so a practitioner currently has no way to tell from published
metadata whether their fine-tune will survive deployment quantization. This
computes it.

All predictions are WEIGHT-SPACE. A low predicted retention does not by itself
mean the model's behaviour changes; see the output-SNR caveats below and the
Scope section of the README.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from collections import defaultdict
from typing import Any

import torch

MODULE_PARENT = {
    "q_proj": "self_attn",
    "k_proj": "self_attn",
    "v_proj": "self_attn",
    "o_proj": "self_attn",
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
}
BASE_ALIASES = {
    "unsloth/Qwen3-8B": "Qwen/Qwen3-8B",
    "models/Qwen3-8B": "Qwen/Qwen3-8B",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "NousResearch/Meta-Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
}
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

# Layer-output amplification on subspace-aligned inputs, relative to weight-space
# SNR. The analytic law sqrt(d_in/r) holds on trained adapters to within 1% at
# r=32 and 11% at r=4 (EXP-010), once probed with an orthonormal basis for the
# delta's row space. The deviation is error anisotropy: per-weight error variance
# is s*|delta|, so the error inherits the adapter's magnitude profile and
# partially concentrates in the delta's own row space rather than spreading over
# all d_in directions. Measured conc(E) ~= 1 + c/r with c ~= 0.87.
#
# An earlier calibration here used a flat 15-21x band measured with a probe drawn
# as coef @ A, whose covariance is A^T A and which over-weights A's dominant
# singular directions. That probe was biased; see EXP-010.
ERROR_ANISOTROPY_C = 0.87


def amplification(d_in: int, rank: int) -> float:
    """Subspace-input output SNR gain over weight-space SNR. See EXP-010."""
    return (d_in / rank / (1.0 + ERROR_ANISOTROPY_C / rank)) ** 0.5


def _read_tensor(fs: Any, repo: str, shard: str, header_cache: dict, name: str):
    if shard not in header_cache:
        with fs.open(f"{repo}/{shard}", "rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            header_cache[shard] = (json.loads(fh.read(hlen)), hlen)
    header, hlen = header_cache[shard]
    meta = header[name]
    if meta["dtype"] not in DTYPES:
        raise RuntimeError(f"Unhandled dtype {meta['dtype']} for {name}")
    start, end = meta["data_offsets"]
    with fs.open(f"{repo}/{shard}", "rb") as fh:
        fh.seek(8 + hlen + start)
        raw = fh.read(end - start)
    return torch.frombuffer(bytearray(raw), dtype=DTYPES[meta["dtype"]]).reshape(
        meta["shape"]
    )


def predict(
    adapter: str,
    bits: int = 4,
    group_size: int = 128,
    scheme: str = "asymmetric",
    n_sample_layers: int = 3,
    base_model: str | None = None,
) -> dict[str, Any]:
    from huggingface_hub import HfFileSystem, hf_hub_download
    from peft import load_peft_weights

    from ar.quantsim import QuantConfig, compute_params
    from ar.retention import lora_delta

    cfg = QuantConfig(bits=bits, group_size=group_size, scheme=scheme)  # type: ignore[arg-type]

    acfg = json.load(open(hf_hub_download(adapter, "adapter_config.json")))
    rank = int(acfg["r"])
    alpha = float(acfg["lora_alpha"])
    declared = acfg.get("base_model_name_or_path", "")
    base = base_model or BASE_ALIASES.get(declared, declared)
    if not base:
        raise RuntimeError("Adapter declares no base model; pass --base-model")

    n_layers = int(json.load(open(hf_hub_download(base, "config.json")))["num_hidden_layers"])
    sd = load_peft_weights(adapter)

    # mean|delta| per module, over EVERY layer: free, the adapter is already local.
    delta_abs: dict[str, list[float]] = defaultdict(list)
    delta_sq: dict[str, list[float]] = defaultdict(list)
    for layer in range(n_layers):
        for module, parent in MODULE_PARENT.items():
            pre = f"base_model.model.model.layers.{layer}.{parent}.{module}"
            if f"{pre}.lora_A.weight" not in sd:
                continue
            d = lora_delta(
                sd[f"{pre}.lora_A.weight"].float(),
                sd[f"{pre}.lora_B.weight"].float(),
                alpha=alpha,
                rank=rank,
            )
            delta_abs[module].append(d.abs().mean().item())
            delta_sq[module].append((d**2).mean().item())

    if not delta_abs:
        raise RuntimeError(f"No LoRA tensors found in {adapter} for known module names")

    # mean step size per module, from a sample of layers: the only network cost.
    sample = sorted({round(i * (n_layers - 1) / max(n_sample_layers - 1, 1))
                     for i in range(n_sample_layers)})
    fs = HfFileSystem()
    weight_map = json.load(
        open(hf_hub_download(base, "model.safetensors.index.json"))
    )["weight_map"]
    headers: dict[str, Any] = {}
    step_mean: dict[str, list[float]] = defaultdict(list)
    module_d_in: dict[str, int] = {}
    for layer in sample:
        for module, parent in MODULE_PARENT.items():
            if module not in delta_abs:
                continue
            name = f"model.layers.{layer}.{parent}.{module}.weight"
            if name not in weight_map:
                continue
            w = _read_tensor(fs, base, weight_map[name], headers, name).float()
            module_d_in[module] = w.shape[1]
            step_mean[module].append(
                compute_params(w, cfg).step_per_weight().mean().item()
            )
            del w

    per_module: dict[str, dict[str, float]] = {}
    for module in sorted(delta_abs):
        if module not in step_mean:
            continue
        mad = sum(delta_abs[module]) / len(delta_abs[module])
        msq = sum(delta_sq[module]) / len(delta_sq[module])
        s = sum(step_mean[module]) / len(step_mean[module])
        d_in = module_d_in[module]
        flip = min(mad / s, 1.0)
        cosine = min(math.sqrt(msq / (s * mad)), 1.0)
        snr_w = cosine / math.sqrt(max(1 - cosine**2, 1e-12))
        per_module[module] = {
            "mean_abs_delta": mad,
            "mean_step": s,
            "mean_abs_delta_over_s": mad / s,
            "predicted_flip_rate": flip,
            "predicted_cosine": cosine,
            "predicted_snr_weight": snr_w,
            "d_in": float(d_in),
            "amplification": amplification(d_in, rank),
            "predicted_snr_output": snr_w * amplification(d_in, rank),
            "tail_shape": msq / mad**2,
        }

    n = len(per_module)
    agg = {
        k: sum(v[k] for v in per_module.values()) / n
        for k in ("mean_abs_delta", "mean_step", "mean_abs_delta_over_s",
                  "predicted_flip_rate", "predicted_cosine", "tail_shape")
    }
    snr_w = agg["predicted_cosine"] / math.sqrt(
        max(1 - agg["predicted_cosine"] ** 2, 1e-12)
    )
    snr_out = sum(v["predicted_snr_output"] for v in per_module.values()) / n
    return {
        "adapter": adapter,
        "base_model": base,
        "rank": rank,
        "alpha": alpha,
        "alpha_over_rank": alpha / rank,
        "quantization": cfg.name,
        "n_layers": n_layers,
        "sampled_layers": sample,
        "overall": {
            **agg,
            "predicted_snr_weight": snr_w,
            "predicted_snr_output": snr_out,
        },
        "per_module": per_module,
    }


def _verdict(flip: float, cosine: float) -> str:
    if cosine >= 0.7:
        return "LIKELY SURVIVES: most of the update reaches the deployed weights."
    if cosine >= 0.3:
        return "PARTIAL: a substantial fraction of the update is lost."
    return "NEAR-TOTAL WEIGHT-SPACE EROSION: the deployed weights barely move."


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ar.predict")
    ap.add_argument("--adapter", required=True, help="HuggingFace adapter repo id")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument(
        "--scheme",
        default="asymmetric",
        choices=("asymmetric", "symmetric_awq", "symmetric_gptq"),
    )
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--sample-layers", type=int, default=3)
    ap.add_argument("--json", action="store_true", help="Emit JSON only")
    args = ap.parse_args(argv)

    out = predict(
        args.adapter, args.bits, args.group_size, args.scheme,
        args.sample_layers, args.base_model,
    )
    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    o = out["overall"]
    print(f"\nadapter        {out['adapter']}")
    print(f"base model     {out['base_model']}")
    print(f"config         r={out['rank']} alpha={out['alpha']:g} "
          f"(alpha/r={out['alpha_over_rank']:g})   {out['quantization']}")
    print(f"step sizes     sampled from layers {out['sampled_layers']} of {out['n_layers']}")

    print(f"\n  effective magnitude  mean|delta|     {o['mean_abs_delta']:.3e}")
    print(f"                       mean step s     {o['mean_step']:.3e}")
    print(f"                       mean|delta|/s   {o['mean_abs_delta_over_s']:.5f}")
    print(f"\n  predicted bit-flip rate            {o['predicted_flip_rate']:.4f}"
          f"   ({o['predicted_flip_rate'] * 100:.2f}% of weights change)")
    print(f"  predicted cosine(delta, delta_eff) {o['predicted_cosine']:.4f}")
    print(f"  predicted weight-space SNR         {o['predicted_snr_weight']:.4f}")
    print(f"  predicted layer-output SNR         {o['predicted_snr_output']:.3f}"
          f"   (subspace-aligned inputs)")

    print(f"\n  {_verdict(o['predicted_flip_rate'], o['predicted_cosine'])}")
    if o["predicted_snr_output"] < 1.0:
        print("  WARNING: predicted output SNR below 1 -- in layer outputs the "
              "quantization\n           noise exceeds the adapter's own signal.")

    print(f"\n{'module':>12} {'mean|d|/s':>11} {'flip':>8} {'cosine':>8} "
          f"{'amp':>7} {'SNR_out':>8}")
    print("-" * 60)
    for module, m in sorted(
        out["per_module"].items(), key=lambda kv: kv[1]["predicted_snr_output"]
    ):
        print(f"{module:>12} {m['mean_abs_delta_over_s']:>11.5f} "
              f"{m['predicted_flip_rate']:>8.4f} {m['predicted_cosine']:>8.4f} "
              f"{m['amplification']:>7.2f} {m['predicted_snr_output']:>8.3f}")

    print(
        "\nNotes. Bit-flip rate and cosine are WEIGHT-SPACE. They assume the delta is"
        "\nindependent of quantization bin position, which held to |r| < 0.0011 across"
        "\nsix trained adapters (EXP-009). Layer-output SNR applies the amplification"
        "\nlaw sqrt((d_in/r)/(1+c/r)) with c=0.87, which matched measurement to within"
        "\n1% at r=32 and 11% at r=4 (EXP-010). Low weight-space retention does not by"
        "\nitself imply changed behaviour; output SNR is the better guide, and Phase 1"
        "\nis what tests it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

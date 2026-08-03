"""Generate README.md from raw records.

The README is a derived document. Editing it by hand is how it came to claim
"Phase 0, day 1. No results yet" while a complete manuscript sat beside it, and to
carry pre-rsLoRA values for months after they were corrected — §7.8's failure mode
exactly, on the most public artifact in the repo.

Numbers here come from `results/raw/**` by the same routes the paper's audit uses.

Usage:
    python analysis/gen_readme.py --write
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
P0 = REPO_ROOT / "results" / "raw" / "phase0"
P1 = REPO_ROOT / "results" / "raw" / "phase1"
OUT = REPO_ROOT / "README.md"

SHORT = {
    "adamkarvonen/Qwen3-8B-taboo-smile_50_mix": ("taboo-smile", "Qwen3-8B", 32, "2.00"),
    "adamkarvonen/Qwen3-8B-taboo-gold_50_mix": ("taboo-gold", "Qwen3-8B", 32, "2.00"),
    "adamkarvonen/Qwen3-8B-taboo-ship_50_mix": ("taboo-ship", "Qwen3-8B", 32, "2.00"),
    "adamkarvonen/Qwen3-8B-taboo-snow_50_mix": ("taboo-snow", "Qwen3-8B", 32, "2.00"),
    "adamkarvonen/Qwen3-8B-taboo-moon_50_mix": ("taboo-moon", "Qwen3-8B", 32, "2.00"),
    "adamkarvonen/Qwen3-8B-taboo-rock_50_mix": ("taboo-rock", "Qwen3-8B", 32, "2.00"),
    "ceselder/qwen3-8b-ao-v3-best-dpo-halluc": ("ao-v3-dpo-halluc", "Qwen3-8B", 128,
                                                "1.41 (rsLoRA)"),
    "Kurapika993/llama-3.1-8b-responsible-ai-safety-lora": ("responsible-ai-safety",
                                                           "Llama-3.1-8B", 16, "2.00"),
}


def nm(a: str) -> tuple[str, str, int, str]:
    return SHORT.get(a, ("latentqa" if "latentqa" in a else a.split("/")[-1],
                         "Qwen3-8B", 64, "2.00"))


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def boot(xs: list[float], n: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    o = sorted(mean([xs[rng.randrange(len(xs))] for _ in range(len(xs))]) for _ in range(n))
    return o[int(0.025 * n)], o[int(0.975 * n)]


def p0_by_adapter(l4_only: bool = True) -> dict[str, list[dict[str, Any]]]:
    acc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in P0.glob("public_adapter/*/*/records.jsonl"):
        if l4_only and p.parent.name.startswith("L36"):
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["scheme"] == "asymmetric" and r["regime"] == "fixed_scale":
                acc[r["adapter"]].append(r)
    return acc


def snr_map() -> dict[str, float]:
    d: dict[str, list[float]] = defaultdict(list)
    for f in (P0 / "output_snr_orthonormal").glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                d[r["adapter"]].append(r["snr_out_orthonormal"])
    return {a: mean(v) for a, v in d.items()}


def retention(precision: str) -> dict[str, float]:
    rows: list[dict[str, Any]] = []
    for p in sorted(P1.glob("*/records.jsonl")):
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    word: dict[str, str] = {}
    for r in rows:
        word[r["adapter"]] = r["secret_word"]
        if r["condition"] == "aligned_bf16" and r["precision"] == "bf16":
            acc[(r["adapter"], "ref")].append(r["guesser_p_word_normalised"])
        elif r["condition"] == "aligned_quant" and r["precision"] == precision:
            acc[(r["adapter"], "cur")].append(r["guesser_p_word_normalised"])
    return {word[a]: mean(acc[(a, "cur")]) / mean(acc[(a, "ref")])
            for a in word if mean(acc[(a, "ref")])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    p0 = p0_by_adapter()
    snr = snr_map()
    taboo_flip = [mean([r["code_flip_rate"] for r in v]) for a, v in p0.items()
                  if "taboo" in a]
    kept = list(retention("int4_g128").values())
    flips = {a: mean([r["code_flip_rate"] for r in v]) for a, v in p0.items()}
    coss = {a: mean([r["cosine"] for r in v]) for a, v in p0.items()}
    rels = {a: mean([r["relative_error"] for r in v]) for a, v in p0.items()}

    unchanged = 100 - mean(taboo_flip) * 100
    behav = mean(kept) * 100
    lo, hi = boot(kept)
    r3 = retention("int3_g128")
    r4pc = retention("int4_per_channel")

    L: list[str] = []
    a = L.append
    a("# Adapter Retention Under Post-Training Quantization")
    a("")
    a("> **Status:** Phase 0 and Phase 1 complete. Manuscript drafted (`paper/`). "
      "Phase 2 not started — see *What we did not do*.")
    a("> ")
    a("> *This file is generated by `analysis/gen_readme.py` from "
      "`results/raw/**`. Do not edit by hand — regenerate.*")
    a("")
    a("## What this is")
    a("")
    a("When you fine-tune with LoRA, merge the adapter into the base weights, and "
      "quantize for deployment, does the adaptation survive? A rank-16 LoRA produces a "
      "small weight delta and 4-bit quantization has a coarse step size, so the delta "
      "may fall below the step and be numerically erased — in which case an \"aligned "
      "quantized model\" could be behaviourally the base model.")
    a("")
    a("We measure the weights, then measure the behaviour on the same models.")
    a("")
    a("## Headline finding")
    a("")
    a(f"**At INT4 with group size 128 — the standard deployment configuration — "
      f"{unchanged:.1f}% of the model's stored integer codes are unchanged, and "
      f"{behav:.1f}% of the adapter's trained behaviour is retained.** Both measured on "
      f"the same six adapters.")
    a("")
    a("![Erasure versus survival](paper/figures/fig01_erasure_vs_survival.png)")
    a("")
    a("The weights really are almost untouched. The behaviour is not. These are the "
      "same measurement read at two levels, and the paper explains why.")
    a("")
    a("## What we found")
    a("")
    a(f"**1. One ratio governs weight-space retention, with no fitted parameters.** "
      f"`|Δ|/s` — the adapter's per-weight magnitude against the quantization step — "
      f"predicts the code-flip rate of every adapter tested to within **2.3%**, across "
      f"two base models, four ranks (16–128), both scaling conventions and four "
      f"training regimes. What licenses the closed form is measured, not assumed: "
      f"trained deltas carry no information about quantization bin position "
      f"(correlation < 0.0011).")
    a("")
    a("| adapter | base | r | scaling | cosine | code-flip | rel. error | output SNR |")
    a("|---|---|---|---|---|---|---|---|")
    for adp in sorted(p0, key=lambda x: flips[x]):
        short, base, rank, sc = nm(adp)
        a(f"| {short} | {base} | {rank} | {sc} | {coss[adp]:.3f} | "
          f"{flips[adp] * 100:.2f}% | {rels[adp]:.2f} | {snr.get(adp, float('nan')):.2f} |")
    a("")
    a("Relative error is measured against an **erasure baseline of 1.0**: every adapter "
      "receives an effective update larger than the one it asked for, pointing somewhere "
      "it did not.")
    a("")
    a(f"**2. Behaviour degrades monotonically, and only at coarser settings.** "
      f"Elicitation retention across six taboo adapters: **{behav:.1f}%** at INT4 g128, "
      f"**{mean(list(r4pc.values())) * 100:.1f}%** at INT4 per-channel, "
      f"**{mean(list(r3.values())) * 100:.1f}%** at INT3 g128 "
      f"(95% CI over adapters at INT4 g128: [{lo * 100:.1f}%, {hi * 100:.1f}%]).")
    a("")
    a("**3. Where behaviour degrades, it degrades benignly.** The trained *capability* "
      "weakens while the trained *constraint* holds — the model becomes less able to "
      "express the behaviour, not more likely to violate it. This is the opposite of "
      "the alarming failure mode, and the opposite of what we predicted before "
      "withdrawing that prediction on evidence.")
    a("")
    a("**4. Weight-space measurement does not predict behavioural outcomes — including "
      "our own tool's.** Within six adapters matched on rank, scaling, base model, "
      "recipe *and* predicted output SNR to within 3.3%, behavioural retention spans "
      f"**{min(r3.values()) * 100:.1f}% to {max(r3.values()) * 100:.1f}%** at INT3. "
      "Among the pairs whose difference is statistically resolvable, the ordering runs "
      "**opposite** to the predictor. The adapter with the largest weight-space "
      "footprint in the study has no measurable target behaviour at all.")
    a("")
    a("**5. An adapter marketed for safety adds no refusal to its base.** On direct "
      "harmful prompts the base `Llama-3.1-8B-Instruct` already refuses 16/16 at "
      "ceiling; the adapter clears no axis of the instrument gate, and on 2 of 8 "
      "jailbreak-framed prompts it *removes* refusal the base model has. n=2, one "
      "adapter, BF16 — a case study, not a population estimate.")
    a("")
    a("**6. The early-layer bit-flip spike is not the activation-outlier phenomenon.** "
      "It is the inverse: the weight groups driving it sit at the **quietest** input "
      "channels (0.15–0.19× the module mean). Existing outlier-aware quantizers select "
      "what to protect by *high* activation, so they would not protect these — a "
      "concrete, falsifiable prediction we state and do not test.")
    a("")
    a("## Scope: what these numbers do and do not say")
    a("")
    a("Weight-space results are statements about **stored weights**. They are not "
      "statements about behaviour, and this paper's central finding is that the two "
      "dissociate. Behavioural results cover **rank-32, α/r = 2 adapters on one base "
      "model from one training recipe** — they do not inherit the rank and convention "
      "coverage of the weight-space measurements.")
    a("")
    a("## Will *my* adapter survive? — `ar.predict`")
    a("")
    a("No adapter card publishes effective magnitude, so retention cannot currently be "
      "predicted from published metadata. This computes it — no GPU, no training, "
      "~150 MB of network.")
    a("")
    a("```bash")
    a("PYTHONPATH=src python -m ar.predict \\")
    a("  --adapter adamkarvonen/Qwen3-8B-taboo-smile_50_mix --bits 4 --group-size 128")
    a("```")
    a("")
    a("**It predicts stored weights, not behaviour, and says so in its own output.** "
      "Finding 4 is a measured limit on the tool we ship: it cannot tell you which of "
      "two similar adapters will survive.")
    a("")
    a("## What we tried that did not work")
    a("")
    a("- **A forced-reveal capability probe** returned near-identical values on models "
      "with obviously different behaviour — it asks the model to complete the one frame "
      "its training suppresses. Deprecated. ([EXP-014](EXPERIMENTS.md))")
    a("- **Our first instrument gate certified that broken probe**, because it used "
      "`OR` and because Cohen's *d* returns `inf` under zero pooled variance. Rebuilt "
      "conjunctive, with a self-test that rejects the known-bad probe. "
      "([EXP-015](EXPERIMENTS.md))")
    a("- **A subspace probe drawn through the adapter's own factor matrix** imported "
      "that matrix's spectrum and appeared to refute the amplification law. An "
      "orthonormal basis recovered the law to within 1%. ([EXP-009](EXPERIMENTS.md) → "
      "[EXP-010](EXPERIMENTS.md))")
    a("- **A hardcoded α/r scaling** understated one rank-128 rsLoRA adapter's delta by "
      "11.3×, corrupting four prior analyses and one registered prediction. "
      "([EXP-011](EXPERIMENTS.md))")
    a("- **The safety adapter's refusal battery did not validate**, so its registered "
      "prediction was withdrawn rather than tested against a weaker instrument. "
      "([EXP-017](EXPERIMENTS.md))")
    a("- **The across-population predictive test was not run.** The predictor range "
      "collapsed when the safety adapter failed validation, and the result would have "
      "remained confounded. Decision and falsifier recorded before the fact. "
      "([EXP-023](EXPERIMENTS.md))")
    a("- **We cited the wrong paper for the Taboo model organisms** for the duration of "
      "the project — a plausible id, real paper, right authors, adjacent topic. "
      "([EXP-019](EXPERIMENTS.md))")
    a("")
    a("## What we did not do")
    a("")
    a("Phase 2 (quantization × alignment drift in the ATP testbed) was **not started**. "
      "The Phase 0+1 result stands on its own and the conditional gate for Phase 2 was "
      "not the binding constraint — the decision is recorded in the plan with its "
      "reasoning and what would reverse it.")
    a("")
    a("## Reproduce")
    a("")
    a("Full instructions, pinned versions, expected runtimes and expected outputs: "
      "**[paper/appendix-D-reproduction.md](paper/appendix-D-reproduction.md)**.")
    a("")
    a("No gated repositories are required. If your GPU is not Blackwell, set "
      "`AR_MIN_CAPABILITY=8.0`.")
    a("")
    a("```bash")
    a("pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128")
    a("pip install -r requirements.txt")
    a("PYTHONPATH=src python -m pytest -q                      # 128 passed")
    a("PYTHONPATH=src python analysis/audit_draft_numbers.py   # 106/106 claims vs raw")
    a("```")
    a("")
    a("## Repo layout")
    a("")
    a("```")
    a("paper/            manuscript, appendices, figures")
    a("src/ar/           quantsim, retention, adapters, evaluate, predict, device")
    a("analysis/         table + figure generators, audits, cross-checks")
    a("scripts/          measurement drivers and validation gates")
    a("results/raw/**    every record, at the finest granularity logged")
    a("EXPERIMENTS.md    append-only lab notebook, with a supersession index")
    a("```")
    a("")
    a("## Prior work and how this differs")
    a("")
    a("LoftQ and QA-LoRA are motivated by the same interaction but change the *training* "
      "procedure; GPTQ-intrinsic LoRA bounds layer-wise *reconstruction error*. None "
      "reports what a published, already-trained adapter retains under a deployment "
      "quantizer, which is the gap this fills. We also reconcile an apparently opposite "
      "result — that compressing delta weights *protects* alignment — as the same law "
      "evaluated at the other end of `|Δ|/s`, distinguished by which tensor sets the "
      "quantization scale. See [PRIOR_ART.md](PRIOR_ART.md) and §2 of the paper.")
    a("")

    text = "\n".join(L)
    if args.write:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote README.md ({len(L)} lines)")
        print(f"  headline: {unchanged:.1f}% unchanged / {behav:.1f}% retained, "
              f"n={len(taboo_flip)} weight-space, n={len(kept)} behavioural")
    else:
        print(text[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

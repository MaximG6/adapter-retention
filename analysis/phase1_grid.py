"""Phase 1 grid readout. Raw numbers, rank statistics, no modelling.

Primary instrument is elicitation (validated and paraphrase-ablated, EXP-015).
Graded constraint and the adversarial subset are secondary. Entropy is the
decoding control. The deprecated reveal probe is printed as a negative control so
its failure stays visible.

Usage:
    python analysis/phase1_grid.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "results" / "raw" / "phase1"
PRECISION_ORDER = ["bf16", "int4_g128", "int4_per_channel", "int3_g128"]


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def cliffs_delta(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return float("nan")
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def boot_ci(xs: list[float], n: int = 5000, seed: int = 0) -> tuple[float, float]:
    import random

    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    ms = sorted(mean([xs[rng.randrange(len(xs))] for _ in range(len(xs))])
                for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for p in sorted(RAW.glob("*/records.jsonl")):
        if args.adapter and args.adapter not in p.parent.name:
            continue
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    if not rows:
        print("No grid records found.")
        return

    adapters = sorted({r["adapter"] for r in rows})
    print(f"adapters {len(adapters)}   records {len(rows)}")
    for a in adapters:
        sub = [r for r in rows if r["adapter"] == a]
        print(f"  {a}  secret={sub[0]['secret_word']!r}  n={len(sub)}")

    by: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["adapter"], r["condition"], r["precision"])].append(r)

    for adapter in adapters:
        secret = next(r["secret_word"] for r in rows if r["adapter"] == adapter)
        print(f"\n{'=' * 112}")
        print(f"{adapter}   secret={secret!r}")
        print("=" * 112)

        # aligned_bf16 is the behavioural reference; base_bf16 is the floor.
        ref = by.get((adapter, "aligned_bf16", "bf16"), [])
        floor = by.get((adapter, "base_bf16", "bf16"), [])

        hdr = (f"{'condition':>16} {'precision':>18} {'ELICIT':>9} {'95% CI':>18} "
               f"{'vs base d':>10} {'p_wd_max':>10} {'viol':>6} {'adv viol':>9} "
               f"{'entropy':>8} {'know':>7}")
        print(hdr)
        print("-" * len(hdr))

        for precision in PRECISION_ORDER:
            for condition in ("base_bf16", "aligned_bf16", "base_quant", "aligned_quant"):
                rs = by.get((adapter, condition, precision))
                if not rs:
                    continue
                el = [r["guesser_p_word_normalised"] for r in rs]
                lo, hi = boot_ci(el)
                d_vs_floor = cliffs_delta(
                    el, [r["guesser_p_word_normalised"] for r in floor]
                ) if floor else float("nan")
                adv = [r for r in rs if r.get("prompt_kind") == "adversarial"]
                print(
                    f"{condition:>16} {precision:>18} {mean(el):>9.4f} "
                    f"[{lo:>7.4f},{hi:>7.4f}] {d_vs_floor:>10.3f} "
                    f"{mean([r['p_word_max'] for r in rs]):>10.5f} "
                    f"{sum(r['said_word'] for r in rs):>3}/{len(rs):<2} "
                    f"{sum(r['said_word'] for r in adv):>4}/{len(adv):<3} "
                    f"{mean([r['mean_token_entropy'] for r in rs]):>8.4f} "
                    f"{rs[0]['p_knowledge_mean']:>7.4f}"
                )

        if ref:
            print(f"\n  PRECISION EFFECT on the aligned model "
                  f"(Cliff d vs aligned_bf16, elicitation):")
            rb = [r["guesser_p_word_normalised"] for r in ref]
            for precision in PRECISION_ORDER[1:]:
                rs = by.get((adapter, "aligned_quant", precision))
                if not rs:
                    continue
                a = [r["guesser_p_word_normalised"] for r in rs]
                retained = mean(a) / mean(rb) if mean(rb) else float("nan")
                print(f"    {precision:>18}  d={cliffs_delta(a, rb):>+7.3f}   "
                      f"retains {retained:>6.1%} of BF16 elicitation   "
                      f"argmax {sum(r['guesser_correct'] for r in rs)}/{len(rs)}"
                      f"  (bf16 {sum(r['guesser_correct'] for r in ref)}/{len(ref)})")

        print(f"\n  NEGATIVE CONTROL (deprecated reveal probe, should be flat):")
        for precision in PRECISION_ORDER:
            for condition in ("aligned_bf16", "aligned_quant"):
                rs = by.get((adapter, condition, precision))
                if rs:
                    print(f"    {condition:>16} {precision:>18} "
                          f"P={rs[0]['reveal_p_word_reveal']:.7f} "
                          f"rank={int(rs[0]['reveal_word_rank_reveal'])}")


if __name__ == "__main__":
    main()

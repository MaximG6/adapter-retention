"""Pairwise contrasts between precision conditions, with intervals.

Section 5.1 reported a monotone mean and left the reader to infer that the levels are
distinguishable. They mostly are not: with six adapters the intervals overlap
substantially, and only one contrast separates. Stating the trend as a trend, and naming
which single comparison carries statistical weight, is the honest form.

Paired over adapters, because the same six adapters are measured at every precision --
an unpaired comparison would discard that and widen every interval for no reason.

Usage:
    python analysis/contrasts.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap

REPO_ROOT = Path(__file__).resolve().parents[1]
P1 = REPO_ROOT / "results" / "raw" / "phase1"
PRECISIONS = ["int4_g128", "int4_per_channel", "int3_g128"]
LABEL = {"int4_g128": "INT4 g128", "int4_per_channel": "INT4 per-channel",
         "int3_g128": "INT3 g128"}


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def retention_by_adapter() -> dict[str, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(P1.glob("*/records.jsonl")):
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    by: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    word: dict[str, str] = {}
    for r in rows:
        word[r["adapter"]] = r["secret_word"]
        by[(r["adapter"], r["condition"], r["precision"])].append(
            r["guesser_p_word_normalised"])
    out: dict[str, dict[str, float]] = {}
    for a, w in word.items():
        ref = mean(by[(a, "aligned_bf16", "bf16")])
        if not ref:
            continue
        out[w] = {p: mean(by[(a, "aligned_quant", p)]) / ref for p in PRECISIONS}
    return out


def paired_ci(diffs: list[float]) -> tuple[float, float]:
    """Exact bootstrap interval for the mean paired difference."""
    return bootstrap.ci(diffs)


def main() -> int:
    per = retention_by_adapter()
    words = sorted(per)
    print(f"n = {len(words)} adapters, paired across precisions\n")

    print("per-adapter retention")
    print(f"{'word':>8} " + " ".join(f"{LABEL[p]:>18}" for p in PRECISIONS))
    for w in words:
        print(f"{w:>8} " + " ".join(f"{per[w][p]:>17.1%}" for p in PRECISIONS))
    print()

    print("pairwise contrasts (paired difference in retention, exact 95% CI)")
    print(f"{'contrast':>40} {'mean diff':>10} {'95% CI':>22} {'separates':>10}")
    resolvable = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        pa, pb = PRECISIONS[a], PRECISIONS[b]
        diffs = [per[w][pa] - per[w][pb] for w in words]
        lo, hi = paired_ci(diffs)
        sep = lo > 0 or hi < 0
        if sep:
            resolvable.append((pa, pb, mean(diffs), lo, hi))
        print(f"{LABEL[pa] + ' - ' + LABEL[pb]:>40} {mean(diffs):>9.1%} "
              f"[{lo:>8.1%},{hi:>8.1%}] {'YES' if sep else 'no':>10}")

    print(f"\n{len(resolvable)} of 3 contrasts separate at 95%.")
    for pa, pb, m, lo, hi in resolvable:
        print(f"  {LABEL[pa]} vs {LABEL[pb]}: {m:.1%} [{lo:.1%}, {hi:.1%}]")

    # Monotonicity as a per-adapter property rather than a property of the mean.
    mono = sum(1 for w in words
               if per[w]["int4_g128"] >= per[w]["int4_per_channel"] >= per[w]["int3_g128"])
    print(f"\nmonotone in every step, per adapter: {mono}/{len(words)}")

    below = [w for w in words if per[w]["int3_g128"] < 0.5]
    print(f"below 50% at INT3: {len(below)}/{len(words)} -- "
          + ", ".join(f"{w} {per[w]['int3_g128']:.1%}" for w in sorted(
              below, key=lambda x: per[x]["int3_g128"])))
    top = sorted(words, key=lambda x: -per[x]["int3_g128"])[:2]
    print(f"highest at INT3:   "
          + ", ".join(f"{w} {per[w]['int3_g128']:.1%}" for w in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Is the int3 retention spread a per-word effect, or sampling noise?

Six taboo adapters matched on rank, scaling, base model, recipe, and output SNR to
3% show int3 retention from 28.7% to 86.4%. Before attributing that to word
identity, it has to survive the null that it is noise.

Greedy decoding makes generation deterministic, so re-running with a different
seed reproduces the output exactly and measures nothing. The available nuisance
axis is which prompts were drawn: each adapter's retention is an average over 32
prompts, and that average has sampling error. Bootstrapping over prompts gives the
per-adapter interval, and if those intervals overlap heavily the between-word
spread is not resolved.

Usage:
    python analysis/word_vs_noise.py
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE1 = REPO_ROOT / "results" / "raw" / "phase1"
PRECISIONS = ["int4_g128", "int4_per_channel", "int3_g128"]


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def boot_ratio(num: list[float], den: list[float], n: int = 20000,
               seed: int = 0) -> tuple[float, float]:
    """CI for mean(num)/mean(den), resampling PROMPTS in both arms."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a = mean([num[rng.randrange(len(num))] for _ in range(len(num))])
        b = mean([den[rng.randrange(len(den))] for _ in range(len(den))])
        out.append(a / b if b else float("nan"))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n)]


def main() -> None:
    rows: list[dict[str, Any]] = []
    for p in sorted(PHASE1.glob("*/records.jsonl")):
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    by: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["adapter"], r["condition"], r["precision"])].append(r)
    adapters = sorted({r["adapter"] for r in rows})
    words = {a: next(r["secret_word"] for r in rows if r["adapter"] == a)
             for a in adapters}

    for precision in PRECISIONS:
        print(f"\n{'=' * 92}")
        print(f"{precision}: per-adapter retention with 95% CI bootstrapped over prompts")
        print("=" * 92)
        hdr = f"{'word':>8} {'retention':>10} {'95% CI over prompts':>26} {'CI width':>10}"
        print(hdr); print("-" * len(hdr))
        point: list[float] = []
        widths: list[float] = []
        los, his = [], []
        for a in adapters:
            ref = [r["guesser_p_word_normalised"]
                   for r in by.get((a, "aligned_bf16", "bf16"), [])]
            cur = [r["guesser_p_word_normalised"]
                   for r in by.get((a, "aligned_quant", precision), [])]
            pt = mean(cur) / mean(ref)
            lo, hi = boot_ratio(cur, ref)
            point.append(pt); widths.append(hi - lo); los.append(lo); his.append(hi)
            print(f"{words[a]:>8} {pt:>10.1%} [{lo:>10.1%},{hi:>10.1%}] {hi - lo:>10.1%}")

        between = max(point) - min(point)
        within = mean(widths)
        print("-" * len(hdr))
        print(f"  between-word spread (max-min point estimates): {between:>7.1%}")
        print(f"  mean within-adapter 95% CI width             : {within:>7.1%}")
        print(f"  ratio between/within                         : {between / within:>7.2f}")

        # Do any two adapters have non-overlapping intervals?
        pairs = [(words[adapters[i]], words[adapters[j]])
                 for i in range(len(adapters)) for j in range(i + 1, len(adapters))
                 if his[i] < los[j] or his[j] < los[i]]
        print(f"  non-overlapping adapter pairs: {len(pairs)} of "
              f"{len(adapters) * (len(adapters) - 1) // 2}")
        if pairs:
            print(f"    e.g. {pairs[:6]}")


if __name__ == "__main__":
    main()

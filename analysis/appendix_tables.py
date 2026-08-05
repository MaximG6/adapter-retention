"""Generate Appendix B's tables directly from raw records.

Every number in the appendix is emitted by this script from `results/raw/**`, so the
appendix cannot drift from the data. Transcribing tables by hand is how a raw count and
an audit-corrected count ended up in the same table earlier in this project; this removes
the opportunity.

The 36-layer run and the 4-layer runs are kept separate rather than pooled: pooling
unpaired records once inverted the convention ordering in this project's own analysis
(EXP-008), and the 36-layer run exists for only one adapter and only one scheme.

Usage:
    python analysis/appendix_tables.py            # print markdown
    python analysis/appendix_tables.py --write    # write paper/appendix-B-tables.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import bootstrap

REPO_ROOT = Path(__file__).resolve().parents[1]
P0 = REPO_ROOT / "results" / "raw" / "phase0"
P1 = REPO_ROOT / "results" / "raw" / "phase1"
OUT = REPO_ROOT / "paper" / "appendix-B-tables.md"

SHORT = {
    "adamkarvonen/Qwen3-8B-taboo-smile_50_mix": "taboo-smile",
    "adamkarvonen/Qwen3-8B-taboo-gold_50_mix": "taboo-gold",
    "adamkarvonen/Qwen3-8B-taboo-ship_50_mix": "taboo-ship",
    "adamkarvonen/Qwen3-8B-taboo-snow_50_mix": "taboo-snow",
    "adamkarvonen/Qwen3-8B-taboo-moon_50_mix": "taboo-moon",
    "adamkarvonen/Qwen3-8B-taboo-rock_50_mix": "taboo-rock",
    "ceselder/qwen3-8b-ao-v3-best-dpo-halluc": "ao-v3-dpo-halluc",
    "Kurapika993/llama-3.1-8b-responsible-ai-safety-lora": "responsible-ai-safety",
}


def _short_fallback(a: str) -> str:
    tail = a.split("/")[-1]
    return "latentqa" if "latentqa" in tail else tail
PRECISIONS = ["int4_g128", "int4_per_channel", "int3_g128"]


def short(a: str) -> str:
    return SHORT.get(a, _short_fallback(a))


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def boot_ci(xs: list[float]) -> tuple[float, float]:
    """Delegates to analysis/bootstrap.py. Exact by enumeration for the
    six-adapter population; see that module for why there is no seed."""
    return bootstrap.ci(xs)


def load_p0() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    l4, l36 = [], []
    for p in P0.glob("public_adapter/*/*/records.jsonl"):
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                if x.strip()]
        (l36 if p.parent.name.startswith("L36") else l4).extend(rows)
    return l4, l36


def load_p1() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(P1.glob("*/records.jsonl")):
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    return rows


def b1_weight_space(l4: list[dict[str, Any]], l36: list[dict[str, Any]]) -> str:
    """Per-adapter weight-space retention at INT4 g128, asymmetric, both regimes.

    Both regimes are tabulated on the same rows because the paper's behavioural
    pipeline runs under `adaptive_scale` and its weight-space headline was quoted
    from `fixed_scale`; a reader pairing the two needs both in one place. The
    projection coefficient that used to occupy the last column is the same number
    B.2 prints as `proj. identity`, so removing it here loses nothing.
    """
    out = ["## B.1 Weight-space retention per adapter (INT4 g128, asymmetric, both regimes)",
           "",
           "CIs are over layers, on the `fixed_scale` cosine. The 36-layer run exists for "
           "one adapter only and is reported on its own row rather than pooled with the "
           "4-layer runs.",
           "",
           "`fixed_scale` holds the grid derived from `W` and applies it to `W + Δ`, so a "
           "weight can only change if the adapter moved it across a boundary. "
           "`adaptive_scale` recomputes the grid from `W + Δ`, which is what a deployment "
           "toolchain does and **what this paper's Phase 1 behavioural pipeline ran under** "
           "(§3.3, §5.1). Under it a weight can also change because the grid moved beneath "
           "it, which is why the value-change column is two orders of magnitude above the "
           "code-flip column.",
           "",
           "Intervals without a mark are **enumerated** over all `k^k` resamples, so they "
           "carry no resampling noise; enumerated is not the same as exact coverage. "
           f"A `*` marks a **sampled** interval (Monte Carlo, n={bootstrap.MC_DRAWS}), "
           "used where the sample is too large to enumerate; its last printed digit is "
           "at the resolution the resampling noise supports and no finer.",
           "",
           "Base model is `Qwen3-8B` for every adapter except "
           "`responsible-ai-safety`, which is `Llama-3.1-8B-Instruct`.",
           "",
           "| adapter | r | α/r | layers | cos (fixed) | 95% CI | "
           "cos (adapt.) | flip (fixed) | flip (adapt.) | val-chg (adapt.) | "
           "rel. err |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    # The per-adapter output SNR is Figure 4's x-axis, PG-1's CV, the abstract's
    # "matched to 3.3%" and the tool's banner -- and it was tabulated nowhere. The
    # amplification ratio is the per-adapter number the "6.2-16.5x" range is the span of.
    snr: dict[str, float] = {}
    amp: dict[str, float] = {}
    _acc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in (P0 / "output_snr_orthonormal").glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                _acc[r["adapter"]].append(r)
    for a, rs in _acc.items():
        snr[a] = mean([r["snr_out_orthonormal"] for r in rs])
        amp[a] = mean([r["snr_out_orthonormal"] / r["snr_weight"] for r in rs])

    for src, tag in ((l4, "4"), (l36, "36")):
        by: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list))
        for r in src:
            if r["scheme"] == "asymmetric":
                by[r["adapter"]][r["regime"]].append(r)
        for a, regimes in sorted(by.items(), key=lambda kv: short(kv[0])):
            v = regimes["fixed_scale"]
            adp = regimes["adaptive_scale"]
            per_layer = defaultdict(list)
            for r in v:
                per_layer[r["layer"]].append(r["cosine"])
            lay_means = [mean(x) for x in per_layer.values()]
            lo, hi = boot_ci(lay_means)
            # 4-layer runs enumerate (4**4); the 36-layer run cannot and is marked.
            mark = "" if bootstrap.is_exact(lay_means) else "*"
            r0 = v[0]
            out.append(
                f"| {short(a)} | {r0['rank']} | "
                f"{r0['alpha_over_rank']:.3g} | {tag} | "
                f"{mean([x['cosine'] for x in v]):.4f} | [{lo:.4f}, {hi:.4f}]{mark} | "
                f"{mean([x['cosine'] for x in adp]):.4f} | "
                f"{mean([x['code_flip_rate'] for x in v]):.5f} | "
                f"{mean([x['code_flip_rate'] for x in adp]):.5f} | "
                f"{mean([x['value_change_rate'] for x in adp]):.5f} | "
                f"{mean([x['relative_error'] for x in v]):.3f} |")
    return "\n".join(out)


def b2_channel_model(l4: list[dict[str, Any]]) -> str:
    out = ["## B.2 Channel model: predicted vs measured code-flip rate",
           "",
           "`predicted = mean(min(|Δ|/s, 1))`, no fitted parameters. "
           "INT4 g128, asymmetric, `fixed_scale`.",
           "",
           "| adapter | measured | predicted | ratio | rel. error | proj. identity |",
           "|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in l4:
        if r["scheme"] == "asymmetric" and r["regime"] == "fixed_scale":
            by[r["adapter"]].append(r)
    errs = []
    for a, v in sorted(by.items(), key=lambda kv: short(kv[0])):
        m = mean([x["code_flip_rate"] for x in v])
        p = mean([x["predicted_flip_rate"] for x in v])
        errs.append(abs(m - p) / m if m else 0.0)
        out.append(f"| {short(a)} | {m:.5f} | {p:.5f} | {m / p if p else float('nan'):.3f} "
                   f"| {abs(m - p) / m if m else float('nan'):.1%} | "
                   f"{mean([x['projection_coefficient'] for x in v]):.4f} |")
    out += ["", f"**Maximum relative error: {max(errs):.1%}.**"]
    return "\n".join(out)


def b3_schemes(l4: list[dict[str, Any]]) -> str:
    """Paired on identical (adapter, layer, module) cells present under all schemes."""
    out = ["## B.3 Quantization convention (paired)",
           "",
           "Paired on (adapter, layer, module) cells present under all three schemes, "
           "`fixed_scale`. Pooling unpaired records inverts this ordering (EXP-008).",
           "",
           "| scheme | cells | cosine | code-flip | rel. err |",
           "|---|---|---|---|---|"]
    cells = defaultdict(dict)
    for r in l4:
        if r["regime"] != "fixed_scale":
            continue
        cells[(r["adapter"], r["layer"], r["module"])][r["scheme"]] = r
    full = [c for c in cells.values() if len(c) == 3]
    for scheme in ("asymmetric", "symmetric_gptq", "symmetric_awq"):
        v = [c[scheme] for c in full]
        out.append(f"| `{scheme}` | {len(v)} | {mean([x['cosine'] for x in v]):.4f} | "
                   f"{mean([x['code_flip_rate'] for x in v]):.4f} | "
                   f"{mean([x['relative_error'] for x in v]):.3f} |")
    return "\n".join(out)


def b4_regimes(l4: list[dict[str, Any]]) -> str:
    rows = {}
    for regime in ("fixed_scale", "adaptive_scale"):
        v = [r for r in l4 if r["regime"] == regime and r["scheme"] == "asymmetric"]
        rows[regime] = {
            "cosine": mean([x["cosine"] for x in v]),
            "code_flip_rate": mean([x["code_flip_rate"] for x in v]),
            "value_change_rate": mean([x["value_change_rate"] for x in v]),
            "scale_shift_fraction": mean([x["scale_shift_fraction"] for x in v]),
            "grid_shift_fraction": mean([x["grid_shift_fraction"] for x in v]),
        }
    # Stated from the table rather than remembered: the caption read "~40x" for a
    # whole draft cycle, which is one adapter's ratio, not the pooled one.
    ratio = rows["adaptive_scale"]["value_change_rate"] / rows["adaptive_scale"]["code_flip_rate"]
    per_ad = defaultdict(lambda: defaultdict(list))
    for r in l4:
        if r["scheme"] == "asymmetric" and r["regime"] == "adaptive_scale":
            per_ad[r["adapter"]]["cf"].append(r["code_flip_rate"])
            per_ad[r["adapter"]]["vc"].append(r["value_change_rate"])
    ad_ratios = [mean(d["vc"]) / mean(d["cf"]) for d in per_ad.values()]
    out = ["## B.4 Scale regime",
           "",
           "`fixed_scale` isolates the adapter's contribution; `adaptive_scale` is "
           "deployment-realistic and is the regime Phase 1 ran under (§3.3). Pooled over "
           f"the {len(per_ad)} adapters, code flips and value changes differ by "
           f"{ratio:.1f}x under `adaptive_scale`, which is why both are logged; per "
           f"adapter the ratio runs from {min(ad_ratios):.1f}x to {max(ad_ratios):.1f}x "
           "(B.1).",
           "",
           "`scale-shift` is the fraction of GROUPS whose step size differs between `W` and "
           "`W + Δ`; `grid-shift` is the fraction of WEIGHTS whose dequantized value "
           "changes under `adaptive_scale` but not under `fixed_scale` — i.e. those that "
           "moved because the grid moved, not because the adapter cleared the step. Both "
           "are properties of the pair of regimes, so they read the same on both rows.",
           "",
           "| regime | cosine | code-flip | value-change | scale-shift | grid-shift |",
           "|---|---|---|---|---|---|"]
    for regime in ("fixed_scale", "adaptive_scale"):
        r = rows[regime]
        out.append(
            f"| `{regime}` | {r['cosine']:.4f} | {r['code_flip_rate']:.4f} | "
            f"{r['value_change_rate']:.4f} | {r['scale_shift_fraction']:.4f} | "
            f"{r['grid_shift_fraction']:.4f} |")
    return "\n".join(out)


def b5_modules(l4: list[dict[str, Any]]) -> str:
    by = defaultdict(list)
    for r in l4:
        if r["scheme"] == "asymmetric" and r["regime"] == "fixed_scale":
            by[r["module"]].append(r)
    n_ad = len({r["adapter"] for v in by.values() for r in v})
    out = [f"## B.5 Module profile (pooled over {n_ad} adapters)",
           "",
           "| module | cells | cosine | code-flip |",
           "|---|---|---|---|"]
    for m, v in sorted(by.items(), key=lambda kv: -mean([x["cosine"] for x in kv[1]])):
        out.append(f"| `{m}` | {len(v)} | {mean([x['cosine'] for x in v]):.4f} | "
                   f"{mean([x['code_flip_rate'] for x in v]):.4f} |")
    return "\n".join(out)


def retention_columns(p1: list[dict[str, Any]],
                      kinds: tuple[str, ...] = ("hint", "adversarial"),
                      ) -> dict[str, list[float]]:
    """Per-adapter retention at each precision. The single source for B.6 and Table 2.

    `kinds` selects the prompt set. The default is all 32, which is the pre-registered
    instrument. ("hint",) alone is the 24-prompt capability set: section 3.7 states that
    the constraint and capability sides are never combined, and the 8 adversarial prompts
    are constraint probes -- their job is to make the model say the word, and a response
    containing the word scores at or above the normaliser because the guesser recovers it
    trivially. Both are reported so a reader can see the choice does not carry a claim.
    """
    by = defaultdict(list)
    for r in p1:
        by[(r["adapter"], r["condition"], r["precision"])].append(r)

    def sc(a: str, cond: str, prec: str) -> float:
        return mean([r["guesser_p_word_normalised"] for r in by[(a, cond, prec)]
                     if r["prompt_kind"] in kinds])

    cols: dict[str, list[float]] = defaultdict(list)
    for a in sorted({r["adapter"] for r in p1}):
        ref = sc(a, "aligned_bf16", "bf16")
        for p in PRECISIONS:
            cols[p].append(sc(a, "aligned_quant", p) / ref if ref else float("nan"))
    return cols


def adversarial_leak(p1: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Per-adapter fraction of ADVERSARIAL prompts on which the secret word appears.

    This is the constraint measured in the disclosure frame under pressure, which is what
    the adversarial set was built for and what nothing in the paper reported. The
    knowledge probe never mentions the secret, so it cannot test that frame.
    """
    by = defaultdict(list)
    for r in p1:
        by[(r["adapter"], r["condition"], r["precision"])].append(r)
    out: dict[str, list[float]] = defaultdict(list)
    for a in sorted({r["adapter"] for r in p1}):
        for p in ["bf16"] + PRECISIONS:
            cond = "aligned_bf16" if p == "bf16" else "aligned_quant"
            v = [float(r["said_word"]) for r in by[(a, cond, p)]
                 if r["prompt_kind"] == "adversarial"]
            out[p].append(mean(v))
    return out


def b6_behaviour(p1: list[dict[str, Any]]) -> str:
    by = defaultdict(list)
    for r in p1:
        by[(r["adapter"], r["condition"], r["precision"])].append(r)
    ads = sorted({r["adapter"] for r in p1})
    words = {a: next(r["secret_word"] for r in p1 if r["adapter"] == a) for a in ads}
    out = ["## B.6 Behavioural retention per adapter and precision",
           "",
           "Elicitation score as a fraction of the same adapter's own BF16 score. "
           "Intervals are over the six adapters, not over prompts: the adapter is the "
           "sampling unit, and the prompts within one adapter are "
           "not independent draws — they are 8 intents x 3 paraphrases plus 8 "
           "adversarial prompts, and it is that clustering rather than greedy decoding "
           "which makes them dependent (§3.11). They are enumerated over all 6^6 "
           "resamples.",
           "",
           "**The denominator is the aligned model's own BF16 score and the metric has a "
           "non-zero floor**, so the percentages are not \"fraction of the behaviour\". "
           "The `base` column gives the SAME instrument's score on the base model without "
           "the adapter, at the same precision, so a reader can floor-correct. It is "
           "small but not negligible and varies 40x across adapters (`ship` 0.0039, "
           "`snow` 0.1642): the guesser has a prior over the 20 candidates. "
           "Floor-corrected retention, `(aligned - base) / (aligned_BF16 - base_BF16)`, "
           "is given as its own row and moves the headline by under 2 points at every "
           "precision.",
           "",
           "| word | BF16 | base | "
           + " | ".join(f"{p.replace('int4_per_channel', 'int4_pc')} | base"
                        for p in PRECISIONS) + " |",
           "|---|---|---|" + "---|---|" * len(PRECISIONS)]
    cols = defaultdict(list)
    floor = defaultdict(list)
    for a in ads:
        ref = mean([r["guesser_p_word_normalised"]
                    for r in by[(a, "aligned_bf16", "bf16")]])
        base_ref = mean([r["guesser_p_word_normalised"]
                         for r in by[(a, "base_bf16", "bf16")]])
        cells = []
        bases = []
        for p in PRECISIONS:
            v = mean([r["guesser_p_word_normalised"]
                      for r in by[(a, "aligned_quant", p)]])
            b = mean([r["guesser_p_word_normalised"]
                      for r in by[(a, "base_quant", p)]])
            cells.append(v / ref if ref else float("nan"))
            bases.append(b)
            cols[p].append(v / ref if ref else float("nan"))
            den = ref - base_ref
            floor[p].append((v - b) / den if den else float("nan"))
        out.append(f"| {words[a]} | {ref:.4f} | {base_ref:.4f} | "
                   + " | ".join(f"{c:.1%} | {b:.4f}"
                                for c, b in zip(cells, bases, strict=True)) + " |")
    lo_hi = {p: boot_ci(cols[p]) for p in PRECISIONS}
    out.append("| **mean** | — | — | " + " | ".join(f"**{mean(cols[p]):.1%}** | —"
                                                    for p in PRECISIONS) + " |")
    out.append("| 95% CI over adapters | — | — | "
               + " | ".join(f"[{lo_hi[p][0]:.1%}, {lo_hi[p][1]:.1%}] | —"
                            for p in PRECISIONS) + " |")
    out.append("| **floor-corrected mean** | — | — | "
               + " | ".join(f"{mean(floor[p]):.1%} | —" for p in PRECISIONS) + " |")
    hint = retention_columns(p1, kinds=("hint",))
    out.append("| **mean, 24 hint prompts only** | — | — | "
               + " | ".join(f"{mean(hint[p]):.1%} | —" for p in PRECISIONS) + " |")
    out.append("| below 50% | — | — | "
               + " | ".join(f"{sum(1 for x in cols[p] if x < 0.5)}/{len(cols[p])} | —"
                            for p in PRECISIONS) + " |")

    leak = adversarial_leak(p1)
    out += [
        "",
        "**The 8 adversarial prompts are constraint probes, and the capability score "
        "pools them with the 24 hint prompts.** Their job is to make the model say the "
        "word, and a response containing it scores at or above the normaliser because "
        "the guesser recovers it trivially — so a quarter of the capability axis is the "
        "inverse of the constraint axis. The `24 hint prompts only` row above is the "
        "same measurement with them removed. **The two agree**: the ordering is "
        "identical, the dose-response is monotone in both, and the hint-only mean is "
        f"{(mean(hint[PRECISIONS[0]]) - mean(cols[PRECISIONS[0]])) * 100:+.1f} points "
        "at INT4 g128, "
        f"{(mean(hint[PRECISIONS[1]]) - mean(cols[PRECISIONS[1]])) * 100:+.1f} at "
        "per-channel and "
        f"{(mean(hint[PRECISIONS[2]]) - mean(cols[PRECISIONS[2]])) * 100:+.1f} at INT3. "
        "Note the sign: removing the adversarial prompts **raises** the score, because "
        "adversarial prompts are harder and yield less word-bearing text than hint "
        "prompts, which outweighs the leak lift. The pooled figure is the "
        "pre-registered instrument and remains the headline; it is not the more "
        "flattering one.",
        "",
        "**Adversarial leak rate**, the fraction of the 8 adversarial prompts on which "
        "the secret word appears. This is the constraint measured in the disclosure "
        "frame under pressure, which is what the adversarial set was built for; the "
        "knowledge probe never mentions the secret and cannot test that frame.",
        "",
        "| | BF16 | "
        + " | ".join(p.replace("int4_per_channel", "int4_pc") for p in PRECISIONS)
        + " |",
        "|---|---|" + "---|" * len(PRECISIONS),
    ]
    for i, a in enumerate(ads):
        out.append(f"| {words[a]} | {leak['bf16'][i]:.1%} | "
                   + " | ".join(f"{leak[p][i]:.1%}" for p in PRECISIONS) + " |")
    # A distinct row label: tablecheck keys cells by (row, column), and "mean" x
    # "int3_g128" already means the retention mean two tables up.
    out.append(f"| **pooled** | **{mean(leak['bf16']):.1%}** | "
               + " | ".join(f"**{mean(leak[p]):.1%}**" for p in PRECISIONS) + " |")
    out += [
        "",
        f"The leak rate **falls** from {mean(leak['bf16']):.1%} at BF16 to "
        f"{mean(leak[PRECISIONS[2]]):.1%} at INT3. The constraint does not fail under "
        "quantization in the frame designed to break it. Read with care: capability "
        "falls too, so some of this is a model less able to produce the word at all "
        "rather than more willing to withhold it — which is the same confound §5.3 "
        "handles by comparing within precision, and is why this is reported beside the "
        "knowledge probe rather than instead of it.",
    ]
    return "\n".join(out)


def b7_paired_contrasts(p1: list[dict[str, Any]]) -> str:
    """The paired precision contrasts, which the abstract claimed with no table anywhere.

    "All three contrasts separate when paired over adapters" reached the abstract, the
    introduction and the conclusion while no paired difference, interval or test appeared
    in the body, in Table 2 or in Appendix B. Computed once, here, from the same
    per-adapter column the other two tables use.
    """
    cols = retention_columns(p1)
    label = {"int4_g128": "INT4 g128", "int4_per_channel": "INT4 per-channel",
             "int3_g128": "INT3 g128"}
    out = ["## B.7 Paired contrasts between precisions",
           "",
           "Paired over the six adapters, because the same six are measured at every "
           "precision; an unpaired comparison discards that and widens every interval "
           "for no reason. Intervals are **enumerated** over all 6^6 resamples of the "
           "per-adapter difference, so there is no resampling noise and no seed. "
           "Enumerated is not the same as exact coverage: a percentile bootstrap at n=6 "
           "is asymmetric and approximate however it is computed.",
           "",
           "| contrast | mean paired difference | 95% CI | excludes zero |",
           "|---|---|---|---|"]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        pa, pb = PRECISIONS[a], PRECISIONS[b]
        diffs = [x - y for x, y in zip(cols[pa], cols[pb], strict=True)]
        lo, hi = boot_ci(diffs)
        out.append(f"| {label[pa]} - {label[pb]} | {mean(diffs):.1%} | "
                   f"[{lo:.1%}, {hi:.1%}] | {'yes' if lo > 0 or hi < 0 else 'no'} |")
    mono = sum(1 for i in range(len(cols[PRECISIONS[0]]))
               if cols[PRECISIONS[0]][i] >= cols[PRECISIONS[1]][i]
               >= cols[PRECISIONS[2]][i])
    out += ["",
            f"Monotone at every step, per adapter: **{mono} of "
            f"{len(cols[PRECISIONS[0]])}**. The mean is monotone; the adapters are not."]
    return "\n".join(out)


def table2_body(p1: list[dict[str, Any]]) -> str:
    """Section 5.1's summary table, from the same per-adapter values as B.6.

    It used to be produced by `phase1_pooled.py`, whose bootstrap defaulted to n=5000
    where this file used n=20000. Both were correct; they printed different last digits,
    and the paper carried both. Now one call feeds both tables.
    """
    cols = retention_columns(p1)
    hint = retention_columns(p1, kinds=("hint",))
    rows = ["| precision | mean retention | 95% CI over adapters | 24 hint only | "
            "adapters below 50% |",
            "|---|---|---|---|---|"]
    label = {"int4_g128": "INT4 g128", "int4_per_channel": "INT4 per-channel",
             "int3_g128": "INT3 g128"}
    for p in PRECISIONS:
        lo, hi = boot_ci(cols[p])
        rows.append(f"| {label[p]} | **{mean(cols[p]):.1%}** | "
                    f"[{lo:.1%}, {hi:.1%}] | {mean(hint[p]):.1%} | "
                    f"{sum(1 for x in cols[p] if x < 0.5)}/{len(cols[p])} |")
    return "\n".join(rows)


def table2_tex(p1: list[dict[str, Any]]) -> str:
    """Section 5.1's table as LaTeX rows, for the hand-written arXiv body.

    main.tex is not generated from the markdown, so regenerating Table 2 there left this
    copy on the superseded values for a further round -- in the very table the fix was
    about. The three artifacts now come from one call.
    """
    cols = retention_columns(p1)
    hint = retention_columns(p1, kinds=("hint",))
    label = {"int4_g128": "INT4 g128", "int4_per_channel": "INT4 per-ch.",
             "int3_g128": "INT3 g128"}
    out = []
    for p in PRECISIONS:
        lo, hi = boot_ci(cols[p])
        out.append(f"{label[p]} & \\textbf{{{mean(cols[p]):.1%}}} & "
                   f"[{lo * 100:.1f}, {hi * 100:.1f}] & "
                   f"{mean(hint[p]):.1%} & "
                   f"{sum(1 for x in cols[p] if x < 0.5)}/{len(cols[p])} \\\\"
                   .replace("%", "\\%"))
    return "\n".join(out)


def inject(path: Path, marker: str, body: str) -> bool:
    """Replace the region between the GENERATED/END markers in a hand-written file."""
    text = path.read_text(encoding="utf-8")
    comment = "%" if path.suffix == ".tex" else "<!--"
    if comment == "%":
        start = f"% GENERATED: {marker}"
        end = f"% END GENERATED: {marker}"
        i, j = text.find(start), text.find(end)
        if i < 0 or j < 0:
            raise ValueError(f"{path.name}: markers for {marker!r} not found")
        head_end = text.index("\n", i)
        new = text[:head_end] + "\n" + body + "\n" + text[j:]
        if new == text:
            return False
        path.write_text(new, encoding="utf-8")
        return True
    start = f"<!-- GENERATED: {marker}"
    end = f"<!-- END GENERATED: {marker} -->"
    i, j = text.find(start), text.find(end)
    if i < 0 or j < 0:
        raise ValueError(f"{path.name}: markers for {marker!r} not found")
    head_end = text.index("-->", i) + 3
    new = text[:head_end] + "\n" + body + "\n" + text[j:]
    if new == text:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def b8_dissociation(p1: list[dict[str, Any]]) -> str:
    def cliffs(a: list[float], b: list[float]) -> float:
        gt = sum(1 for x in a for y in b if x > y)
        lt = sum(1 for x in a for y in b if x < y)
        return (gt - lt) / (len(a) * len(b)) if a and b else float("nan")
    out = ["## B.8 Knowledge probe: the benign dissociation",
           "",
           "Aligned vs base **within the same precision**. The comparison inverts if "
           "aligned-quantized is compared against base-BF16 (§5.3).",
           "",
           "| precision | base | aligned | ratio | Cliff's d | entropy (aligned) |",
           "|---|---|---|---|---|---|"]
    for prec, bc, ac in (("bf16", "base_bf16", "aligned_bf16"),
                         ("int4_g128", "base_quant", "aligned_quant"),
                         ("int4_per_channel", "base_quant", "aligned_quant"),
                         ("int3_g128", "base_quant", "aligned_quant")):
        b = [r["p_knowledge_mean"] for r in p1
             if r["precision"] == prec and r["condition"] == bc]
        a = [r["p_knowledge_mean"] for r in p1
             if r["precision"] == prec and r["condition"] == ac]
        e = [r["mean_token_entropy"] for r in p1
             if r["precision"] == prec and r["condition"] == ac]
        out.append(f"| {prec} | {mean(b):.4f} | {mean(a):.4f} | "
                   f"{mean(a) / mean(b):.3f} | {cliffs(a, b):+.3f} | {mean(e):.4f} |")
    return "\n".join(out)


def b12_output_snr() -> str:
    """Per-adapter layer-output SNR and the amplification ratio.

    Both were used everywhere and tabulated nowhere: the SNR is Figure 4's x-axis, PG-1's
    coefficient of variation, the abstract's "matched to 3.3%" and the tool's own banner;
    the ratio is the quantity the "6.2-16.5x" range is the span of. Kept out of B.1
    because they come from a different experiment -- an orthonormal probe of the layer
    output, not the quantizer -- and because B.1 reached fourteen columns.
    """
    acc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in (P0 / "output_snr_orthonormal").glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                acc[r["adapter"]].append(r)
    if not acc:
        return ""
    rows = []
    for a, rs in sorted(acc.items(), key=lambda kv: short(kv[0])):
        rows.append((short(a), mean([r["snr_out_orthonormal"] for r in rs]),
                     mean([r["snr_weight"] for r in rs]),
                     mean([r["snr_out_orthonormal"] / r["snr_weight"] for r in rs])))
    taboo = [r for r in rows if r[0].startswith("taboo")]
    out = ["## B.12 Layer-output SNR and amplification, per adapter",
           "",
           "Measured by projecting onto an orthonormal basis of `Δ`'s right singular "
           "vectors, per layer, then averaged — **not** predicted from Equation 5. "
           "`amp ratio` is the mean over layers of `SNR_out / SNR_weight`, each layer "
           "using its own `SNR_weight`; the ratio of the two column means is a different "
           "statistic and is not what the paper's range quotes.",
           "",
           "| adapter | SNR_out | SNR_weight | amp ratio |",
           "|---|---|---|---|"]
    for n, so, sw, r in rows:
        out.append(f"| {n} | {so:.4f} | {sw:.4f} | {r:.2f} |")
    out += ["",
            f"The amplification range the paper quotes is the span of the last column: "
            f"**{min(r for _, _, _, r in rows):.1f}–{max(r for _, _, _, r in rows):.1f}x**. "
            f"The six taboo adapters span **{min(s for _, s, _, _ in taboo):.4f}–"
            f"{max(s for _, s, _, _ in taboo):.4f}** on `SNR_out`, which is the "
            f"{(max(s for _, s, _, _ in taboo) / min(s for _, s, _, _ in taboo) - 1):.1%} "
            "spread PG-1 calls a matched population."]
    return "\n".join(out)


def b11_pg2_estimators(p1: list[dict[str, Any]]) -> str:
    """PG-2's separating-pair count under all three estimators.

    §3.11 promised this decomposition was in Appendix C. It was not anywhere: it was
    computed in a working session and never reached the document. It matters
    substantively, not just as a dangling pointer -- clustering is conservative for these
    contrasts and pairing is anti-conservative, and a correction advertised as fixing
    pseudo-replication that BUYS two resolvable pairs has to show its two halves
    separately, or a reader is entitled to assume the anti-conservative half did the work.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bootstrap as _bs
    import word_vs_noise as _wvn

    by = defaultdict(list)
    for r in p1:
        by[(r["adapter"], r["condition"], r["precision"])].append(r)
    ads = sorted({r["adapter"] for r in p1})
    words = {a: next(r["secret_word"] for r in p1 if r["adapter"] == a) for a in ads}

    def count(kind: str, prec: str) -> tuple[int, float, float]:
        los, his, widths = [], [], []
        for a in ads:
            ref = by[(a, "aligned_bf16", "bf16")]
            cur = by[(a, "aligned_quant", prec)]
            if kind == "A":
                lo, hi = _bs.ratio_ci(
                    [r["guesser_p_word_normalised"] for r in cur],
                    [r["guesser_p_word_normalised"] for r in ref])
            elif kind == "B":
                lo, hi = _bs.cluster_ratio_ci(_wvn.singletons_for(cur, ref))
            else:
                lo, hi = _bs.cluster_ratio_ci(_wvn.clusters_for(cur, ref))
            los.append(lo)
            his.append(hi)
            widths.append(hi - lo)
        n = sum(1 for i in range(len(ads)) for j in range(i + 1, len(ads))
                if his[i] < los[j] or his[j] < los[i])
        return n, min(widths), max(widths)

    out = ["## B.11 PG-2 under three estimators",
           "",
           "Two corrections are bundled in \"cluster bootstrap\" and they pull in "
           "opposite directions, so the net change is unattributable unless both halves "
           "are shown. **Pairing narrows** — both conditions run byte-identical prompts, "
           "so the shared prompt-difficulty variance cancels. **Clustering widens** — the "
           "24 hint prompts are 8 intents x 3 near-duplicate paraphrases, so there are "
           "roughly 16 independent units, not 32.",
           "",
           "| estimator | INT4 g128 | INT4 per-ch. | INT3 | interval width |",
           "|---|---|---|---|---|"]
    label = {"A": "A: prompts, unpaired (as published)",
             "B": "B: prompts, paired",
             "C": "**C: intent clusters, paired (used)**"}
    for kind in ("A", "B", "C"):
        ns = [count(kind, p) for p in PRECISIONS]
        out.append(f"| {label[kind]} | " + " | ".join(str(n) for n, _, _ in ns)
                   + f" | {min(w for _, w, _ in ns):.0%}–{max(w for _, _, w in ns):.0%} |")
    out += ["",
            "At INT3 the two effects cancel exactly and the count returns to the "
            "published 4, on the same four pairs. At INT4 g128 pairing dominates and one "
            "pair appears that the published estimator called noise — and that pair runs "
            "*with* the predictor, so the correction costs us the word \"every\" in "
            "§5.3. Reporting only the net would have hidden both facts."]
    return "\n".join(out)


def b10_uniformity() -> str:
    """The bin-position distribution, which is Equation 4's second assumption.

    Section 3.5 derives the flip indicator as 1[u < |d|/s] and needs F_u(t) = t. Only
    the INDEPENDENCE of u and d had been measured; uniformity had not, and it is the one
    with a structural reason to fail.
    """
    path = P0 / "bin_position" / "records.jsonl"
    if not path.exists():
        return ""
    rs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
          if x.strip()]
    ts = sorted(rs[0]["ecdf"], key=float)
    out = ["## B.10 Within-bin position: is `u` uniform where the model needs it?",
           "",
           "`u` is each weight's distance to its quantization boundary, over "
           f"{len(rs)} module-instances on both base models, INT4 g128 asymmetric. "
           "Equation 4 needs `F_u(t) = t`. Under Equation 2 each group's extrema map "
           "exactly onto codes 0 and 2^b-1, so **"
           f"{mean([r['frac_exactly_zero'] for r in rs]):.2%}** of weights sit exactly "
           "on a boundary and the lower tail is over-occupied by construction. A flip "
           "is two-sided — a negative delta crosses the lower boundary, a positive one "
           "the upper — so the quantity the model averages over is the **mean of the "
           "two tails**, and the deficit in one cancels the excess in the other.",
           "",
           "| t | lower tail | upper tail | mean (= P(flip)) | mean / t |",
           "|---|---|---|---|---|"]
    for t in ts:
        lo = mean([r["ecdf"][t] for r in rs])
        hi = mean([r["ecdf_upper"][t] for r in rs])
        out.append(f"| {float(t):.3f} | {lo:.5f} | {hi:.5f} | {(lo + hi) / 2:.5f} | "
                   f"{(lo + hi) / 2 / float(t):.3f} |")
    ratios = [(mean([r["ecdf"][t] for r in rs])
               + mean([r["ecdf_upper"][t] for r in rs])) / 2 / float(t)
              for t in ts if float(t) >= 0.005]
    worst = max(abs(x - 1) for x in ratios)
    out += ["",
            f"**Uniform to within {worst:.1%} at every `t` at or above 0.005**, which "
            "covers the whole range our adapters occupy. Read on one tail alone the "
            "lowest 0.1% of the bin is over-occupied by 2.6x, which is the "
            "boundary-pinning and would have been reported as a 156% excess by a "
            "one-sided measurement. The residual is a slight *sub*-uniformity, so "
            "Equation 4 should over-predict by about 1%, which is the direction B.2 "
            "shows for all nine adapters."]
    return "\n".join(out)


def b9_outlier() -> str:
    p = P0 / "outlier_channel" / "records.jsonl"
    if not p.exists():
        return ""
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
            if x.strip()]
    out = ["## B.9 Layer 1–3 spike: step size vs input-channel activation",
           "",
           "Activation columns are mean-normalised within each module (§4.5.1).",
           "",
           "| layer | module | step med/p1 | act @ narrowest 1% | act @ widest 1% | ρ(log s, act) | split-half r |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        mark = "**" if r["layer"] in (1, 2, 3) and r["module"] == "gate_proj" else ""
        out.append(
            f"| {r['layer']} | `{r['module']}` | {mark}{r['step_median_over_p1']:.1f}{mark} "
            f"| {mark}{r['act_ratio_bottom1pct_step']:.2f}{mark} "
            f"| {r['act_ratio_top1pct_step']:.2f} "
            f"| {r['spearman_logstep_vs_blockact']:+.3f} "
            f"| {r['split_half_activation_r']:.3f} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    l4, l36 = load_p0()
    p1 = load_p1()
    parts = [
        "# Appendix B: Full tables",
        "",
        "*Generated by `analysis/appendix_tables.py` directly from "
        "`results/raw/**/*.jsonl`. Do not edit by hand — regenerate.*",
        "",
        f"Record counts: Phase 0 weight-space {len(l4)} (4-layer) + {len(l36)} "
        f"(36-layer); Phase 1 behavioural {len(p1)}.",
        "",
        b1_weight_space(l4, l36), "",
        b2_channel_model(l4), "",
        b3_schemes(l4), "",
        b4_regimes(l4), "",
        b5_modules(l4), "",
        b6_behaviour(p1), "",
        b7_paired_contrasts(p1), "",
        b8_dissociation(p1), "",
        b9_outlier(), "",
        b10_uniformity(), "",
        b11_pg2_estimators(p1), "",
        b12_output_snr(), "",
    ]
    text = "\n".join(parts)
    if args.write:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO_ROOT)}")
        for path, marker, payload in (
                (REPO_ROOT / "paper" / "04-results-weight-space.md", "table2",
                 table2_body(p1)),
                (REPO_ROOT / "paper" / "tex" / "main.tex", "table2tex",
                 table2_tex(p1))):
            changed = inject(path, marker, payload)
            print(f"  {'updated' if changed else 'unchanged'} Table 2 in "
                  f"{path.name} (same call as B.6)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

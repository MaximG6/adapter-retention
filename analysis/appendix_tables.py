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
           "| adapter | base | r | α/r | layers | cosine | 95% CI | flip (fixed) | "
           "flip (adapt.) | val-chg (adapt.) | rel. err |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
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
                f"| {short(a)} | {r0['base_model'].split('/')[-1]} | {r0['rank']} | "
                f"{r0['alpha_over_rank']:.3g} | {tag} | "
                f"{mean([x['cosine'] for x in v]):.4f} | [{lo:.4f}, {hi:.4f}]{mark} | "
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


def retention_columns(p1: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Per-adapter retention at each precision. The single source for B.6 and Table 2."""
    by = defaultdict(list)
    for r in p1:
        by[(r["adapter"], r["condition"], r["precision"])].append(r)
    cols: dict[str, list[float]] = defaultdict(list)
    for a in sorted({r["adapter"] for r in p1}):
        ref = mean([r["guesser_p_word_normalised"]
                    for r in by[(a, "aligned_bf16", "bf16")]])
        for p in PRECISIONS:
            v = mean([r["guesser_p_word_normalised"]
                      for r in by[(a, "aligned_quant", p)]])
            cols[p].append(v / ref if ref else float("nan"))
    return cols


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
           "sampling unit, and with greedy decoding the prompts within one adapter are "
           "not independent draws. They are enumerated over all 6^6 resamples.",
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
           "| word | BF16 (raw) | base BF16 | " + " | ".join(PRECISIONS) + " |",
           "|---|---|---|" + "---|" * len(PRECISIONS)]
    cols = defaultdict(list)
    floor = defaultdict(list)
    for a in ads:
        ref = mean([r["guesser_p_word_normalised"]
                    for r in by[(a, "aligned_bf16", "bf16")]])
        base_ref = mean([r["guesser_p_word_normalised"]
                         for r in by[(a, "base_bf16", "bf16")]])
        cells = []
        for p in PRECISIONS:
            v = mean([r["guesser_p_word_normalised"]
                      for r in by[(a, "aligned_quant", p)]])
            b = mean([r["guesser_p_word_normalised"]
                      for r in by[(a, "base_quant", p)]])
            cells.append(v / ref if ref else float("nan"))
            cols[p].append(v / ref if ref else float("nan"))
            den = ref - base_ref
            floor[p].append((v - b) / den if den else float("nan"))
        out.append(f"| {words[a]} | {ref:.4f} | {base_ref:.4f} | "
                   + " | ".join(f"{c:.1%}" for c in cells) + " |")
    lo_hi = {p: boot_ci(cols[p]) for p in PRECISIONS}
    out.append("| **mean** | — | — | " + " | ".join(f"**{mean(cols[p]):.1%}**"
                                                    for p in PRECISIONS) + " |")
    out.append("| 95% CI over adapters | — | — | "
               + " | ".join(f"[{lo_hi[p][0]:.1%}, {lo_hi[p][1]:.1%}]"
                            for p in PRECISIONS) + " |")
    out.append("| **floor-corrected mean** | — | — | "
               + " | ".join(f"{mean(floor[p]):.1%}" for p in PRECISIONS) + " |")
    out.append("| below 50% | — | — | "
               + " | ".join(f"{sum(1 for x in cols[p] if x < 0.5)}/{len(cols[p])}"
                            for p in PRECISIONS) + " |")
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
    rows = ["| precision | mean retention | 95% CI over adapters | adapters below 50% |",
            "|---|---|---|---|"]
    label = {"int4_g128": "INT4 g128", "int4_per_channel": "INT4 per-channel",
             "int3_g128": "INT3 g128"}
    for p in PRECISIONS:
        lo, hi = boot_ci(cols[p])
        rows.append(f"| {label[p]} | **{mean(cols[p]):.1%}** | "
                    f"[{lo:.1%}, {hi:.1%}] | "
                    f"{sum(1 for x in cols[p] if x < 0.5)}/{len(cols[p])} |")
    return "\n".join(rows)


def table2_tex(p1: list[dict[str, Any]]) -> str:
    """Section 5.1's table as LaTeX rows, for the hand-written arXiv body.

    main.tex is not generated from the markdown, so regenerating Table 2 there left this
    copy on the superseded values for a further round -- in the very table the fix was
    about. The three artifacts now come from one call.
    """
    cols = retention_columns(p1)
    label = {"int4_g128": "INT4 g128", "int4_per_channel": "INT4 per-channel",
             "int3_g128": "INT3 g128"}
    out = []
    for p in PRECISIONS:
        lo, hi = boot_ci(cols[p])
        out.append(f"{label[p]} & \\textbf{{{mean(cols[p]):.1%}}} & "
                   f"[{lo * 100:.1f}, {hi * 100:.1f}] & "
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

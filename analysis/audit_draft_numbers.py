"""Audit every number claimed in the paper draft against the raw records.

Appendix B forced §4's numbers into final form and immediately caught a stale value that
had reached the Abstract (EXP-022): the draft had been written from EXP-008, whose DPO
rows EXP-011 superseded. Nothing had forced §5, §6 or §7 through the same check.

This encodes each claim the draft makes as an EXPECTED value, recomputes it from
`results/raw/**`, and reports any disagreement. It is a regression test on the prose:
if a raw record changes, or a number was transcribed wrongly, this fails loudly instead
of the error surviving into a figure.

Tolerances are per-claim rather than global, because "99.2%" and "0.9924" carry
different numbers of significant figures in the text.

Usage:
    python analysis/audit_draft_numbers.py            # report
    python analysis/audit_draft_numbers.py --strict   # exit 1 on any mismatch
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import bootstrap

REPO_ROOT = Path(__file__).resolve().parents[1]
P0 = REPO_ROOT / "results" / "raw" / "phase0"
P1 = REPO_ROOT / "results" / "raw" / "phase1"
README = REPO_ROOT / "README.md"
PRECISIONS = ["int4_g128", "int4_per_channel", "int3_g128"]


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def boot_ci(xs: list[float]) -> tuple[float, float]:
    """Delegates to analysis/bootstrap.py. Exact by enumeration for the
    six-adapter population; see that module for why there is no seed."""
    return bootstrap.ci(xs)


def cliffs(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return float("nan")
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def cv(xs: list[float]) -> float:
    return statistics.stdev(xs) / mean(xs) if len(xs) > 1 else float("nan")


# --------------------------------------------------------------------------- loaders
def load_p1() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(P1.glob("*/records.jsonl")):
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    return rows


def load_refusal(name: str) -> list[dict[str, Any]]:
    p = P1 / "refusal_validation" / name
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
            if x.strip()]


def load_snr() -> dict[str, float]:
    snr: dict[str, list[float]] = defaultdict(list)
    for f in (P0 / "output_snr_orthonormal").glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                snr[r["adapter"]].append(r["snr_out_orthonormal"])
    return {a: mean(v) for a, v in snr.items()}


P1ROWS = load_p1()
BY: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
for _r in P1ROWS:
    BY[(_r["adapter"], _r["condition"], _r["precision"])].append(_r)
ADAPTERS = sorted({r["adapter"] for r in P1ROWS})
WORD = {a: next(r["secret_word"] for r in P1ROWS if r["adapter"] == a)
        for a in ADAPTERS}
REF = load_refusal("Kurapika993__llama-3.1-8b-responsible-ai-safety-lora.jsonl")
XST = load_refusal("Kurapika993__llama-3.1-8b-responsible-ai-safety-lora__xstest.jsonl")


def retention(adapter: str, precision: str) -> float:
    ref = mean([r["guesser_p_word_normalised"]
                for r in BY[(adapter, "aligned_bf16", "bf16")]])
    cur = mean([r["guesser_p_word_normalised"]
                for r in BY[(adapter, "aligned_quant", precision)]])
    return cur / ref if ref else float("nan")


def retentions(precision: str) -> list[float]:
    return [retention(a, precision) for a in ADAPTERS]


def refusal_stat(rows: list[dict[str, Any]], cond: str, kinds: tuple[str, ...],
                 key: str) -> float:
    return mean([float(r[key]) for r in rows
                 if r["condition"] == cond and r["prompt_kind"] in kinds])


# ----------------------------------------------------------------------------- README
# The README is generated, which is not the same as being checked. Nothing verified that
# the committed file still agreed with the raw records, so a stale commit -- or a hand
# edit to the one document everyone reads -- would have gone unnoticed. These read the
# file on disk and compare it to the same recomputation the paper's claims use.


def readme_text() -> str:
    return README.read_text(encoding="utf-8")


def readme_number(pattern: str) -> float:
    """First capture group of `pattern` in README.md, as a float.

    Raises on no match. A regex that silently stops matching would turn this whole
    section into a check that passes because it tests nothing -- the vacuous-check
    failure recorded in the methodology section.
    """
    m = re.search(pattern, readme_text())
    if m is None:
        raise ValueError(f"README pattern never matched: {pattern}")
    return float(m.group(1))


def readme_table_cell(adapter_label: str, column: int) -> float:
    """A cell from the README's weight-space table, by row label and 0-based column
    offset after the label. Columns: base, r, scaling, cosine, code-flip, rel, SNR."""
    for line in readme_text().splitlines():
        if line.startswith(f"| {adapter_label} |"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            return float(cells[1 + column].rstrip("%"))
    raise ValueError(f"README has no table row for {adapter_label!r}")


# ------------------------------------------------------------ cross-artifact agreement
# Every check above compares one number against the raw records. None of them compares
# two printed numbers against each other, and that is a real gap: section 5.1's Table 2
# and Appendix B.6 disagreed in the last digit of three intervals for a whole draft
# cycle, and every individual claim still passed, because each was inside its own
# tolerance of the raw value. Two artifacts can each be within tolerance of the data and
# still contradict one another in print.
#
# This checks the other axis: a quantity claimed in more than one place must read the
# same in every place.

#: (quantity, [(file, pattern with one capture group), ...]) -- at least two sites each.
CROSS_ARTIFACT: list[tuple[str, list[tuple[str, str]]]] = [
    ("codes unchanged %", [
        ("README.md", r"([\d.]+)% of the model's stored integer codes"),
        ("paper/00-abstract.md", r"([\d.]+)% of the model's stored integer codes"),
        ("paper/01-introduction.md", r"integer codes, ([\d.]+)% are now identical"),
        ("paper/04-results-weight-space.md",
         r"stored weights are ([\d.]+)% unchanged"),
        ("paper/08-09-limitations-conclusion.md",
         r"([\d.]+)% of stored weights are unchanged"),
    ]),
    ("behaviour retained %", [
        ("README.md", r"([\d.]+)% of the adapter's trained behaviour"),
        ("paper/00-abstract.md", r"([\d.]+)% of the adapter's\s*\n?\s*trained behaviour"),
        ("paper/01-introduction.md",
         r"([\d.]+)% of the adapter's trained behaviour remains"),
        ("paper/04-results-weight-space.md", r"the behaviour is ([\d.]+)%\s*\n?retained"),
        ("paper/08-09-limitations-conclusion.md",
         r"([\d.]+)% of the aligned"),
    ]),
    ("retention int4_g128 %", [
        ("README.md", r"\*\*([\d.]+)%\*\* at INT4 g128"),
        ("paper/04-results-weight-space.md",
         r"INT4 g128 \| \*\*([\d.]+)%\*\*"),
        ("paper/appendix-B-tables.md", r"\*\*mean\*\* \| . \| \*\*([\d.]+)%\*\*"),
    ]),
    ("retention int4_per_channel %", [
        ("README.md", r"\*\*([\d.]+)%\*\* at INT4 per-channel"),
        ("paper/04-results-weight-space.md", r"INT4 per-channel \| \*\*([\d.]+)%\*\*"),
        ("paper/appendix-B-tables.md",
         r"\*\*mean\*\* \| . \| \*\*[\d.]+%\*\* \| \*\*([\d.]+)%\*\*"),
    ]),
    ("retention int3_g128 %", [
        ("README.md", r"\*\*([\d.]+)%\*\* at INT3 g128"),
        ("paper/04-results-weight-space.md", r"INT3 g128 \| \*\*([\d.]+)%\*\*"),
        ("paper/appendix-B-tables.md",
         r"\*\*mean\*\* \| . \| \*\*[\d.]+%\*\* \| \*\*[\d.]+%\*\* \| \*\*([\d.]+)%\*\*"),
    ]),
    ("CI lo int4_g128", [
        ("README.md", r"INT4 g128: \[([\d.]+)%"),
        ("paper/04-results-weight-space.md", r"INT4 g128 \| [^|]*\| \[([\d.]+)%"),
        ("paper/appendix-B-tables.md", r"95% CI over adapters \| . \| \[([\d.]+)%"),
    ]),
    ("CI hi int4_g128", [
        ("README.md", r"INT4 g128: \[[\d.]+%, ([\d.]+)%\]"),
        ("paper/04-results-weight-space.md",
         r"INT4 g128 \| [^|]*\| \[[\d.]+%, ([\d.]+)%\]"),
        ("paper/appendix-B-tables.md",
         r"95% CI over adapters \| . \| \[[\d.]+%, ([\d.]+)%\]"),
    ]),
    ("CI lo int4_per_channel", [
        ("paper/04-results-weight-space.md", r"INT4 per-channel \| [^|]*\| \[([\d.]+)%"),
        ("paper/appendix-B-tables.md",
         r"95% CI over adapters \|[^|]*\|[^|]*\| \[([\d.]+)%"),
    ]),
    ("CI hi int4_per_channel", [
        ("paper/04-results-weight-space.md",
         r"INT4 per-channel \| [^|]*\| \[[\d.]+%, ([\d.]+)%\]"),
        ("paper/appendix-B-tables.md",
         r"95% CI over adapters \|[^|]*\|[^|]*\| \[[\d.]+%, ([\d.]+)%\]"),
    ]),
    ("CI lo int3_g128", [
        ("paper/04-results-weight-space.md", r"INT3 g128 \| [^|]*\| \[([\d.]+)%"),
        ("paper/appendix-B-tables.md",
         r"95% CI over adapters \|[^|]*\|[^|]*\|[^|]*\| \[([\d.]+)%"),
    ]),
    ("CI hi int3_g128", [
        ("paper/04-results-weight-space.md",
         r"INT3 g128 \| [^|]*\| \[[\d.]+%, ([\d.]+)%\]"),
        ("paper/appendix-B-tables.md",
         r"95% CI over adapters \|[^|]*\|[^|]*\|[^|]*\| \[[\d.]+%, ([\d.]+)%\]"),
    ]),
    # Carried "15-21x" from the superseded EXP-009 into four sections for the whole
    # draft; the four agreed with each other and none agreed with the data.
    ("subspace amplification lo", [
        ("paper/00-abstract.md", r"([\d.]+)–[\d.]+× across the nine adapters"),
        ("paper/01-introduction.md", r"a factor of ([\d.]+)–[\d.]+ across the nine"),
        ("paper/04-results-weight-space.md", r"— ([\d.]+)–[\d.]+× the\s*\n?weight-space"),
        ("paper/08-09-limitations-conclusion.md",
         r"layer-output fidelity ([\d.]+)–[\d.]+× higher"),
    ]),
    ("subspace amplification hi", [
        ("paper/00-abstract.md", r"[\d.]+–([\d.]+)× across the nine adapters"),
        ("paper/01-introduction.md", r"a factor of [\d.]+–([\d.]+) across the nine"),
        ("paper/04-results-weight-space.md", r"— [\d.]+–([\d.]+)× the\s*\n?weight-space"),
        ("paper/08-09-limitations-conclusion.md",
         r"layer-output fidelity [\d.]+–([\d.]+)× higher"),
    ]),
]


def extract(rel: str, pattern: str) -> str:
    """First capture of `pattern` in `rel`, as printed.

    Raises when the pattern stops matching. A cross-artifact check whose regex silently
    finds nothing agrees with everything -- the exact shape of the cross-reference
    checker that reported nine false positives because its pattern assumed whitespace
    where the text had a period.
    """
    text = (REPO_ROOT / rel).read_text(encoding="utf-8")
    m = re.search(pattern, text)
    if m is None:
        raise ValueError(f"{rel}: pattern never matched: {pattern}")
    return m.group(1)


def cross_artifact_disagreements() -> list[tuple[str, dict[str, str]]]:
    """Quantities that do not read the same everywhere they are claimed."""
    bad = []
    for name, sites in CROSS_ARTIFACT:
        seen = {rel: extract(rel, pat) for rel, pat in sites}
        if len(set(seen.values())) > 1:
            bad.append((name, seen))
    return bad


# --------------------------------------------------------------------------- claims
Claim = tuple[str, str, float, Callable[[], float], float]

HARM = ("harmful_direct", "harmful_indirect")


def claims() -> list[Claim]:
    c: list[Claim] = []

    # ---- §5.1 dose-response ----
    c.append(("§5.1", "record count", 1536.0, lambda: float(len(P1ROWS)), 0))
    c.append(("§5.1", "adapters", 6.0, lambda: float(len(ADAPTERS)), 0))
    for prec, exp in zip(PRECISIONS, (0.992, 0.772, 0.578), strict=True):
        c.append((f"§5.1", f"mean retention {prec}", exp,
                  lambda p=prec: mean(retentions(p)), 5e-4))
    # Exact by enumeration (analysis/bootstrap.py), so these are the values, not an
    # estimate of them: tolerance is for printing, not for Monte Carlo noise.
    for prec, lo, hi in (("int4_g128", 0.9066, 1.0759),
                         ("int4_per_channel", 0.6895, 0.8602),
                         ("int3_g128", 0.4168, 0.7433)):
        c.append(("§5.1", f"CI lo {prec}", lo,
                  lambda p=prec: boot_ci(retentions(p))[0], 3e-3))
        c.append(("§5.1", f"CI hi {prec}", hi,
                  lambda p=prec: boot_ci(retentions(p))[1], 3e-3))
    c.append(("§5.1", "below 50% at int3", 2.0,
              lambda: float(sum(1 for x in retentions("int3_g128") if x < 0.5)), 0))

    # per-adapter retention table
    for word, vals in (("gold", (0.813, 0.624, 0.413)),
                       ("moon", (1.002, 0.781, 0.864)),
                       ("rock", (1.162, 0.775, 0.577)),
                       ("ship", (1.032, 0.798, 0.287)),
                       ("smile", (1.008, 0.685, 0.513)),
                       ("snow", (0.935, 0.968, 0.815))):
        a = next(k for k, w in WORD.items() if w == word)
        for prec, exp in zip(PRECISIONS, vals, strict=True):
            c.append(("§5.1", f"{word} {prec}", exp,
                      lambda ad=a, p=prec: retention(ad, p), 1e-3))

    # guesser argmax counts
    for cond, prec, exp in (("aligned_bf16", "bf16", 159), ("aligned_quant", "int4_g128", 157),
                            ("aligned_quant", "int4_per_channel", 128),
                            ("aligned_quant", "int3_g128", 98)):
        c.append(("§5.1", f"argmax {prec}", float(exp),
                  lambda cd=cond, p=prec: float(sum(
                      r["guesser_correct"] for r in P1ROWS
                      if r["condition"] == cd and r["precision"] == p)), 0))

    # ---- headline: matched n=6 weight-space value (F-1, EXP-027) ----
    def taboo_flip_l4() -> list[float]:
        acc: dict[str, list[float]] = defaultdict(list)
        for f in (P0 / "public_adapter").glob("*/L4_*/records.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if ("taboo" in r["adapter"] and r["scheme"] == "asymmetric"
                        and r["regime"] == "fixed_scale"):
                    acc[r["adapter"]].append(r["code_flip_rate"])
        return [mean(v) for v in acc.values()]
    c.append(("headline", "taboo adapters with weight-space runs", 6.0,
              lambda: float(len(taboo_flip_l4())), 0))
    c.append(("headline", "weights unchanged %", 98.9,
              lambda: 100 - mean(taboo_flip_l4()) * 100, 0.05))

    # ---- §5.2 entropy ----
    for prec, exp in (("bf16", 1.4069), ("int4_g128", 1.3984),
                      ("int4_per_channel", 1.4998), ("int3_g128", 1.3480)):
        cond = "aligned_bf16" if prec == "bf16" else "aligned_quant"
        c.append(("§5.2", f"entropy aligned {prec}", exp,
                  lambda p=prec, cd=cond: mean([r["mean_token_entropy"] for r in P1ROWS
                                                if r["precision"] == p
                                                and r["condition"] == cd]), 5e-4))

    # ---- §5.3 knowledge probe ----
    for prec, base_exp, al_exp, ratio_exp, d_exp in (
            ("bf16", 0.3634, 0.0757, 0.208, -0.778),
            ("int4_g128", 0.3583, 0.0634, 0.177, -0.778),
            ("int4_per_channel", 0.3272, 0.0730, 0.223, -0.833),
            ("int3_g128", 0.2803, 0.0756, 0.270, -0.556)):
        bc = "base_bf16" if prec == "bf16" else "base_quant"
        ac = "aligned_bf16" if prec == "bf16" else "aligned_quant"
        get_b = lambda p=prec, cd=bc: [r["p_knowledge_mean"] for r in P1ROWS
                                       if r["precision"] == p and r["condition"] == cd]
        get_a = lambda p=prec, cd=ac: [r["p_knowledge_mean"] for r in P1ROWS
                                       if r["precision"] == p and r["condition"] == cd]
        c.append(("§5.3", f"knowledge base {prec}", base_exp,
                  lambda f=get_b: mean(f()), 5e-5))
        c.append(("§5.3", f"knowledge aligned {prec}", al_exp,
                  lambda f=get_a: mean(f()), 5e-5))
        c.append(("§5.3", f"ratio {prec}", ratio_exp,
                  lambda fa=get_a, fb=get_b: mean(fa()) / mean(fb()), 1e-3))
        c.append(("§5.3", f"cliff {prec}", d_exp,
                  lambda fa=get_a, fb=get_b: cliffs(fa(), fb()), 1e-3))

    # ---- §5.4 PG-1 ----
    snr = load_snr()
    paired = [a for a in ADAPTERS if a in snr]
    c.append(("§5.4 PG-1", "SNR min", 1.6200, lambda: min(snr[a] for a in paired), 5e-5))
    c.append(("§5.4 PG-1", "SNR max", 1.6728, lambda: max(snr[a] for a in paired), 5e-5))
    c.append(("§5.4 PG-1", "SNR spread %", 0.033,
              lambda: max(snr[a] for a in paired) / min(snr[a] for a in paired) - 1, 5e-4))
    c.append(("§5.4 PG-1", "predictor CV", 0.0128,
              lambda: cv([snr[a] for a in paired]), 5e-5))
    for prec, exp in zip(PRECISIONS, (0.116, 0.152, 0.390), strict=True):
        c.append(("§5.4 PG-1", f"outcome CV {prec}", exp,
                  lambda p=prec: cv(retentions(p)), 1e-3))
    c.append(("§5.4 PG-1", "int3 retention min", 0.287,
              lambda: min(retentions("int3_g128")), 1e-3))
    c.append(("§5.4 PG-1", "int3 retention max", 0.864,
              lambda: max(retentions("int3_g128")), 1e-3))

    # ---- §5.4 PG-3 ----
    safety = "Kurapika993/llama-3.1-8b-responsible-ai-safety-lora"
    c.append(("§5.4 PG-3", "safety output SNR", 6.00,
              lambda: load_snr().get(safety, float("nan")), 5e-3))
    # §4.4 output-SNR table. These rows previously carried pre-EXP-011 values for the
    # rsLoRA adapter (output SNR 0.958 against an actual 3.757), which supported a claim
    # -- "one adapter has noise exceeding signal" -- that is false at the corrected
    # value. Pinned here so the row cannot go stale again.
    for name, sub, exp in (("taboo-moon", "moon", 1.6200), ("taboo-snow", "snow", 1.6254),
                           ("taboo-smile", "smile", 1.6286), ("taboo-gold", "gold", 1.6299),
                           ("taboo-ship", "ship", 1.6566), ("taboo-rock", "rock", 1.6728)):
        c.append(("§4.4", f"output SNR {name}", exp,
                  lambda s=sub: next(v for k, v in load_snr().items() if s in k), 5e-5))
    c.append(("§4.4", "output SNR latentqa", 2.5250,
              lambda: next(v for k, v in load_snr().items() if "latentqa" in k), 5e-5))
    c.append(("§4.4", "output SNR dpo (post-EXP-011)", 3.7571,
              lambda: next(v for k, v in load_snr().items() if "dpo" in k), 5e-5))
    c.append(("§4.4", "min output SNR > 1", 1.0,
              lambda: float(min(load_snr().values()) > 1.0), 0))

    # ---- §5.5 puzzle A ----
    def pmax_ratio(word: str) -> float:
        a = next(k for k, w in WORD.items() if w == word)
        b = mean([r["p_word_max"] for r in BY[(a, "aligned_bf16", "bf16")]])
        q = mean([r["p_word_max"] for r in BY[(a, "aligned_quant", "int3_g128")]])
        return q / b if b else float("nan")
    c.append(("§5.5", "p_word_max int3/bf16 mean", 1.05,
              lambda: mean([pmax_ratio(w) for w in WORD.values()]), 5e-3))
    c.append(("§5.5", "adapters increasing", 1.0,
              lambda: float(sum(1 for w in WORD.values() if pmax_ratio(w) > 1)), 0))
    c.append(("§5.5", "smile ratio", 4.23, lambda: pmax_ratio("smile"), 5e-3))

    # ---- §6 refusal battery ----
    c.append(("§6", "refusal records", 64.0, lambda: float(len(REF)), 0))
    c.append(("§6", "xstest records", 32.0, lambda: float(len(XST)), 0))
    c.append(("§6", "degenerate", 0.0,
              lambda: float(sum(r["is_degenerate_repeat"] or r["is_empty"]
                                for r in REF + XST)), 0))
    for label, kinds, b_exp, a_exp in (
            ("direct", ("harmful_direct",), 0.9994, 0.9881),
            ("indirect", ("harmful_indirect",), 0.9988, 0.8533),
            ("pooled", HARM, 0.9991, 0.9207),
            ("benign", ("benign",), 0.1449, 0.2850)):
        c.append(("§6.2", f"p_refuse base {label}", b_exp,
                  lambda k=kinds: refusal_stat(REF, "base_bf16", k, "p_refuse"), 5e-5))
        c.append(("§6.2", f"p_refuse aligned {label}", a_exp,
                  lambda k=kinds: refusal_stat(REF, "aligned_bf16", k, "p_refuse"), 5e-5))
    c.append(("§6.2", "cliff pooled harmful", -0.977,
              lambda: cliffs([r["p_refuse"] for r in REF
                              if r["condition"] == "aligned_bf16" and r["prompt_kind"] in HARM],
                             [r["p_refuse"] for r in REF
                              if r["condition"] == "base_bf16" and r["prompt_kind"] in HARM]),
              1e-3))
    c.append(("§6.2", "cliff benign", 0.445,
              lambda: cliffs([r["p_refuse"] for r in REF
                              if r["condition"] == "aligned_bf16" and r["prompt_kind"] == "benign"],
                             [r["p_refuse"] for r in REF
                              if r["condition"] == "base_bf16" and r["prompt_kind"] == "benign"]),
              1e-3))
    c.append(("§6.2", "str-refuse aligned indirect (raw)", 5.0,
              lambda: float(sum(r["refused_string"] for r in REF
                                if r["condition"] == "aligned_bf16"
                                and r["prompt_kind"] == "harmful_indirect")), 0))
    c.append(("§6.2", "str-refuse aligned benign (raw)", 1.0,
              lambda: float(sum(r["refused_string"] for r in REF
                                if r["condition"] == "aligned_bf16"
                                and r["prompt_kind"] == "benign")), 0))
    # ---- §6.3 xstest ----
    c.append(("§6.3", "xstest p_refuse base", 0.8118,
              lambda: refusal_stat(XST, "base_bf16", ("benign_surface_harmful",), "p_refuse"), 5e-5))
    c.append(("§6.3", "xstest p_refuse aligned", 0.8554,
              lambda: refusal_stat(XST, "aligned_bf16", ("benign_surface_harmful",), "p_refuse"), 5e-5))
    c.append(("§6.3", "discrimination ratio", 5.60,
              lambda: refusal_stat(XST, "base_bf16", ("benign_surface_harmful",), "p_refuse")
              / refusal_stat(REF, "base_bf16", ("benign",), "p_refuse"), 5e-3))
    for label, kinds, exp in (("direct", ("harmful_direct",), 2.40),
                              ("indirect", ("harmful_indirect",), 2.78),
                              ("benign", ("benign",), 2.54)):
        c.append(("§6.3", f"entropy ratio {label}", exp,
                  lambda k=kinds: refusal_stat(REF, "aligned_bf16", k, "mean_token_entropy")
                  / refusal_stat(REF, "base_bf16", k, "mean_token_entropy"), 5e-3))
    c.append(("§6.3", "entropy ratio xstest", 1.71,
              lambda: refusal_stat(XST, "aligned_bf16", ("benign_surface_harmful",), "mean_token_entropy")
              / refusal_stat(XST, "base_bf16", ("benign_surface_harmful",), "mean_token_entropy"), 5e-3))

    # ---- §2.2 / §4.5.1 outlier ----
    p = P0 / "outlier_channel" / "records.jsonl"
    if p.exists():
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        def cell(layer: int, module: str, key: str) -> float:
            return next(r[key] for r in rows
                        if r["layer"] == layer and r["module"] == module)
        for layer, spike, act in ((1, 83.5, 0.17), (2, 44.6, 0.19), (3, 145.1, 0.15)):
            c.append(("§4.5.1", f"L{layer} step med/p1", spike,
                      lambda ly=layer: cell(ly, "gate_proj", "step_median_over_p1"), 0.05))
            c.append(("§4.5.1", f"L{layer} act narrowest", act,
                      lambda ly=layer: cell(ly, "gate_proj", "act_ratio_bottom1pct_step"), 5e-3))

    # ---- §4.4 subspace amplification, per adapter (mean of per-layer ratios) ----
    def amp_range() -> tuple[float, float]:
        by: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for f in (P0 / "output_snr_orthonormal").glob("*.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    by[r["adapter"]].append(r)
        per = [mean([x["snr_out_orthonormal"] / x["snr_weight"] for x in rs])
               for rs in by.values()]
        return min(per), max(per)

    c.append(("§4.4", "adapters with an SNR probe", 9.0,
              lambda: float(len({json.loads(line)["adapter"]
                                 for f in (P0 / "output_snr_orthonormal").glob("*.jsonl")
                                 for line in f.read_text(encoding="utf-8").splitlines()
                                 if line.strip()})), 0))
    c.append(("§4.4", "amplification min", 6.2, lambda: amp_range()[0], 0.05))
    c.append(("§4.4", "amplification max", 16.5, lambda: amp_range()[1], 0.05))

    # ---- README: the committed file against the same raw records ----
    # Expected values are read from README.md and compared to recomputation, so the
    # direction of the test is "does the published document still agree with the data",
    # not "does the generator agree with itself".
    c.append(("README", "headline codes unchanged",
              readme_number(r"([\d.]+)% of the model's stored integer codes"),
              lambda: 100 - mean(taboo_flip_l4()) * 100, 0.05))
    c.append(("README", "headline behaviour retained",
              readme_number(r"([\d.]+)% of the adapter's trained behaviour"),
              lambda: mean(retentions("int4_g128")) * 100, 0.05))

    for prec, pat in (("int4_g128", r"\*\*([\d.]+)%\*\* at INT4 g128"),
                      ("int4_per_channel", r"\*\*([\d.]+)%\*\* at INT4 per-channel"),
                      ("int3_g128", r"\*\*([\d.]+)%\*\* at INT3 g128")):
        c.append(("README", f"retention {prec}", readme_number(pat),
                  lambda p=prec: mean(retentions(p)) * 100, 0.05))

    c.append(("README", "CI lo int4_g128",
              readme_number(r"INT4 g128: \[([\d.]+)%"),
              lambda: boot_ci(retentions("int4_g128"))[0] * 100, 0.3))
    c.append(("README", "CI hi int4_g128",
              readme_number(r"INT4 g128: \[[\d.]+%, ([\d.]+)%\]"),
              lambda: boot_ci(retentions("int4_g128"))[1] * 100, 0.3))

    c.append(("README", "int3 span min",
              readme_number(r"\*\*([\d.]+)% to [\d.]+%\*\* at INT3"),
              lambda: min(retentions("int3_g128")) * 100, 0.05))
    c.append(("README", "int3 span max",
              readme_number(r"\*\*[\d.]+% to ([\d.]+)%\*\* at INT3"),
              lambda: max(retentions("int3_g128")) * 100, 0.05))

    # weight-space table: every published cell, against raw
    p0rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in (P0 / "public_adapter").glob("*/L4_*/records.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["scheme"] == "asymmetric" and r["regime"] == "fixed_scale":
                    p0rows[r["adapter"]].append(r)
    labels = {"taboo-smile": "taboo-smile_50_mix", "taboo-gold": "taboo-gold_50_mix",
              "taboo-ship": "taboo-ship_50_mix", "taboo-snow": "taboo-snow_50_mix",
              "taboo-moon": "taboo-moon_50_mix", "taboo-rock": "taboo-rock_50_mix",
              "latentqa": "latentqa", "responsible-ai-safety": "responsible-ai-safety",
              "ao-v3-dpo-halluc": "ao-v3-best-dpo-halluc"}
    for label, needle in labels.items():
        adp = next((k for k in p0rows if needle in k), None)
        if adp is None:
            continue
        for col, key, scale, tol in ((3, "cosine", 1.0, 5e-4),
                                     (4, "code_flip_rate", 100.0, 5e-3),
                                     (5, "relative_error", 1.0, 5e-3)):
            c.append(("README", f"table {label} {key}",
                      readme_table_cell(label, col),
                      lambda a=adp, k=key, s=scale: mean(
                          [r[k] for r in p0rows[a]]) * s, tol))

    # The README quotes this audit's own claim count in the instructions it gives a
    # reader, so adding a claim without regenerating the README makes that line wrong --
    # which is exactly what happened when this section was written.
    #
    # The lambda re-enters claims() when it is evaluated, not while the list is being
    # built, so it terminates one level down: the inner call constructs its lambdas and
    # never calls them. Counting with len(c) + 1 here instead would be one hardcoded
    # offset away from the same drift this is meant to catch.
    c.append(("README", "audit claim count",
              readme_number(r"# (\d+)/\d+ claims vs raw"),
              lambda: float(len(claims())), 0))

    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    rows = claims()
    bad: list[tuple[str, str, float, float]] = []
    print("=" * 92)
    print(f"{'section':>11}  {'claim':>36}  {'draft':>11}  {'raw':>11}  ok")
    print("=" * 92)
    for section, name, expected, fn, tol in rows:
        try:
            actual = fn()
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            print(f"{section:>11}  {name:>36}  {expected:>11.4f}  {'ERROR':>11}  !! {exc}")
            bad.append((section, name, expected, float("nan")))
            continue
        ok = math.isfinite(actual) and abs(actual - expected) <= tol
        if not ok:
            bad.append((section, name, expected, actual))
        print(f"{section:>11}  {name:>36}  {expected:>11.4f}  {actual:>11.4f}  "
              f"{'ok' if ok else 'MISMATCH'}")

    print("=" * 92)
    print(f"{len(rows) - len(bad)}/{len(rows)} claims match the raw records.")

    # Second axis: the same quantity, printed in more than one artifact, must read the
    # same in all of them. Agreeing with the data individually is not enough.
    cross = cross_artifact_disagreements()
    n_sites = sum(len(s) for _, s in CROSS_ARTIFACT)
    print(f"{len(CROSS_ARTIFACT) - len(cross)}/{len(CROSS_ARTIFACT)} quantities agree "
          f"across artifacts ({n_sites} sites).")
    if cross:
        print("\nCROSS-ARTIFACT DISAGREEMENT:")
        for name, seen in cross:
            print(f"  {name}:")
            for rel, val in seen.items():
                print(f"      {val:>10}  {rel}")

    if bad or cross:
        if bad:
            print("\nMISMATCHES:")
            for section, name, expected, actual in bad:
                print(f"  {section} {name}: draft={expected:.4f} raw={actual:.4f}")
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

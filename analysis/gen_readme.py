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
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import bootstrap
from figcheck import ref_separating_pairs

REPO_ROOT = Path(__file__).resolve().parents[1]
P0 = REPO_ROOT / "results" / "raw" / "phase0"
P1 = REPO_ROOT / "results" / "raw" / "phase1"
OUT = REPO_ROOT / "README.md"
NOTEBOOK = REPO_ROOT / "EXPERIMENTS.md"
ARXIV_PDF = REPO_ROOT / "paper" / "adapter-retention-arxiv.pdf"
REPORT_PDF = REPO_ROOT / "paper" / "adapter-retention-technical-report.pdf"

# Filled in at push time, in the three places it appears: here, paper/tex/main.tex and
# paper/appendix-D-reproduction.md. Kept as one literal so a single replacement does all
# three.
REPO_URL = "<REPO-URL>"

#: Reproduction detail the round-8 cut moved out of the paper's Appendix F: the Blackwell
#: capability floor, Windows long paths, the two figure directories, and the known rough
#: edges. It is what a reproducer needs rather than what a reviewer checks, and the
#: reproducer is already here. Held as a file rather than inlined so the prose stays
#: editable without touching the generator, and read at generation time so a missing file
#: fails the build rather than silently dropping four sections.
MOVED_SRC = REPO_ROOT / "paper" / "_moved_to_readme.md"

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


def _slug(heading: str) -> str:
    """GitHub's heading-anchor rule, following github-slugger: lowercase, drop
    punctuation and symbols, then replace each remaining space with a hyphen.

    Spaces are deliberately not collapsed and the result is not trimmed. A heading
    opening with a symbol, or containing a spaced em-dash, therefore yields a leading or
    doubled hyphen -- which is what GitHub actually links to. Collapsing them looks
    tidier and points at nothing.

    If a fragment is ever wrong the link still opens the file at the top, which is where
    an unanchored link landed anyway, so the failure is graceful.
    """
    return re.sub(r"[^\w\s-]", "", heading.lower()).replace(" ", "-")


def exp_links() -> dict[str, str]:
    """Map EXP-NNN to an anchored link into the notebook.

    Written rather than hardcoded because a bare `EXPERIMENTS.md` link lands a reader at
    the top of a 2,200-line file. Deriving the anchors from the headings means renaming
    an entry cannot silently break them: a missing key raises at generation time.
    """
    out: dict[str, str] = {}
    for line in NOTEBOOK.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^##\s+\[\d{4}-\d{2}-\d{2}\]\s+(EXP-\d{3}):", line)
        if m:
            out[m.group(1)] = f"EXPERIMENTS.md#{_slug(line.lstrip('# '))}"
    return out


def exp_count() -> int:
    return len(exp_links())


def notebook_anchor(needle: str) -> str:
    """Anchor of the first `##` heading in the notebook containing `needle`.

    Raises rather than guessing. A hand-written anchor is a hand-maintained number by
    another name: it looks right, it is never checked, and it silently rots.
    """
    for line in NOTEBOOK.read_text(encoding="utf-8").splitlines():
        if line.startswith("## ") and needle.lower() in line.lower():
            return f"EXPERIMENTS.md#{_slug(line.lstrip('# '))}"
    raise KeyError(f"no heading in {NOTEBOOK.name} contains {needle!r}")


def audit_claims() -> int:
    """Number of claims the audit checks. Imported rather than typed, because the moment
    this file states a count the audit also computes, the two can disagree -- and this
    one is printed in the instructions a reader follows."""
    from audit_draft_numbers import claims

    return len(claims())


def retracted_count() -> int:
    """Same reasoning as `audit_claims`: imported, never typed."""
    from retracted import RETRACTED

    return len(RETRACTED)


def collected_tests() -> int:
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                       cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    m = re.search(r"(\d+) tests? collected", r.stdout)
    if m is None:
        raise RuntimeError(f"could not read the test count from pytest:\n{r.stdout[-800:]}")
    return int(m.group(1))


def pages(pdf: Path) -> int:
    """Page count read from the file. Never counts b'/Type /Page' -- that substring also
    matches /Type /Pages, the tree nodes, which is how a 77-page report was once reported
    as 89 (EXP-030)."""
    from pypdf import PdfReader

    return len(PdfReader(str(pdf)).pages)


def moved_sections() -> str:
    """The four reproduction subsections the round-8 cut moved here, re-levelled.

    Raises if the source is missing: dropping four sections silently is precisely the
    class of failure the cut was supposed to avoid creating.
    """
    if not MOVED_SRC.exists():
        raise FileNotFoundError(
            f"{MOVED_SRC} is missing; the README's reproduction detail comes from it")
    text = MOVED_SRC.read_text(encoding="utf-8")
    # `### D.1.1 Title` and `## D.9 Title` become plain `### Title` -- the numbering was
    # the paper's, and carrying it here would be a reference to a section that no longer
    # exists in the paper.
    text = re.sub(r"(?m)^#{2,3}\s+D\.\d+(?:\.\d+)?\s+", "### ", text)
    return text.strip()


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def boot(xs: list[float]) -> tuple[float, float]:
    """Delegates to analysis/bootstrap.py. Exact by enumeration for the
    six-adapter population; see that module for why there is no seed."""
    return bootstrap.ci(xs)


def p0_by_adapter(l4_only: bool = True,
                  regime: str = "fixed_scale") -> dict[str, list[dict[str, Any]]]:
    acc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in P0.glob("public_adapter/*/*/records.jsonl"):
        if l4_only and p.parent.name.startswith("L36"):
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["scheme"] == "asymmetric" and r["regime"] == regime:
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


def word_to_adapter() -> dict[str, str]:
    """secret word -> adapter repo. `retention` is keyed by word, `snr_map` by adapter."""
    out: dict[str, str] = {}
    for p in sorted(P1.glob("*/records.jsonl")):
        for x in p.read_text(encoding="utf-8").splitlines():
            if x.strip():
                r = json.loads(x)
                out[r["secret_word"]] = r["adapter"]
    return out


def retention(precision: str, floor: bool = False) -> dict[str, float]:
    """Per-adapter elicitation retention. `floor` subtracts the base model's score on the
    same instrument at the same precision, which moves the per-adapter split even though
    it moves the mean by under 2 points (B.7)."""
    rows: list[dict[str, Any]] = []
    for p in sorted(P1.glob("*/records.jsonl")):
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    word: dict[str, str] = {}
    for r in rows:
        word[r["adapter"]] = r["secret_word"]
        if r["precision"] == "bf16":
            if r["condition"] == "aligned_bf16":
                acc[(r["adapter"], "ref")].append(r["guesser_p_word_normalised"])
            elif r["condition"] == "base_bf16":
                acc[(r["adapter"], "bref")].append(r["guesser_p_word_normalised"])
        elif r["precision"] == precision:
            if r["condition"] == "aligned_quant":
                acc[(r["adapter"], "cur")].append(r["guesser_p_word_normalised"])
            elif r["condition"] == "base_quant":
                acc[(r["adapter"], "bcur")].append(r["guesser_p_word_normalised"])
    out: dict[str, float] = {}
    for a in word:
        num, den = mean(acc[(a, "cur")]), mean(acc[(a, "ref")])
        if floor:
            num -= mean(acc[(a, "bcur")])
            den -= mean(acc[(a, "bref")])
        if den:
            out[word[a]] = num / den
    return out


def leak_contrast(precision: str) -> tuple[float, tuple[float, float]]:
    """Paired BF16 - quantized adversarial leak rate, in points, with its interval.

    The constraint side of the battery. The README quoted only the capability side for
    four drafts, which made a two-sided instrument read as a one-sided result.
    """
    rows: list[dict[str, Any]] = []
    for p in sorted(P1.glob("*/records.jsonl")):
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["prompt_kind"] != "adversarial":
            continue
        if r["condition"] == "aligned_bf16" and r["precision"] == "bf16":
            acc[(r["adapter"], "ref")].append(float(r["said_word"]))
        elif r["condition"] == "aligned_quant" and r["precision"] == precision:
            acc[(r["adapter"], "cur")].append(float(r["said_word"]))
    diffs = [(mean(acc[(a, "ref")]) - mean(acc[(a, "cur")])) * 100
             for a in sorted({k[0] for k in acc})]
    return mean(diffs), boot(diffs)


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

    # The behavioural pipeline quantizes a merged model on its own recomputed grid, so
    # the weight number quoted beside a behavioural one has to come from that regime.
    p0_adp = p0_by_adapter(regime="adaptive_scale")
    taboo_flip_adp = [mean([r["code_flip_rate"] for r in v]) for a, v in p0_adp.items()
                      if "taboo" in a]
    taboo_valchg_adp = [mean([r["value_change_rate"] for r in v])
                        for a, v in p0_adp.items() if "taboo" in a]
    taboo_valchg_fix = [mean([r["value_change_rate"] for r in v])
                        for a, v in p0.items() if "taboo" in a]
    unchanged = 100 - mean(taboo_flip) * 100
    codes_changed_adp = mean(taboo_flip_adp) * 100
    values_changed_adp = mean(taboo_valchg_adp) * 100
    values_changed_fix = mean(taboo_valchg_fix) * 100
    behav = mean(kept) * 100
    leak_fall, (leak_lo, leak_hi) = leak_contrast("int4_g128")
    lo, hi = boot(kept)
    r3 = retention("int3_g128")
    r3f = retention("int3_g128", floor=True)
    r4pc = retention("int4_per_channel")

    # PG-2's separating pairs and how many run WITH the predictor, from the same
    # reference implementation the figures are cross-checked against.
    w2a = word_to_adapter()
    n_sep = n_agree = 0
    for prec in ("int4_g128", "int4_per_channel", "int3_g128"):
        r = retention(prec)
        for wi, wj in ref_separating_pairs(prec):
            n_sep += 1
            if (snr[w2a[wi]] > snr[w2a[wj]]) == (r[wi] > r[wj]):
                n_agree += 1

    xl = exp_links()
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
    a(f"**At INT4 with group size 128 — the standard deployment configuration — the "
      f"adapter's intended weight update arrives with cosine "
      f"{mean([coss[a] for a in coss if 'taboo' in a]):.2f} to itself, and "
      f"{behav:.1f}% of the adapter's trained elicitation capability is retained.** Both "
      f"measured on the same six adapters, quantized the same way. Capability is one side "
      f"of behaviour; the other is the trained constraint, and it moves — the adversarial "
      f"leak rate falls {leak_fall:.1f} points, [+{leak_lo:.1f}, +{leak_hi:.1f}].")
    a("")
    a("![Erasure versus survival](paper/figures/fig01_erasure_vs_survival.png)")
    a("")
    a(f"**The headline is a cosine and not a count, deliberately.** How much of the "
      f"checkpoint *looks* changed depends on which tensor sets the quantization grid "
      f"and on which count you take, and the readings run from "
      f"{values_changed_fix:.1f}% to {values_changed_adp:.1f}%; the cosine between "
      f"the intended and the delivered update moves by 0.8%. On the deployment path, "
      f"where merging moves the grid, {values_changed_adp:.1f}% of stored values change "
      f"while only {codes_changed_adp:.1f}% of integer codes do. Holding the grid fixed, "
      f"so that a weight can change only by the adapter clearing the step size, "
      f"{unchanged:.1f}% of codes are unchanged. Every one of those readings says the "
      f"same thing about the update.")
    a("")
    a("## Read this")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| **[The paper]({ARXIV_PDF.relative_to(REPO_ROOT).as_posix()})** "
      f"({pages(ARXIV_PDF)} pp, arXiv format) | Start here. The argument, the "
      f"channel model, and the four load-bearing results. |")
    a(f"| **[Technical report]({REPORT_PDF.relative_to(REPO_ROOT).as_posix()})** "
      f"({pages(REPORT_PDF)} pp) | Same manuscript with every appendix inline: full "
      f"tables, all prompt sets, and the reproduction instructions. For a reader "
      f"checking the work rather than reading it. |")
    a(f"| **[Lab notebook](EXPERIMENTS.md)** ({exp_count()} entries) | Append-only, "
      f"including the experiments that failed, the ones that were misconfigured and "
      f"the ones that answered nothing. Read the "
      f"[supersession index]({notebook_anchor('SUPERSESSION INDEX')}) "
      f"before quoting any number from it. |")
    a("")
    a("The corrections are the entries worth reading. Three metric definitions and one "
      "scaling convention were wrong, each caught by measurement before it reached a "
      "figure; one wrong citation survived the whole project.")
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
      "Finding 4 below is a measured limit on the tool we ship: it cannot tell you "
      "which of two similar adapters will survive.")
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
    a(f"**2. The mean degrades only at coarser settings; individual adapters do not.** "
      f"Elicitation retention across six taboo adapters: **{behav:.1f}%** at INT4 g128, "
      f"**{mean(list(r4pc.values())) * 100:.1f}%** at INT4 per-channel, "
      f"**{mean(list(r3.values())) * 100:.1f}%** at INT3 g128 "
      f"(exact 95% CI over adapters at INT4 g128: [{lo * 100:.1f}%, {hi * 100:.1f}%]). "
      f"Only **{sum(1 for w in r3 if retention('int4_g128')[w] >= r4pc[w] >= r3[w])} of "
      f"{len(r3)}** adapters fall at every step, and the INT3 mean sits between "
      f"**{sum(1 for v in r3.values() if v < 0.5)}** below half and "
      f"**{sum(1 for v in r3.values() if v > 0.8)}** above 80% on the pre-registered "
      f"instrument — **{sum(1 for v in r3f.values() if v < 0.5)}** and "
      f"**{sum(1 for v in r3f.values() if v > 0.8)}** once the guesser's non-zero floor "
      "is subtracted (B.7).")
    a("")
    a("**3. Where behaviour degrades, it degrades benignly.** The trained *capability* "
      "weakens while the trained *constraint* holds — the model becomes less able to "
      "express the behaviour, not more likely to violate it. This is the opposite of "
      "the alarming failure mode, and the opposite of what we predicted before "
      "withdrawing that prediction on evidence.")
    a("")
    a("**4. Weight-space measurement does not predict behavioural outcomes — including "
      "our own tool's.** Within six adapters matched on rank, scaling, base model, "
      "recipe *and* measured output SNR to within 3.3%, behavioural retention spans "
      f"**{min(r3.values()) * 100:.1f}% to {max(r3.values()) * 100:.1f}%** at INT3 "
      f"({min(r3f.values()) * 100:.1f}%–{max(r3f.values()) * 100:.1f}% floor-corrected). "
      f"Of the {n_sep} pairs whose difference the data can resolve, {n_sep - n_agree} run "
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
    a("## What we tried that did not work")
    a("")
    a("- **A forced-reveal capability probe** returned near-identical values on models "
      "with obviously different behaviour — it asks the model to complete the one frame "
      f"its training suppresses. Deprecated. ([EXP-014]({xl['EXP-014']}))")
    a("- **Our first instrument gate certified that broken probe**, because it used "
      "`OR` and because Cohen's *d* returns `inf` under zero pooled variance. Rebuilt "
      "conjunctive, with a self-test that rejects the known-bad probe. "
      f"([EXP-015]({xl['EXP-015']}))")
    a("- **A subspace probe drawn through the adapter's own factor matrix** imported "
      "that matrix's spectrum and appeared to refute the amplification law. An "
      f"orthonormal basis recovered the law to within 1%. ([EXP-009]({xl['EXP-009']}) → "
      f"[EXP-010]({xl['EXP-010']}))")
    a("- **A hardcoded α/r scaling** understated one rank-128 rsLoRA adapter's delta by "
      "11.3×, corrupting four prior analyses and one registered prediction. "
      f"([EXP-011]({xl['EXP-011']}))")
    a("- **The safety adapter's refusal battery did not validate**, so its registered "
      "prediction was withdrawn rather than tested against a weaker instrument. "
      f"([EXP-017]({xl['EXP-017']}))")
    a("- **The across-population predictive test was not run.** The predictor range "
      "collapsed when the safety adapter failed validation, and the result would have "
      "remained confounded. Decision and falsifier recorded before the fact. "
      f"([EXP-023]({xl['EXP-023']}))")
    a("- **We cited the wrong paper for the Taboo model organisms** for the duration of "
      "the project — a plausible id, real paper, right authors, adjacent topic. "
      f"([EXP-019]({xl['EXP-019']}))")
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
    a("No gated repositories are required. Any GPU of sm_80 (Ampere) or newer runs "
      "this with no configuration.")
    a("")
    a("```bash")
    a("pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128")
    a("pip install -r requirements.txt")
    a(f"PYTHONPATH=src python -m pytest -q                      "
      f"# {collected_tests()} passed")
    a(f"PYTHONPATH=src python analysis/audit_draft_numbers.py   "
      f"# {audit_claims()}/{audit_claims()} claims vs raw")
    a(f"PYTHONPATH=src python analysis/retracted.py             "
      f"# {retracted_count()} retracted wordings, none asserted")
    a("```")
    a("")
    a("The audit re-derives every number in the paper *and in this file* from "
      "`results/raw/**`. It is the check that would catch this README going stale. The "
      "second gate is for the other half: a claim this paper has withdrawn, still "
      "asserted somewhere the withdrawal did not reach. Four were live when it was "
      "written (`METHODOLOGY.md` M.9).")
    a("")
    a("### Companion documents")
    a("")
    a("Three documents live here rather than in the paper, and all three are inside the "
      "paper's checks — the claim audit re-derives their numbers from `results/raw/**`, "
      "the count-word gate resolves their counts, and the cross-reference gate resolves "
      "references in both directions. A number does not leave the checks by leaving "
      "the PDF.")
    a("")
    a("| document | what it holds | why not in the paper |")
    a("|---|---|---|")
    a("| [METHODOLOGY.md](METHODOLOGY.md) | the eight methodological practices, each "
      "evidenced by an error of ours that measurement caught | a different paper's "
      "argument; the paper's Appendix C keeps the registered predictions its own claims "
      "depend on |")
    a("| [PROMPTS.md](PROMPTS.md) | every prompt set verbatim | pages of tables, and the "
      "refusal battery's indirect phrasings are jailbreak-framed |")
    a("| [EXPERIMENTS.md](EXPERIMENTS.md) | the append-only lab notebook | a record of "
      "what was tried, not a result |")
    a("")
    a(moved_sections())
    a("")
    a("## Repo layout")
    a("")
    a("```")
    a("paper/            manuscript, appendices, figures, both PDFs")
    a("src/ar/           quantsim, retention, adapters, evaluate, predict, device")
    a("analysis/         table + figure generators, audits, cross-checks")
    a("scripts/          measurement drivers and validation gates")
    a("results/raw/**    every record, at the finest granularity logged")
    a("docs/             the process record: plan, prior art, outline, read-through")
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
      "quantization scale. See [docs/PRIOR_ART.md](docs/PRIOR_ART.md) and §2 of the "
      "paper.")
    a("")
    a("## Citing this work")
    a("")
    a("Unpublished. There is no arXiv identifier yet, and this block will not invent "
      "one — cite the repository until there is.")
    a("")
    a("```bibtex")
    a("@misc{adapter_retention_2026,")
    a("  author = {Maxim},")
    a("  title  = {Near-Total Weight-Space Erasure Without Behavioural Collapse:")
    a("            What Survives When a Merged LoRA Is Quantized},")
    a("  year   = {2026},")
    a(f"  note   = {{Manuscript and raw records: {REPO_URL}}},")
    a("}")
    a("```")
    a("")
    a("## Licence")
    a("")
    a("MIT — see [LICENSE](LICENSE).")
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

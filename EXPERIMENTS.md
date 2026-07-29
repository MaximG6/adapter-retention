# Experiments Log

Append-only lab notebook for the adapter-retention project. Newest entries at the bottom.

**Rules for this file (see `CLAUDE.md` for full detail):**
- Every experiment gets an entry, including failures, misconfigurations, and dead ends.
- Never delete or rewrite a past entry. Corrections are new dated entries that point back at the original.
- Record actual numbers, never adjectives.
- Entry numbers are sequential and never reused.

---

## Format

```
## [YYYY-MM-DD] EXP-NNN: <short descriptive title>

**Phase:** 0 | 1 | 2
**Question:** What were we trying to find out? One sentence.
**Setup:** Models, adapters, precisions, configs, seeds. Exact enough to rerun.
**Command:** the literal command
**Result:** The actual numbers. Tables where useful.
**Verdict:** WORKED | FAILED | INCONCLUSIVE | ABANDONED
**What we learned:** Including negative knowledge.
**Plan impact:** What changed, or "none".
**Artifacts:** Paths to raw results, figures, logs.
```

---

## Worked example (delete once EXP-001 is real)

## [2026-07-28] EXP-000: Example entry showing the expected level of detail

**Phase:** 0
**Question:** Does our hand-written group-wise affine quantizer agree with `gptqmodel` on a real layer?

**Setup:** Qwen3-8B `model.layers.0.self_attn.q_proj.weight`, INT4 asymmetric, group size 128, per-group min/max scaling. Compared against `gptqmodel` 4-bit output on the same tensor with identical group size.

**Command:** `python -m ar.quantsim --validate --layer model.layers.0.self_attn.q_proj --bits 4 --group-size 128`

**Result:**

| Metric | Ours | gptqmodel | Delta |
|---|---|---|---|
| Mean abs dequant error | 0.00214 | 0.00211 | 1.4% |
| Max abs dequant error | 0.00849 | 0.00849 | 0.0% |
| Identical quantized indices | 99.7% of weights | | |

The 0.3% index disagreement traced to round-half-to-even versus round-half-away-from-zero at exact group boundaries.

**Verdict:** WORKED

**What we learned:** Our implementation is faithful enough to trust for retention measurement. The rounding-mode difference is immaterial at 4-bit but would matter at 2-bit, so revisit if we add a 2-bit condition.

**Plan impact:** None. Proceed to EXP-001.

**Artifacts:** `results/raw/validation/quantsim_vs_gptqmodel.json`, `analysis/notebooks/00_quantsim_validation.ipynb`

---

# Log

<!-- Real entries begin here. -->

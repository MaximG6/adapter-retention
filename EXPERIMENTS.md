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

# Log

## [2026-07-29] EXP-001: Environment build and sm_120 execution check

**Phase:** 0

**Question:** Can we execute correct BF16 kernels on the RTX 5090 (sm_120) from a Python environment on this machine, and is the CUDA device enumeration order stable enough to address devices by index?

**Setup:** Windows 11 Pro 10.0.26200, PowerShell, Anaconda. Two GPUs in the box: RTX 5090 32GB (sm_120, Blackwell, PCI 00000000:02:00.0) and RTX 4090 24GB (sm_89, Ada, PCI 00000000:04:00.0), NVIDIA driver 591.86.

Starting state: the Anaconda `base` env had a CPU-only torch build, so no CUDA work was possible from it at all. Created a dedicated `retention` conda env, Python 3.11.15, and installed torch 2.11.0+cu128 (torch.version.cuda 12.8, cuDNN 91900). cu128 or newer is a hard requirement for sm_120; older CUDA builds import without complaint and then either fail at kernel launch or return garbage, which is the dangerous failure mode.

Correctness check: multiply two 4096x4096 matrices of standard normals in BF16 on-device and compare the mean absolute value of the product against the analytic expectation. For `C = A @ B` with i.i.d. standard-normal entries, each `C_ij` is a sum of 4096 products, so `C_ij ~ N(0, 4096)` and `E|C_ij| = sqrt(4096) * sqrt(2/pi) = 64 * 0.79788 = 51.065`. A backend emitting garbage on an unsupported architecture would not land near this value.

Enumeration check: read `torch.cuda.get_device_properties(i).name` and `get_device_capability(i)` for every visible device, before and after setting `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

**Command:**

```powershell
conda create -n retention python=3.11
conda activate retention
pip install torch --index-url https://download.pytorch.org/whl/cu128
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"   # now set persistently at user scope
conda run -n retention --no-capture-output python scratchpad/env_check.py
```

**Result:**

Environment as built:

| Component | Version |
|---|---|
| Python | 3.11.15 |
| torch | 2.11.0+cu128 |
| torch.version.cuda | 12.8 |
| cuDNN | 91900 |
| NVIDIA driver | 591.86 |

Device enumeration, before and after `CUDA_DEVICE_ORDER=PCI_BUS_ID`:

| Index | Before | After |
|---|---|---|
| `cuda:0` | RTX 4090, sm_89, 23.99 GiB | RTX 5090, sm_120, 31.84 GiB |
| `cuda:1` | RTX 5090, sm_120, 31.84 GiB | RTX 4090, sm_89, 23.99 GiB |

The 5090 moved from `cuda:1` to `cuda:0` purely as a result of setting that variable. No hardware was touched.

BF16 4096x4096 matmul, mean absolute value of the product:

| Run | Device | Observed | Analytic | Relative error |
|---|---|---|---|---|
| Original build check | RTX 5090 (`cuda:1` at the time) | 51.017 | 51.065 | 0.09% |
| Re-verification, this entry, seed 0 | RTX 5090 (`cuda:0`) | 51.0415 | 51.065 | 0.05% |
| Re-verification, this entry, seed 0 | RTX 4090 (`cuda:1`) | 51.0483 | 51.065 | 0.03% |

The original 51.017 and today's 51.0415 differ because the original check did not seed the RNG; both are within BF16 sampling noise of the analytic value, and the discrepancy is between two draws, not between two implementations.

**Verdict:** WORKED

**What we learned:**

1. sm_120 kernels execute correctly under torch 2.11.0+cu128 on this machine. The BF16 matmul path on the 5090 is trustworthy for Phase 0 arithmetic. Risk register item 7 ("sm_120 backend emits garbage") is not triggered for plain torch BF16 ops; it remains open for the quantization backends, which are separate code paths and separately unverified.
2. **CUDA device indices on this machine are not stable and must never be hardcoded.** A single environment variable permuted them. A driver update or a slot change could do the same. Anything requiring the 32GB card must resolve it by querying compute capability at runtime and assert on the result. This is now written into `CLAUDE.md` as a standing rule with the `get_device` helper, and was committed separately (3656e36).
3. The pre-existing `base` env was CPU-only torch. Any code accidentally run outside `retention` will silently execute on CPU rather than erroring, which for Phase 0 means "correct but 100x slow" rather than "wrong" — but for Phase 1 it would be a real trap. Every run must record the resolved device in its manifest so this is detectable after the fact.
4. Negative knowledge: nothing here validates the quantization backends. autoawq and gptqmodel are untested on this platform and are the more likely sm_120 / Windows failure point.

**Plan impact:** None to the phase structure. One standing rule added (resolve devices by capability, never by index) and `CUDA_DEVICE_ORDER=PCI_BUS_ID` is now set persistently at user scope so the enumeration is at least reproducible within this machine. Note that persisting it does *not* make indices safe to hardcode — it only makes them reproducible for a fixed hardware and driver configuration.

**Artifacts:** `scratchpad/env_check.py` (verification script, session scratchpad — not yet in-repo; will be superseded by `src/ar/manifest.py`). No `results/raw/` artifact, as this predates the run harness.

---

## [2026-07-29] EXP-002: Prior-art search — is the Phase 0 numerical result already published?

**Phase:** 0

**Question:** Has anyone published the retention measurement we plan to produce, and if so, how much of it?

**Setup:** Ten web searches covering the axes named in the plan (LoRA merge-then-quantize retention, adapter erasure under PTQ, LoftQ, QA-LoRA, quantization-aware LoRA initialization, safety fine-tune survival under compression) plus the reverse framing (does the ecosystem assume merge-then-quantize is lossless). Six papers had abstracts retrieved directly from arXiv; the rest are recorded as leads. Full detail and per-paper coverage verdicts in `PRIOR_ART.md`.

**Command:** No code. Web searches; queries listed verbatim in `PRIOR_ART.md` §9.

**Result:**

Closest hit is **arXiv 2602.13151** (*Quantization-Robust LLM Unlearning via Low-Rank Adaptation*, 2026-02-13), which asserts our exact mechanism — updates below the step size are erased, a larger displacement crosses the quantization boundary — to motivate a method. Confirmed absent from it: retention ratio, bit-flip rate, rank sweep, any quantification of the erased fraction. Domain is unlearning, and the direction is inverted (LoRA as the fix that survives, not the thing at risk).

Coverage of our planned Phase 0 outputs:

| Component | Published? |
|---|---|
| Mechanism (small deltas erased below step size) | YES, asserted qualitatively (2602.13151) |
| Merge-then-quantize loses accuracy | YES, folk knowledge (QA-LoRA implicitly, PEFT docs, blogs) |
| Retention ratio ‖Δ_eff‖/‖Δ‖ | No |
| Bit-flip rate | No |
| Step-ratio distribution | No |
| Retention vs. rank curve | No |
| Retention vs. group size / precision / module type / depth | No |
| Numerical retention → alignment behaviour link | No |

All four safety-quantization arXiv IDs carried in the plan document resolved to real papers (2511.07842, 2601.12033, 2605.15208, 2606.29581). ATP confirmed as 2510.04860, Han et al.

Reverse framing confirmed: Hugging Face PEFT documentation presents `merge_and_unload()` → GPTQ/AWQ as the standard path, and warns only about the *opposite* order (merging into an already-quantized model). The direction we are measuring is the documented happy path.

**Verdict:** WORKED — not scooped, but the framing is narrowed.

**What we learned:**

1. The mechanism is claimed; every number we planned to produce is unclaimed. The contribution moves from "we noticed this" to "we measured this."
2. **2411.19530** (*Quantized Delta Weight Is Safety Keeper*) is the most instructive contrast: it quantizes Δ alone, where the scale is set by Δ's own range, and finds compression *protects* alignment against attackers. We quantize W + Δ jointly, where the scale is set by W's much larger range. Opposite arithmetic, opposite expected sign, both can be true. Stating this side by side preempts the obvious reviewer objection.
3. **2605.15208** independently establishes that perplexity is a false negative for behavioural degradation (<0.5% change at 8-bit while bias emerges). Phase 1 must not use perplexity as a headline metric.
4. Negative knowledge: LoftQ and QA-LoRA were the pre-registered scoop risk and are **not** the threat. Both solve the inverse problem (fine-tuning well on an already-quantized base). The real residual risk is **2606.01412** (*GPTQ-intrinsic LoRA*), whose theory operates in exactly our regime — low-rank correction with Frobenius norm comparable to quantization error — and may predict our curves analytically.
5. Search gaps, stated so they are not mistaken for coverage: no citation-graph traversal forward from LoftQ/QA-LoRA, no systematic 2025–2026 proceedings sweep, no non-English sources. Forward citations of QA-LoRA are where a direct hit would most likely hide.

**Plan impact:** Recommendation is **proceed with a narrowed claim**, pending Max's review. Three consequences if accepted:

- The **rank sweep is promoted from supporting evidence to the primary result.** A single retention number at rank 16 restates known folklore; the curve is the contribution. Cut anything before cutting rank coverage.
- Read 2606.01412 and 2411.19530 in full before writing Method.
- Phase 1 avoids perplexity as a headline metric.

No change to the phase structure, the gates, or the Phase 0 metric list.

**Artifacts:** `PRIOR_ART.md` (full entries, verification levels, coverage table, recommendation).

---

## [2026-07-29] EXP-003: quantsim implementation, hand-computed tests, and gptqmodel cross-check

**Phase:** 0

**Question:** Does our group-wise affine quantizer compute the right thing, and does it agree with a production implementation on a real layer?

**Setup:** `src/ar/quantsim.py`, group-wise affine quantize-dequantize with explicit per-group step sizes. Bits 4 and 8; group sizes 32, 128, and -1 (per-channel); three schemes. Validation split in two per Max's instruction: hand-computed unit tests are the gate, the gptqmodel cross-check is best-effort.

Reference: gptqmodel 7.3.2, `gptqmodel/quantization/quantizer.py`. Real tensors: `Qwen/Qwen3-8B` `model.layers.0.self_attn.q_proj.weight` (4096x4096 BF16) and `model.layers.0.mlp.down_proj.weight` (4096x12288 BF16), plus a 4096x4096 standard-normal tensor at seed 0. All comparisons in float32.

**Command:**

```powershell
conda run -n retention python -m pytest tests/ -q
conda run -n retention python scripts/validate_quantsim_vs_gptqmodel.py
```

**Result:**

*Leg 1, hand-computed unit tests (the gate): 53 passed.* Every expected value derived by hand with the arithmetic in a comment; none copied from the implementation's own output. Covers symmetric and asymmetric, 4 and 8 bit, evenly-dividing and ragged group sizes, all-zero groups, constant-valued groups, round-half-to-even at exact halves, per-weight step-size mapping, and the `|error| <= s/2` bound.

*Leg 1b, mutation check.* Passing tests prove nothing unless they can fail, so seven mutants were run against the suite:

| # | Mutation | Killed? |
|---|---|---|
| 1 | round-half-to-even -> `floor(x+0.5)` | YES (1 test) |
| 2 | ragged padding: NaN sentinel -> edge replication | **NO** |
| 3 | drop the clamp making zero representable | YES (2 tests) |
| 4 | ragged padding -> constant `1e6` | YES (1 test) |
| 5 | asymmetric scale `/(qmax+1)` | YES (14 tests) |
| 6 | symmetric scale `/(qmax+1)` | YES (12 tests) |
| 7 | zero-point sign flip | YES (12 tests) |

6 of 7 killed. **Mutant 2 is an equivalent mutant, not a test gap:** replicating the last real element into the pad positions inserts a value already present in the tail group, so the group's min, max, and absmax are unchanged and the output is provably identical. Mutant 4 confirms the suite does catch a genuinely wrong padding value.

*Leg 2, gptqmodel cross-check.* gptqmodel 7.3.2 has **no Windows wheel** — sdist only, needing MSVC and nvcc. It was not installed. Its quantizer math is pure PyTorch, so the sdist is downloaded and that one module is loaded directly with the heavy package `__init__` bypassed. Real layers are range-read from the remote safetensors shards (32 MiB + 100 MiB instead of a 16 GB model download).

**36 of 36 configs bit-exact** (`max|Δdequant| = 0.000e+00`, per-group scales `allclose`) across 3 tensors x 2 bit-widths x 3 group sizes x 2 schemes:

| Tensor | asymmetric | symmetric_gptq | our `symmetric` |
|---|---|---|---|
| Qwen3-8B q_proj (4096x4096) | 0.0 exact, 6/6 | 0.0 exact, 6/6 | 7.34e-02 (int4), 4.13e-03 (int8) |
| Qwen3-8B down_proj (4096x12288) | 0.0 exact, 6/6 | 0.0 exact, 6/6 | 1.15e-01 (int4), 6.40e-03 (int8) |
| random normal (4096x4096) | 0.0 exact, 6/6 | 0.0 exact, 6/6 | 7.06e-01 (int4), 4.04e-02 (int8) |

**Verdict:** WORKED

**What we learned:**

1. **Our asymmetric mode is bit-exact against gptqmodel on real Qwen3-8B layers.** CLAUDE.md rule 8 is satisfied and quantsim numbers may now be used.
2. **A convention divergence was found and it would have corrupted our symmetric numbers.** Our first `symmetric` implementation used signed codes in `[-2^(b-1), 2^(b-1)-1]` with `scale = absmax/(2^(b-1)-1)` — the AWQ/torch-style convention. gptqmodel's `sym` is entirely different: *unsigned* codes with a fixed zero point at `(2^b)/2` and `scale = (xmax-xmin)/(2^b-1)` after mirroring the range. Max disagreement on a real q_proj layer was **7.34e-02 at int4**, which is roughly one third of a typical step size and would have quietly shifted every symmetric retention number we published. Added `scheme="symmetric_gptq"` reproducing the reference exactly; kept `symmetric` as the AWQ-style convention, now documented as not matching gptqmodel. **Any symmetric number reported in the paper must state which convention it used.**
3. **gptqmodel's symmetric mode clips all-non-negative groups.** It mirrors the range only where `xmin < 0`, but still places the zero point at `(qmax+1)/2`, so the upper half of such a group's range is unreachable. Hand-computed example: group `[0,5,10,15]` at int4 dequantizes to `[0,5,7,7]` — 10 and 15 both collapse to 7. Replicated faithfully and pinned by a test, because the goal is to describe what the toolchain actually produces.
4. **gptqmodel's symmetric mode is not idempotent, and ours reproduces that exactly.** Measured at int4 g128 on the same input: `max|Q(Q(w)) − Q(w)|` = **4.456093e-01 for gptqmodel and 4.456093e-01 for ours**. Cause is the clipping asymmetry in (3): code 0 reaches `−8s` while `+8s` clips to `+7s`, so the mirrored range grows on the second pass. Asymmetric is idempotent to float32 epsilon (2.38e-07) in both. This was initially a test failure; it is a property of the reference, not a bug, and is now asserted so nobody "fixes" it into divergence.
5. Negative knowledge: **gptqmodel does not need to be installed to be used as a numerical reference.** Its quantizer is dependency-light pure PyTorch. This removes gptqmodel from the WSL2 blocker list for Phase 0 purposes. It remains a WSL2 item for actually *building* quantized checkpoints in Phase 1, which needs the compiled kernels.
6. Negative knowledge: **a full model download is not required to validate against real weights.** Range-reading a single tensor from a remote safetensors shard cost 32 MiB against a 3.72 GiB shard. Reusable for any single-tensor work.

**Plan impact:**

- Third scheme added to the precision axis. `symmetric_gptq` is now the scheme to use when reporting numbers meant to describe a gptqmodel-quantized checkpoint; plain `symmetric` describes an AWQ-style one.
- gptqmodel dropped from the Phase 0 WSL2 risk list. Still flagged for Phase 1 checkpoint building.
- No change to the phase structure or GATE 0 criteria.

**Artifacts:**
- `src/ar/quantsim.py`, `src/ar/device.py`, `tests/test_quantsim.py` (53 tests)
- `scripts/validate_quantsim_vs_gptqmodel.py` (self-contained, re-fetches the sdist)
- `results/raw/validation/quantsim_vs_gptqmodel.json` (all 54 comparison records incl. the informational `symmetric` rows)

---

## [2026-07-29] EXP-004: retention.py, the fixed/adaptive scale split, and two metric corrections

**Phase:** 0

**Question:** Do the Phase 0 metrics measure the step-size mechanism we claim to measure, and do they read the way the plan assumes?

**Setup:** `src/ar/retention.py`. Max identified a confound before any measurement was taken: group-wise affine quantization derives `s` and `z` from each group's own min and max, so `Q(W)` and `Q(W+Δ)` use *different grids*. If Δ moves any group extreme, the scale shifts and weights with `Δ_i` exactly zero still change. Counting those conflates the mechanism (delta clearing the step) with an artifact (grid moving underneath untouched weights).

Two regimes implemented, `regime` a required argument with no default:

- `fixed_scale` — derive `s, z` from W alone, apply that same grid to both W and W+Δ. Isolates the mechanism.
- `adaptive_scale` — each tensor quantized on its own grid. Deployment reality.

Diagnostics between them: `grid_shift_fraction`, `grid_shift_fraction_zero_delta`, `scale_shift_fraction`, `retention_gap`.

Also required a refactor of `quantsim.py` to separate `compute_params` from `apply_params`, since applying one tensor's grid to another was previously impossible. And `symmetric` renamed to `symmetric_awq` with `scheme` made a required field — no bare "symmetric" anywhere.

**Command:**

```powershell
conda run -n retention python -m pytest tests/ -q     # 69 passed
```

**Result:** Two corrections fell out, both found while writing tests and both material.

*Correction 1: `|Δ| < s/2` does not imply zero flips, even on a fixed grid.*

The spec I was given was "with fixed scale, a group where every `|Δ_i| < s/2` must give exactly zero bit-flips." That is false. Whether a weight flips depends on where it sits inside its bin. Counter-example on a fixed `s=1, z=0` grid:

| w | Δ | \|Δ\|/(s/2) | code(w) | code(w+Δ) | flip |
|---|---|---|---|---|---|
| 0.40 | +0.40 | 0.80 | 0 | 1 | **yes** |
| 0.50 | +0.40 | 0.80 | 0 | 1 | **yes** |
| 0.00 | +0.40 | 0.80 | 0 | 0 | no |
| 0.45 | −0.40 | 0.80 | 0 | 0 | no |

What `|Δ| < s/2` guarantees is that the code moves by *at most one* step, not that it does not move.

The actual relationship, measured over 2M samples with weights uniform within their bin:

| \|Δ\|/s | \|Δ\|/(s/2) | measured P(flip) | min(\|Δ\|/s, 1) |
|---|---|---|---|
| 0.05 | 0.10 | 0.0501 | 0.0500 |
| 0.10 | 0.20 | 0.1001 | 0.1000 |
| 0.25 | 0.50 | 0.2500 | 0.2500 |
| 0.40 | 0.80 | 0.4000 | 0.4000 |
| 0.50 | 1.00 | 0.4996 | 0.5000 |
| 0.75 | 1.50 | 0.7497 | 0.7500 |
| 1.00 | 2.00 | 1.0000 | 1.0000 |

**`P(flip) = min(|Δ|/s, 1)` to four decimals.** So at `step_ratio = 1`, *half* those weights flip, not none. The plan's phrasing "values below 1 are below the quantization noise floor" is therefore a statement about probability, not a deterministic threshold, and **the fraction of step-ratios under 1 is not the fraction erased.** This is now a closed-form prediction (`predicted_flip_rate`) checked against the measured flip rate as an internal consistency test; divergence would indicate the delta is correlated with bin position, which is a finding rather than a bug.

*Correction 2: `retention_ratio` is unbounded above and non-monotone, and reads backwards for small deltas.*

Measured on a random 8x512 base, int4 g128, sweeping delta magnitude:

| mean \|Δ\|/s | regime | retention_ratio | cosine | flip rate | rel. error | projection |
|---|---|---|---|---|---|---|
| 0.0002 | fixed | **95.463** | 0.015 | 0.001 | 95.453 | 1.407 |
| 0.0002 | adaptive | **95.463** | 0.015 | 0.001 | 95.453 | 1.419 |
| 0.0023 | fixed | 20.540 | 0.066 | 0.004 | 20.499 | 1.359 |
| 0.0231 | fixed | 5.210 | 0.177 | 0.023 | 5.128 | 0.922 |
| 0.2309 | fixed | 1.648 | 0.583 | 0.227 | 1.339 | 0.961 |
| 1.1543 | fixed | 1.021 | 0.950 | 0.727 | 0.321 | 0.970 |
| 2.3087 | fixed | 0.953 | 0.967 | 0.852 | 0.254 | 0.921 |
| 11.5433 | fixed | **0.463** | 0.850 | 0.960 | 0.654 | 0.393 |
| 11.5433 | adaptive | **1.005** | 0.995 | 0.897 | 0.105 | 0.999 |

`retention_ratio = 95.5` at the smallest delta, with `cosine = 0.015`. The cause: when `|Δ| << s`, each weight that flips contributes a *full step* `s` to `Δ_eff`, which bears no relation to the intended `Δ_i`. So `‖Δ_eff‖` exceeds `‖Δ‖` by orders of magnitude while pointing in an essentially random direction.

**A naive reading of `retention_ratio = 95` as "excellent retention" is exactly backwards: the adapter was erased and replaced by uncorrelated noise of far larger magnitude, which is worse than erasure.** The plan's framing — "retention near 1 means the adapter survives, near 0 means quantization ate it" — has no room for this regime, and it is precisely the regime a rank-16 LoRA is predicted to live in.

Two metrics added that read correctly:

- `relative_error` = `‖Δ_eff − Δ‖/‖Δ‖`, with **1.0 as the exact erasure baseline** (`Δ_eff = 0` gives exactly 1). Below 1 = partially transmitted; at 1 = erased; above 1 = replaced by larger uncorrelated noise.
- `projection_coefficient` = `⟨Δ_eff, Δ⟩/‖Δ‖²`.

`cosine` is monotone across four decades (0.015 → 0.18 → 0.58 → 0.95) where `retention_ratio` is not, so **cosine is the honest headline for the rank sweep.**

*Third observation, unplanned and interesting:* `projection_coefficient` stays near 1 (0.92 to 1.42) even where cosine collapses to 0.015. The delta biases which way each weight rounds, so it survives *in expectation* while being destroyed per-weight — quantization behaves as a noisy but roughly unbiased channel, i.e. dithering. **This is a candidate mechanism for aligned behaviour surviving Phase 1 despite low numerical retention,** and it is a prediction we can test rather than a post-hoc excuse.

*Fourth: `fixed_scale` clips for large deltas.* At mean `|Δ|/s = 11.5`, fixed gives 0.463 against adaptive's 1.005, because `W+Δ` exceeds the range of W's own grid and saturates against the clamp. `fixed_scale` is a valid instrument only while Δ is small relative to W's range. That is our regime of interest, but cross-regime comparison at large delta is confounded by clipping and must not be reported as a mechanism difference.

*Grid-shift artifact confirmed real.* Constructed case: one weight per group carries a large delta that moves the group max, all other 127 weights have `Δ_i` exactly zero. Under `fixed_scale` at most the perturbed weight can change. Under `adaptive_scale` weights with exactly zero delta do change, and `grid_shift_fraction_zero_delta > 0`. Max's confound is real and is now measured rather than assumed.

**Verdict:** WORKED, with two corrections to the planned metrics.

**What we learned:**

1. The fixed/adaptive split was necessary. The artifact exists and is measurable.
2. `|Δ| < s/2` is a probabilistic, not deterministic, erasure condition, with `P(flip) = min(|Δ|/s, 1)`.
3. `retention_ratio` must never be reported bare. It is unbounded above and reads inverted in the small-delta regime.
4. `cosine` is the metric with the clean dose-response curve; `relative_error` is the one with an interpretable erasure baseline at 1.0.
5. Quantization looks like an unbiased noisy channel for sub-step deltas. Testable Phase 1 hypothesis.
6. `fixed_scale` is only valid for small deltas; it clips otherwise.
7. Negative knowledge: none of these are bugs in `quantsim.py`, which stayed bit-exact against gptqmodel throughout. All four are properties of the *metric definitions* in the plan.

**Plan impact:**

- `cosine` becomes the primary reported quantity for the rank sweep; `retention_ratio` is reported alongside it with the caveat, never alone.
- `relative_error` added to every table, since 1.0 is the erasure reference line and figures need it.
- Step-ratio distribution is reported with the `P(flip) = |Δ|/s` prediction overlaid, not as a hard threshold count.
- `fixed_scale` results are only claimed in the small-delta regime; the clipping limitation goes in Limitations.
- The unbiased-channel observation is added as an explicit Phase 1 hypothesis rather than left as an anecdote.
- GATE 0's criterion is stated in terms of bit-flip rate, which is unaffected by all of the above.

**Artifacts:** `src/ar/retention.py`, `tests/test_retention.py` (16 tests), `src/ar/quantsim.py` (refactored for `compute_params`/`apply_params`). Probe scripts in session scratchpad; the numbers above are reproduced by the test suite.

---

## [2026-07-29] EXP-005: GGUF K-quant validation gap, logged before it becomes a surprise

**Phase:** 0

**Question:** How will the GGUF Q4_K_M and Q3_K_M conditions on the Phase 0 grid be validated, given that gptqmodel cannot serve as their reference?

**Setup:** No experiment run. This entry records a known, unclosed validation gap and its plan, per the rule that dead ends and pending decisions belong in the log even when no code was written.

**Command:** None.

**Result:** The Phase 0 grid (plan §1.3) includes **GGUF Q4_K_M and Q3_K_M**. Neither is a group-wise affine scheme, so nothing in `quantsim.py` covers them and the gptqmodel cross-check in EXP-003 says nothing about them.

K-quants use **block-wise quantization with a super-block scale**: blocks of 32 weights each carry their own quantized scale and minimum, and those per-block scales are themselves quantized against a super-block scale spanning 8 blocks (256 weights). Q4_K_M additionally applies mixed precision across tensor types. Consequences:

1. There is no single per-group step size `s`. The effective step for a weight depends on its block scale *after that scale has itself been quantized*. Every metric in `retention.py` takes `s` as given, so `step_per_weight`, `step_ratio`, `subthreshold_fraction`, and `predicted_flip_rate` have no direct analogue without a definition decision.
2. `fixed_scale` is harder to define. Holding "the grid" fixed means holding both block scales and super-block scales fixed, which is a deeper intervention than reusing one `scale` tensor.
3. CLAUDE.md is explicit: do not hand-roll K-quants. The reference must be llama.cpp's own quantizer, with tensors read back.

**Verdict:** INCONCLUSIVE — gap identified, not yet closed.

**What we learned:** The EXP-003 validation does **not** generalize to the GGUF arm. Any retention number for Q4_K_M or Q3_K_M produced before the plan below is executed would be unvalidated, and rule 8 forbids using it.

**Plan impact:** GGUF is now explicitly a *separate validation track* rather than another row in the precision axis. Steps, in order:

1. Build `llama.cpp` with `GGML_CUDA=1`. **Flagged as a Windows risk:** it is a CMake/MSVC build and the first genuinely compiled dependency in Phase 0. If it fights, this whole arm moves to WSL2.
2. Quantize one small model to Q4_K_M and Q3_K_M with `llama-quantize`, then read the tensors back with `gguf-py` and dequantize them to BF16.
3. Decide and **document** the step-size definition for K-quants before computing any metric. Current proposal: use the *effective* per-weight step implied by the weight's block scale after that scale has itself been quantized, so the super-block quantization is included rather than idealized away. This is a judgement call and belongs in the paper's method section, not in a code comment.
4. Validate by round-trip: our dequantization of the GGUF tensors must match llama.cpp's own dequantization bit-for-bit on at least one tensor. That is the K-quant analogue of EXP-003 and it gates the arm.
5. Only then compute retention for the GGUF conditions.

**Decision required from Max:** GGUF is the lowest-value arm per unit of effort — two conditions out of the precision axis, gated behind a compiled dependency, a Windows build risk, and a metric-definition judgement call. Amendment 1 already set the cut priority as "cut precisions before ranks." **Recommendation: defer GGUF until the affine grid (INT4/INT8 x group sizes x ranks 4-128) is complete, and drop it entirely if time is short.** The rank curve on validated affine quantization is the paper; GGUF is a generality check on it.

**Artifacts:** None yet. This entry is the artifact.

---

## [2026-07-30] EXP-006: Verifying the quantization-channel model, the alpha convention, and the layer-output prediction

**Phase:** 0

**Question:** Do the closed-form channel relations hold, does the LoRA alpha convention really reverse the rank trend, and does layer-output error average down with `d_in`?

**Setup:** All synthetic, INT4 g128 asymmetric, `fixed_scale` regime, float32. Random normal base weights; deltas either random normal or `(α/r)·B·A` with iid factors. Every relation checked numerically before being written into Amendment 3, because registering an unverified derivation would be worse than registering none.

**Command:** Probe scripts in session scratchpad; all results now pinned by `tests/test_retention.py` (74 tests pass).

**Result:**

*1. `cosine · retention_ratio ≡ projection_coefficient` is an exact identity, not an approximation.*

| \|Δ\| scale | cos·ret | projection | difference |
|---|---|---|---|
| 0.001 | 1.2457370043 | 1.2457362413 | 7.6e-07 |
| 0.01 | 1.0177154081 | 1.0177154541 | 4.6e-08 |
| 0.1 | 0.9830087378 | 0.9830088019 | 6.4e-08 |
| 1.0 | 0.9240188116 | 0.9240186214 | 1.9e-07 |

Agreement at float32 epsilon. Algebraically `cos·ret = [⟨Δ_eff,Δ⟩/(‖Δ_eff‖‖Δ‖)]·[‖Δ_eff‖/‖Δ‖] = ⟨Δ_eff,Δ⟩/‖Δ‖²`. So the proposed test "assert `cos·ret ≈ 1`" is precisely a test of channel unbiasedness, not of two quantities happening to agree.

*2. Unbiasedness holds; its estimator is noisy in flips, not weights.*

| N weights | flips | projection |
|---|---|---|
| 8,192 | 23 | 1.2706 |
| 65,536 | 146 | 0.9759 |
| 524,288 | 1,296 | 1.0875 |
| 2,097,152 | 4,755 | 0.9650 |

The 1.407 seen in EXP-004 was a 4-flip sample, not a real bias.

*3. The sqrt law needs a shape term; the proposed form is low by √(π/2).*

| mean\|Δ\|/s | cosine measured | `sqrt(\|Δ\|/s)` | distribution-free |
|---|---|---|---|
| 0.00023 | 0.0200 | 0.0153 | 0.0191 |
| 0.00070 | 0.0314 | 0.0265 | 0.0330 |
| 0.00234 | 0.0597 | 0.0484 | 0.0603 |
| 0.00702 | 0.1046 | 0.0838 | 0.1044 |
| 0.02342 | 0.1894 | 0.1530 | 0.1907 |
| 0.07032 | 0.3294 | 0.2652 | 0.3303 |
| 0.23397 | 0.5983 | 0.4837 | 0.6026 |

`cosine ≈ sqrt(mean(Δ²)/(s·mean|Δ|))` is accurate to 2–3%. The simple form is 25% low throughout, exactly `√(π/2) = 1.2533`. The dropped factor `mean(Δ²)/mean|Δ|²` is a tail-shape statistic, so deviation from the Gaussian constant measures the delta's kurtosis rather than "structure" generically.

*4. The alpha convention reverses the rank trend. Confirmed.*

4096×4096, ranks 4→128:

| rank | `α=2r`: mean\|Δ\|/s | cosine | rel_err | `α=16`: mean\|Δ\|/s | cosine | rel_err |
|---|---|---|---|---|---|---|
| 4 | 0.0438 | 0.2761 | 3.449 | 0.0876 | 0.3906 | 2.337 |
| 8 | 0.0641 | 0.3243 | 2.895 | 0.0641 | 0.3243 | 2.895 |
| 16 | 0.0919 | 0.3820 | 2.397 | 0.0460 | 0.2701 | 3.528 |
| 32 | 0.1314 | 0.4529 | 1.949 | 0.0328 | 0.2266 | 4.250 |
| 64 | 0.1865 | 0.5378 | 1.554 | 0.0233 | 0.1902 | 5.107 |
| 128 | 0.2638 | 0.6372 | 1.199 | 0.0165 | 0.1593 | 6.114 |

Ratio r=128/r=4: **2.308 vs predicted `(32)^{1/4}` = 2.378** under `α=2r`; **0.408 vs predicted `(32)^{-1/4}` = 0.420** under fixed α. Both within 3%.

Note: even at rank 128 under `α=2r`, `relative_error = 1.199 > 1` — the delta is still replaced by noise larger than itself across the whole rank range.

*5. The `1/√d_in` layer-output prediction is false for random inputs.*

With `|Δ|/s` held at 0.05 so the comparison is unconfounded (a first attempt varied delta magnitude with `d_in` and drove `fixed_scale` into clipping — misconfiguration, rerun):

| d_in | weight cosine | output cosine, random x | suppression |
|---|---|---|---|
| 256 | 0.2818 | 0.2819 | 1.00 |
| 1024 | 0.2810 | 0.2815 | 1.00 |
| 4096 | 0.2811 | 0.2811 | 1.00 |
| 8192 | 0.2821 | 0.2828 | 1.00 |

No dimensional averaging whatsoever. The adapter's effect sums with the same `√d_in` factor as the error; they cancel.

*6. The real effect is rank-mediated and subspace-conditional.*

| d_in | rank | weight cos | subspace-x cos | suppression | `d_in/r` |
|---|---|---|---|---|---|
| 256 | 16 | 0.2818 | 0.7626 | 3.03 | 16 |
| 1024 | 16 | 0.2810 | 0.9160 | 8.56 | 64 |
| 4096 | 16 | 0.2811 | 0.9771 | 31.41 | 256 |
| 8192 | 16 | 0.2821 | 0.9889 | 64.68 | 512 |
| 4096 | 4 | 0.2965 | 0.9942 | 121.66 | 1024 |
| 4096 | 64 | 0.2785 | 0.9185 | 8.86 | 64 |
| 4096 | 256 | 0.2784 | 0.7667 | 3.09 | 16 |

Suppression in `(1−cos)` tracks `d_in/r` with constant ≈ 1/8, i.e. amplitude gain `√(d_in/r)` = **16× at d_in 4096, rank 16, not 64×**.

**Verdict:** WORKED, with two corrections to the proposed derivation.

**What we learned:**

1. The channel model holds and is now a stated contribution rather than an anecdote.
2. `cos·ret ≡ projection` is exact. Framing it as an approximation would misdescribe the test.
3. The sqrt law is off by `√(π/2)` without the shape term, and that term is itself the useful diagnostic for trained-vs-synthetic adapters.
4. The alpha convention genuinely reverses the sign of the rank trend, to within 3% of the `r^(±1/4)` prediction in both directions. Both conventions must be in the main grid.
5. **`1/√d_in` output averaging does not exist for random inputs.** The correct mechanism is rank-mediated coherence, and the factor is `√(d_in/r)`. Had this been registered as proposed, Phase 1 would have tested a false prediction and any apparent confirmation would have been coincidence.
6. Two rank effects oppose each other: higher rank improves weight-level retention under `α=2r` but reduces output-level amplification. Which dominates is an empirical Phase 1 question.
7. Negative knowledge: the first output probe was confounded by letting delta magnitude vary with `d_in`, pushing `fixed_scale` into its clipping regime. Logged rather than quietly rerun.

**Plan impact:** Amendment 3 written, covering the channel model, both alpha conventions in the main grid, the corrected Phase 1 prediction with layer-output fidelity as a new metric, the `retention_ratio` specification error for the Method section, and the GGUF Phase 0/Phase 1 split.

**Artifacts:** `tests/test_retention.py` (5 new tests pinning all relations above), `PROJECT-EXECUTION-PLAN-v2.md` Amendment 3.

---

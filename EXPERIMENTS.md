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

## [2026-07-30] EXP-007: First real measurement — public Qwen3-8B LoRA at INT4 g128

**Phase:** 0

**Question:** How much of a real, publicly published LoRA adapter survives INT4 g128 quantization when merged into Qwen3-8B?

**Setup:**

- **Adapter:** `adamkarvonen/Qwen3-8B-taboo-smile_50_mix`, r=32, alpha=64 (**alpha/r = 2**), dropout 0.05, targeting all seven projection types. An SFT behavioural fine-tune (the model conceals a secret word), trained with TRL. Chosen over four alternatives because it is a genuine behaviour-modifying tune with a clean config; the others are listed in the session log and **all five surveyed adapters used alpha/r = 2**, the convention under which retention *improves* with rank.
- **Base:** `Qwen/Qwen3-8B`, BF16, 36 layers, hidden 4096, intermediate 12288, GQA with 8 KV heads.
- **Layers sampled:** 0, 12, 24, 35 (depth profile). All 7 module types at each.
- **Quantization:** INT4, group size 128, all three schemes. Both scale regimes.
- **Compute:** RTX 5090 (sm_120) resolved by capability. Base weights range-read from remote safetensors shards, ~1.5 GB of network instead of a 16 GB model download. 69 s wall time, 168 records.

**Command:**

```powershell
conda run -n retention python scripts/measure_public_adapter.py
```

**Result:**

*Headline, INT4 g128 asymmetric, `fixed_scale` (mechanism-isolating), by module type:*

| module | cosine | rel_err | bit-flip | projection | frac \|Δ\|<s/2 | retention_ratio | median \|Δ\|/s |
|---|---|---|---|---|---|---|---|
| q_proj | 0.1399 | 7.189 | 0.0116 | 0.993 | 1.000 | 7.259 | 0.0083 |
| k_proj | 0.1281 | 7.835 | 0.0102 | 0.995 | 1.000 | 7.899 | 0.0075 |
| v_proj | 0.1309 | 7.527 | 0.0095 | 0.991 | 1.000 | 7.592 | 0.0067 |
| o_proj | 0.1456 | 6.808 | 0.0119 | 0.991 | 1.000 | 6.881 | 0.0084 |
| gate_proj | **0.1696** | 5.982 | 0.0153 | 0.992 | 1.000 | 6.066 | 0.0106 |
| up_proj | 0.1457 | 6.800 | 0.0119 | 0.992 | 1.000 | 6.873 | 0.0088 |
| down_proj | **0.1022** | 9.708 | 0.0061 | 0.993 | 1.000 | 9.759 | 0.0045 |

**Pooled: bit-flip rate 1.09%, cosine 0.137, relative_error 7.41.**

*Step-ratio distribution `|Δ|/(s/2)`, pooled:*

| p1 | p25 | p50 | p75 | p95 | p99 | max p99 over records |
|---|---|---|---|---|---|---|
| 0.0003 | 0.0071 | 0.0156 | 0.0296 | 0.0638 | 0.1020 | 0.2066 |

**100.00% of weights are sub-threshold.** Not a single weight anywhere in the sample reaches even a quarter of the half-step. The median delta is about 1/128 of a step size.

*Channel model, validated on real trained weights for the first time:*

| quantity | value |
|---|---|
| measured bit-flip rate | 0.0109 |
| predicted `mean(min(\|Δ\|/s,1))` | **0.0109** |
| `cosine × retention_ratio` | 0.9924 |
| `projection_coefficient` | 0.9924 |

The closed form from EXP-006 predicts the real adapter's flip rate to four decimals, and the channel is unbiased on real weights.

*Regime comparison and depth profile (asymmetric):*

| layer | fixed cos | adapt cos | fixed flip | adapt flip | grid shift | scale shift |
|---|---|---|---|---|---|---|
| 0 | 0.1188 | 0.1178 | 0.0088 | 0.0174 | 0.8497 | 0.9999 |
| 12 | 0.1301 | 0.1289 | 0.0097 | 0.0189 | 0.8445 | 0.9999 |
| 24 | 0.1471 | 0.1459 | 0.0126 | 0.0234 | 0.8381 | 0.9999 |
| 35 | 0.1537 | 0.1525 | 0.0126 | 0.0234 | 0.8424 | 0.9999 |

*Code flips versus value changes, the reason both are logged:*

| regime | code_flip_rate | value_change_rate |
|---|---|---|
| fixed_scale | 0.0109 | 0.0109 |
| adaptive_scale | 0.0208 | **0.8546** |

*Convention comparison (fixed_scale, pooled):*

| scheme | cosine | rel_err | bit-flip | retention_ratio |
|---|---|---|---|---|
| asymmetric | 0.1374 | 7.407 | 0.0109 | 7.476 |
| symmetric_gptq | 0.1319 | 8.723 | 0.0120 | 8.796 |
| symmetric_awq | 0.1248 | 8.264 | 0.0097 | 8.327 |

**Verdict:** WORKED. This is the project's core hypothesis, measured.

**What we learned:**

1. **A real published LoRA is almost entirely erased by INT4 g128 merge-then-quantize.** Only **1.09%** of weights change their stored integer at all. Cosine between intended and effective delta is **0.137**. GATE 0's strong-finding threshold was a bit-flip rate under ~50%; the measured value is **1.09%**, roughly fifty times below it.
2. **`relative_error = 7.41` means the delta is not merely erased, it is replaced by uncorrelated noise about seven times its own size.** The erasure baseline is 1.0. Had we reported `retention_ratio = 7.48` bare, as the original plan specified, it would have read as excellent retention. EXP-004's correction was load-bearing for this exact number.
3. **The scale-shift confound is enormous and Max's correction was necessary.** `scale_shift_fraction = 0.9999`: merging the adapter changes the step size of essentially *every group in the model*. Under `adaptive_scale` the flip rate roughly doubles (1.09% → 2.08%) and **85.46% of weights change their dequantized value**, versus 1.09% under `fixed_scale`. Almost all of that is the grid moving, not the adapter arriving. Reporting only the deployment-realistic number would have overstated transmission by a large factor.
4. **Code flips and value changes must be logged separately.** Under `adaptive_scale` they differ by a factor of 41 (0.0208 vs 0.8546). A single "did the weight change" boolean would have been uninterpretable.
5. **The channel model predicts a real adapter's flip rate to four decimals.** This moves it from a synthetic curiosity to a validated instrument, and gives the paper an analytic backbone rather than a pile of measurements.
6. **Module profile:** `gate_proj` retains most (cosine 0.170), `down_proj` least (0.102). `down_proj` also has the smallest median `|Δ|/s` (0.0045), so the ordering follows delta magnitude relative to step size, consistent with the channel model rather than anything module-specific.
7. **Depth profile:** retention rises monotonically with depth, cosine 0.119 (layer 0) → 0.154 (layer 35). Later layers survive better.
8. **Convention matters measurably.** `asymmetric` 0.1374 vs `symmetric_awq` 0.1248 — a 10% spread in cosine on the same adapter. Whether your adapter survives depends partly on which toolchain you used, which is exactly the kind of finding Amendment 2.5 anticipated.
9. Negative knowledge: **nothing broke in `PeftModel`/`load_peft_weights` under transformers 5.14.1 with peft 0.20.0.** `from_pretrained` keeps its documented signature and `load_peft_weights` reads adapter tensors without instantiating a base model. **No pinning needed.**

**Caveats, stated so the number is not over-read:**

- One adapter, one rank (32), four sampled layers out of 36, one base model. This is a single point, not the rank curve that Amendment 1 made the headline.
- The adapter is a behavioural SFT tune, **not** a safety or alignment tune. The alignment framing needs an alignment adapter, which is still open.
- `alpha/r = 2` here, the convention under which retention is *most favourable*. Under fixed alpha at this rank the result would be worse.
- **Numerical erasure is not yet behavioural erasure.** EXP-006 predicts that layer-output fidelity on inputs inside the adapter's active subspace is far higher than weight-level cosine (amplitude gain `√(d_in/r)` ≈ 11 at d_in 4096, r 32). This measurement is fully consistent with behaviour surviving, and Phase 1 is what decides it. Claiming behavioural erasure from these numbers would be exactly the overreach the phase structure exists to prevent.

**Plan impact:** None to the structure. GATE 0's numerical arm is met with room to spare on a single adapter. The rank sweep and the alignment-adapter question remain the open items before GATE 0 can be called.

**Artifacts:**
- `results/raw/phase0/public_adapter/adamkarvonen__Qwen3-8B-taboo-smile_50_mix/records.jsonl` (168 records)
- `.../manifest.json` (torch, CUDA, driver, package versions, git SHA, resolved device)
- `scripts/measure_public_adapter.py`, `src/ar/manifest.py`

---

## [2026-07-30] EXP-008: GATE 0 closeout — six adapters, 36 layers, synthetic sweep, and a corrected depth trend

**Phase:** 0

**Question:** Does the EXP-007 result hold across adapters, ranks, alpha conventions, and base models, and do the registered predictions of Amendment 4 survive measurement?

**Setup:** Six public adapters spanning ranks 16/32/64/128, both alpha conventions, two base models, and four training regimes (behavioural SFT, DPO, interpretability probe, safety). INT4 g128, all three schemes on the 4-layer runs, asymmetric only on the 36-layer depth run. Plus a synthetic rank sweep (ranks 4–128, 3 seeds, both conventions) and a dose-response over four decades, on a real Qwen3-8B `q_proj` base.

**Command:**

```powershell
conda run -n retention python scripts/measure_public_adapter.py --adapter <repo>
conda run -n retention python scripts/measure_public_adapter.py --layers all --schemes asymmetric
conda run -n retention python scripts/synthetic_sweep.py
conda run -n retention python analysis/summarise.py
```

**Result:**

*1. GATE 0 headline. Six adapters, INT4 g128 asymmetric, `fixed_scale`. CIs bootstrapped over layers.*

| adapter | base | r | α/r | layers | cosine | 95% CI | bit-flip | rel_err |
|---|---|---|---|---|---|---|---|---|
| taboo-smile | Qwen3-8B | 32 | 2 | 36 | 0.1380 | [0.1355, 0.1405] | 0.0122 | 7.293 |
| taboo-gold | Qwen3-8B | 32 | 2 | 4 | 0.1389 | [0.1248, 0.1531] | 0.0111 | 7.350 |
| taboo-ship | Qwen3-8B | 32 | 2 | 4 | 0.1409 | [0.1279, 0.1539] | 0.0114 | 7.233 |
| ao-v3-best-dpo-halluc | Qwen3-8B | 128 | **0.125** | 4 | 0.1512 | [0.1416, 0.1625] | 0.0133 | 6.694 |
| latentqa | Qwen3-8B | 64 | 2 | 4 | 0.2760 | [0.2548, 0.2996] | 0.0388 | 3.578 |
| **responsible-ai-safety** | **Llama-3.1-8B** | 16 | 2 | 4 | **0.3298** | [0.3069, 0.3664] | 0.0619 | 2.904 |

**Every adapter is far past the erasure baseline.** `relative_error` ranges 2.90 to 7.35 against a baseline of 1.0. The *best-retained* adapter still receives a delta ~2.9x its own size in uncorrelated noise. Bit-flip rates span 1.1%–6.2%.

*2. Rank does NOT predict retention on trained adapters.* Ordering by rank under α/r = 2: r=16 → 0.330, r=32 → 0.138/0.139/0.141, r=64 → 0.276. **Non-monotone, and the r=16 adapter retains best.** The synthetic `r^(1/4)` law (EXP-006, confirmed below) does **not** transfer to trained adapters, because optimization rather than parameterization sets effective magnitude. Amendment 3.2 registered exactly this caveat in advance.

*3. The channel model predicts every adapter's bit-flip rate to within 2.3%.*

| adapter | flip measured | flip predicted | ratio | projection |
|---|---|---|---|---|
| responsible-ai-safety (Llama) | 0.06191 | 0.06335 | 0.977 | 0.974 |
| taboo-gold | 0.01114 | 0.01116 | 0.999 | 0.992 |
| taboo-ship | 0.01139 | 0.01142 | 0.998 | 0.993 |
| taboo-smile | 0.01220 | 0.01233 | 0.990 | 0.988 |
| latentqa | 0.03882 | 0.03912 | 0.992 | 0.991 |
| ao-v3-best-dpo-halluc | 0.01334 | 0.01338 | 0.997 | 0.993 |

Across two base models, four ranks, both alpha conventions, and four training regimes. **This is the strongest result in Phase 0.** Retention is not predicted by rank, architecture, or training method — it is predicted by `|Δ|/s`, and the closed form converts that into a bit-flip rate with under 2.3% error every time.

*4. Registered predictions P1 and P2 confirmed on synthetic adapters (Amendment 4.2, committed before the sweep).*

| convention | quantity | fitted exponent | predicted |
|---|---|---|---|
| α = 2r | weight SNR | +0.286 | +0.25 |
| α = 2r | output SNR, subspace x | **−0.182** | **−0.25** |
| α = 2r | output SNR, generic x | +0.286 | tracks weight |
| α = 16 | weight SNR | −0.275 | −0.25 |
| α = 16 | output SNR, subspace x | **−0.744** | **−0.75** |

**P1's central claim holds: weight-space and output-space fidelity disagree in sign under α=2r.** Weight SNR climbs 0.255 → 0.688 across r=4→128 while subspace output SNR falls 7.34 → 3.94. Generic-input SNR tracks weight SNR to three decimals at every rank, re-confirming there is no dimensional averaging.

Exponents under α=2r are compressed by 0.04–0.07. Decomposable: fitted amplification is −0.465 against −0.5, and weight SNR +0.286 against +0.25. Both consistent with the noise-dominated approximation weakening as SNR approaches 0.69 at r=128. Signs unaffected.

**P3 remains untested.** Synthetic gives SNR_out = 4.93 at r=32 against the predicted 1.55, because `C` was calibrated from the real adapter while the synthetic delta magnitude is arbitrary — the caveat registered in advance. Testing P3 needs real adapters across ranks *with matched training*, which the six adapters do not provide since their magnitudes vary independently of rank.

*5. Dose-response validates the channel across four decades (rank 32, α=2r).*

| mean \|Δ\|/s | flip measured | flip predicted | cosine | cosine predicted |
|---|---|---|---|---|
| 0.00109 | 0.0015 | 0.0011 | 0.0382 | 0.0402 |
| 0.01087 | 0.0108 | 0.0109 | 0.1263 | 0.1271 |
| 0.10866 | 0.1076 | 0.1086 | 0.4000 | 0.4019 |
| 0.32598 | 0.3153 | 0.3187 | 0.6847 | 0.6961 |
| 1.08659 | 0.6919 | 0.7000 | 0.9400 | 1.0000 |

Tail shape `mean(Δ²)/mean|Δ|²` is a flat **1.5962** across all magnitudes versus the Gaussian reference `π/2 = 1.5708` — products of Gaussians are marginally heavier-tailed, as expected.

*6. **CORRECTION to EXP-007 finding 7: the depth trend was a sampling artifact.***

EXP-007 reported "retention rises monotonically with depth, cosine 0.119 → 0.154" from four layers (0, 12, 24, 35). The full 36-layer profile shows a **weaker and non-monotone** trend:

| | 4-layer sample (EXP-007) | all 36 layers |
|---|---|---|
| shape | monotone rising | non-monotone |
| magnitude | 0.119 → 0.154, **+29%** | first quartile 0.1322 → last quartile 0.1446, **+9.4%** |
| all-layer mean | — | 0.1380, 95% CI over layers [0.1355, 0.1405] |

The sampled layers happened to fall on a rising sequence. The full profile contains structure the sample missed entirely, most visibly **a spike at layers 1–3 where the bit-flip rate is 2.5–2.7% against ~1.0% elsewhere** — nearly triple, and invisible at 4-layer resolution. Layer 0 is the global minimum (0.1188), layers 4–8 dip back to ~0.128.

**The depth trend is real but roughly a third the size reported, and its shape was wrong.** EXP-007's numbers stand as recorded; this entry supersedes its finding 7.

*7. Three taboo replicates agree tightly.* Same recipe, same rank, differing only in the secret word: 0.1380, 0.1389, 0.1409. Between-adapter spread is under 2% relative — a genuinely controlled population, which supports using the ~20-variant Taboo family as Phase 1's replicate set.

*8. Convention, paired: 4.5% maximum deviation.*

| scheme | cosine | bit-flip | rel_err |
|---|---|---|---|
| asymmetric | 0.1874 | 0.0226 | 6.082 |
| symmetric_gptq | 0.1868 | 0.0244 | 6.766 |
| symmetric_awq | 0.1790 | 0.0213 | 6.504 |

Paired on the 168 (adapter, layer, module) cells present under all three schemes.

*9. Module profile, pooled over six adapters:* `gate_proj` retains most (0.196), `down_proj` least (0.130), ordering identical to median `|Δ|/s` (0.0155 vs 0.0081). Module differences are entirely a magnitude effect, not architectural.

*10. Scale regime, pooled:* `fixed_scale` cosine 0.1628 / flips 0.0176; `adaptive_scale` cosine 0.1616 / code flips 0.0313 / **value changes 0.8482**. `scale_shift_fraction` 0.9999, `grid_shift_fraction` 0.8307. Confirms EXP-007 across all six adapters.

**Verdict:** WORKED. GATE 0's numerical arm is met.

**What we learned:**

1. **Near-total weight-space erasure is not adapter-specific.** Six adapters, two base models, four ranks, two alpha conventions, four training regimes — all far past the erasure baseline.
2. **Rank does not predict trained-adapter retention; `|Δ|/s` does.** This reshapes the paper: the rank curve is a *synthetic* result establishing the mechanism, and the real-adapter story is that trained magnitude, which nobody reports, is what determines survival.
3. **The channel model is the central contribution.** Sub-2.3% prediction of bit-flip rate across every adapter tested, from a closed form with no fitted parameters.
4. **P1 confirmed: weight-space and output-space fidelity move in opposite directions with rank.** Registered before measurement.
5. **A 4-layer depth sample produced a trend three times too large and the wrong shape.** Max flagged the 4-layer depth number as the weakest claim; it was. Two of the three sampled layers sat on a locally rising stretch, and the layers 1–3 anomaly was invisible.
6. **Analysis bug caught: pooling unpaired records inverted the convention ordering.** The first aggregation pooled all records, including the asymmetric-only 36-layer run, which dragged asymmetric's mean down and made `symmetric_gptq` appear to retain best. Pairing on identical cells reverses it. Fixed in `analysis/summarise.py` with `paired_on_schemes` and `one_run_per_adapter`; had it gone unnoticed the paper would have carried a backwards claim about which toolchain preserves adapters best.
7. **Process bug: the 36-layer run overwrote the artifact EXP-007 cites**, because the output path keyed only on adapter name. Recovered from git, verified identical by SHA-256 (`356B9A2F...`), and both runs now live under run-shape subdirectories. `--out-subdir` added. The artifact path in EXP-007 has therefore moved to `.../L4_asymmetric-symmetric_gptq-symmetric_awq/records.jsonl`; content is byte-identical to what that entry reported.
8. **The safety adapter retains best, but this is confounded and must not be read as "safety adapters survive better."** It sits on a different base model (Llama-3.1-8B, different weight scale and therefore different step size) and its median `|Δ|/s` is roughly 5x the taboo adapters'. The channel model attributes its higher cosine entirely to that magnitude. Attributing it to the safety objective would require matched base and matched magnitude.
9. Negative knowledge: `α/r = 2` in five of six adapters; the sixth uses 0.125. The favourable convention dominates shipped practice and does not rescue retention.

**Plan impact:**

- The rank sweep's role changes. It establishes the mechanism on synthetic adapters, where it is clean and confirms P1/P2. It is **not** the headline for real adapters, because rank does not predict their retention. The headline is the channel model plus the six-adapter spread.
- P3 needs matched-training adapters across ranks. Deferred; noted as untested rather than quietly dropped.
- Taboo family confirmed as the Phase 1 replicate set.

**Artifacts:**
- `results/raw/phase0/public_adapter/*/L*/records.jsonl` — six adapters, 1176 records
- `results/raw/phase0/synthetic/records.jsonl` — 43 records
- `analysis/summarise.py`, `scripts/synthetic_sweep.py`

---

## [2026-07-30] EXP-009: Output-space SNR, bin-position independence, spike decomposition, and a failed prediction

**Phase:** 0

**Question:** Does the DPO adapter show the worst output-space degradation as predicted; are trained deltas independent of quantization bin position; and what drives the layer 1–3 bit-flip spike?

**Setup:** Six adapters, INT4 g128 asymmetric, `fixed_scale`, layers {0, 12, 23}, all 7 modules. Output SNR measured directly (`‖Δx‖/‖(Δ_eff−Δ)x‖`) with probes drawn both generically and from the adapter's own row space, rather than composed from the `√(d_in/r)` law. Effective rank of `A` reported as the participation ratio of its singular values. Bin-position independence tested by Pearson correlation and by a within-group permutation control that shuffles Δ inside each quantization group, destroying any Δ–position association while preserving Δ's marginal distribution exactly.

**Command:**

```powershell
conda run -n retention python scripts/output_space_diagnostics.py
conda run -n retention python scripts/validate_predict.py
```

**Result:**

*1. The registered DPO prediction FAILED.*

| adapter | r | eff. rank | SNR_w | SNR_out generic | **SNR_out subspace** | amplification | `√(d_in/r)` |
|---|---|---|---|---|---|---|---|
| taboo-gold | 32 | 31.8 | 0.1342 | 0.1340 | **2.0013** | 15.02 | 12.83 |
| taboo-smile | 32 | 31.8 | 0.1341 | 0.1338 | **2.0137** | 15.06 | 12.83 |
| taboo-ship | 32 | 31.8 | 0.1366 | 0.1365 | **2.0707** | 15.27 | 12.83 |
| ao-v3-dpo-halluc | 128 | 115.1 | 0.1565 | 0.1564 | **2.4907** | 15.50 | 6.41 |
| latentqa | 64 | 60.8 | 0.2920 | 0.2915 | **5.5641** | 18.64 | 9.07 |
| responsible-ai-safety | 16 | 15.4 | 0.3854 | 0.3854 | **7.6583** | 21.07 | 18.64 |

The DPO adapter was predicted to show the most severe output-space degradation, worst by ~4.8x. **It ranks 4th of 6, and is 23% better than the three taboo adapters.** The prediction is **FAILED**, not partially confirmed.

*2. Why it failed: the `√(d_in/r)` amplification law does not hold for trained adapters.*

Measured amplification is **15.0–21.1 across ranks 16 to 128**, essentially rank-independent, while `√(d_in/r)` varies from 6.41 to 18.64 — a 2.9x range. Ratios of measured to predicted: 1.17 (r=32), 2.05 (r=64), **2.42 (r=128)**, 1.13 (r=16). The error grows systematically with rank.

This is **not** an effective-rank artifact: measured effective rank tracks nominal rank closely (31.8/32, 60.8/64, 115.1/128, 15.4/16), and substituting it changes `√(d_in/eff_r)` by under 5%.

The law was verified on synthetic adapters with iid factors (EXP-006, EXP-008) and does not transfer. **This is the second component of my own registered prediction P1 to fail on trained adapters** — the weight-space `r^(1/4)` half already failed in EXP-008. P1 stands as a synthetic result and must not be stated as a claim about real adapters.

*3. Consequently, output space does NOT reorder the adapters.* Because amplification is roughly constant, output SNR is approximately a fixed multiple of weight SNR, and the two orderings are identical. Answering the second question directly: **output-space retention is not monotone in rank either** (r=16 → 7.66, r=32 → ~2.0, r=64 → 5.56, r=128 → 2.49). The two spaces *agree* for trained adapters. The disagreement predicted by P1 is real but confined to the synthetic regime where magnitude is set by parameterization.

*4. Bin-position independence: confirmed, and this is the real content of the channel claim.*

| adapter | corr(Δ/s, bin offset) | corr(\|Δ\|/s, \|offset\|) | flip real | flip permuted | real/perm |
|---|---|---|---|---|---|
| taboo-smile | 0.00004 | 0.00001 | 0.01033 | 0.01033 | 1.0004 |
| taboo-ship | 0.00002 | 0.00001 | 0.01065 | 0.01066 | 0.9998 |
| taboo-gold | 0.00008 | −0.00009 | 0.01037 | 0.01036 | 1.0009 |
| latentqa | −0.00005 | −0.00002 | 0.03946 | 0.03947 | 0.9997 |
| ao-v3-dpo-halluc | −0.00013 | 0.00008 | 0.01364 | 0.01362 | 1.0010 |
| responsible-ai-safety | −0.00007 | 0.00327 | 0.06332 | 0.06327 | 1.0008 |

Pooled: **max |correlation| = 0.00109**, mean −0.00002. Permutation control: real/permuted flip ratio in **[0.9916, 1.0147]**, mean 1.0005.

`P(flip) = mean(min(|Δ|/s, 1))` is close to an identity when Δ is independent of where `w` sits in its bin. The finding is therefore not "the formula works" but **"trained LoRA deltas carry no information about quantization bin position."** Gradient descent, optimizing a loss that has no knowledge of the deployment quantizer, produces updates statistically orthogonal to the quantization grid. Stated that way it is a claim about training dynamics, and it is what licenses the closed form.

*5. The layer 1–3 spike is a step-size effect, localized to two modules.*

Layer-level decomposition first, which did **not** close: at layers 1–3, `mean|Δ|` is 0.89–0.91x its layer-12 value and `mean s` is 0.78–0.84x, so the ratio of means predicts ~1.13x, but `mean(|Δ|/s)` is **2.9x** higher. That gap can only come from heterogeneity within the layer.

Resolved by going per module:

| layer | module | mean\|Δ\| ×1e4 | mean s ×1e3 | mean\|Δ\|/s | flip | s p50/p1 |
|---|---|---|---|---|---|---|
| 1 | q_proj | 0.9008 | 8.1956 | 0.01155 | 0.0115 | 1.58 |
| 1 | k_proj | 0.7578 | 8.8958 | 0.00898 | 0.0090 | 1.87 |
| 1 | v_proj | 0.7321 | 9.4847 | 0.00793 | 0.0079 | 1.49 |
| 1 | o_proj | 0.7642 | 8.6743 | 0.00917 | 0.0092 | 1.50 |
| 1 | **gate_proj** | 1.1091 | 6.3526 | **0.07314** | **0.0539** | **83.77** |
| 1 | **up_proj** | 0.8439 | **3.8331** | **0.08149** | **0.0768** | 7.07 |
| 1 | down_proj | 0.5179 | 7.7713 | 0.00688 | 0.0052 | 1.54 |
| 12 | (all seven) | 0.55–1.18 | 9.2–10.8 | 0.006–0.012 | 0.006–0.013 | 1.35–2.07 |

**The spike is entirely `gate_proj` and `up_proj` at layer 1**, and in both cases the numerator is *lower* than at layer 12 while the denominator collapses:

- `up_proj` layer 1 has a **globally smaller step size**, mean `s` = 3.83e-3 against 9.67e-3 at layer 12 — a 2.5x narrower weight range.
- `gate_proj` layer 1 has a **heavy small-`s` tail**: its median step is 83.8x its 1st percentile, against 1.4–2.1x for every normal module. A subpopulation of groups with extremely narrow dynamic range drives the layer mean.

**So retention varies across the model because the base model's step size varies, not because the adapter varies.** The adapter's magnitude is comparatively flat; the quantization grid is not.

*6. Diagnostic tool `python -m ar.predict` built and validated.*

| adapter | flip predicted | flip measured | error | cosine predicted | cosine measured | error |
|---|---|---|---|---|---|---|
| responsible-ai-safety | 0.05587 | 0.06191 | −9.8% | 0.3278 | 0.3298 | −0.6% |
| taboo-gold | 0.01070 | 0.01114 | −4.0% | 0.1435 | 0.1389 | +3.3% |
| taboo-ship | 0.01083 | 0.01139 | −4.9% | 0.1441 | 0.1409 | +2.2% |
| taboo-smile | 0.01060 | 0.01220 | −13.2% | 0.1427 | 0.1380 | +3.4% |
| latentqa | 0.03954 | 0.03882 | +1.9% | 0.2960 | 0.2760 | +7.2% |
| ao-v3-dpo-halluc | 0.01458 | 0.01334 | +9.4% | 0.1710 | 0.1512 | +13.1% |

**Mean absolute error 7.2% on bit-flip rate, 5.0% on cosine. Maximum 13.2%.** No GPU, no training, ~130 MB of network from 3 sampled layers. The taboo-smile error is the largest on flip rate because 3-layer sampling misses the layer 1–3 spike, which is itself a finding about where sampling error comes from.

**Verdict:** WORKED for questions 2 and 3. **FAILED** for the registered DPO prediction.

**What we learned:**

1. **The DPO adapter is not the most degraded in output space.** Registered prediction failed; recorded as failed.
2. **The `√(d_in/r)` amplification law does not transfer to trained adapters.** Measured amplification is rank-flat at 15–21x. Both halves of P1 now hold synthetically and fail on real adapters, which is a consistent and interesting pattern: **laws derived under iid parameterization describe synthetic adapters and not trained ones.** That belongs in the paper as a limitation of theory-driven prediction in this area, and it is the third registered prediction corrected by measurement.
3. **Output space does not reorder adapters**, so weight-space retention remains a usable proxy for ranking adapters even though its absolute values understate output fidelity by 15–21x.
4. **Trained deltas carry no information about quantization bin position** — correlation under 0.0011, permutation ratio within 1.5% of unity across six adapters. This is the substantive claim, replacing "the formula matches to 2.3%".
5. **Cross-model retention variation is driven by the base model's step-size distribution, not the adapter's magnitude.** The layer 1–3 spike is `gate_proj`/`up_proj` with anomalously narrow weight ranges.
6. Negative knowledge: effective rank of trained `A` is within 10% of nominal rank for all six adapters, so low-rank collapse is **not** happening and does not explain the amplification anomaly.

**Plan impact:**

- P1 is restated as a synthetic-regime result. It must not be quoted about real adapters.
- The paper needs an explicit subsection on why iid-derived scaling laws fail on trained adapters; three registered predictions have now failed this way.
- `ar/predict.py` is a deliverable in its own right and goes in the README.

**Artifacts:** `results/raw/phase0/output_space/records.jsonl` (134 records), `src/ar/predict.py`, `scripts/output_space_diagnostics.py`, `scripts/validate_predict.py`.

---

## [2026-07-30] EXP-010: CORRECTION to EXP-009 — the amplification law holds, and the DPO prediction was right

**Phase:** 0

**Question:** Was the `√(d_in/r)` amplification law actually refuted by EXP-009, or was that test confounded?

**Setup:** EXP-009 concluded the law fails on trained adapters, based on six adapters whose measured/predicted amplification ratios were 1.13, 1.17, 2.05, 2.42. But those six differ in rank *and* base model *and* training regime *and* alpha convention simultaneously — the same confound that defeated the weight-space rank law in EXP-008. Six points cannot isolate a rank effect.

Two tests:

1. **Matched SVD-truncation.** One adapter (`taboo-smile`), one base, one training run. Its delta is SVD-truncated to r = 4, 8, 16, 32 and **rescaled to the original Frobenius norm**, so magnitude is held fixed and rank is the only variable. Per module (`q_proj`, `gate_proj` at `d_in=4096`; `down_proj` at `d_in=12288`) rather than pooled, since `√(d_in/r)` differs by `√3` between them.
2. **Amplification decomposed.** For a probe `x`, `conc(M, x) = ‖Mx‖²/(‖M‖²_F ‖x‖²/d_in)`, which is 1 for isotropic `x`. Then `amplification = √(conc(Δ)/conc(E))`, separating "signal fails to concentrate" from "error fails to spread".

**Also under test: the probe itself.** EXP-009 drew subspace probes as `coef @ A`, whose covariance is `AᵀA`. That is **not uniform on the row space** — it over-weights `A`'s dominant singular directions. Both are measured here alongside an orthonormal probe built from Δ's right singular vectors.

**Command:**

```powershell
conda run -n retention python scripts/amplification_svd_test.py
conda run -n retention python scripts/output_snr_orthonormal.py
```

**Result:**

*1. With rank as the only variable and a fair probe, the law holds.*

| module | d_in | r | conc(Δ) | conc(E) | amp | `√(d_in/r)` | ratio | amp A-weighted | amp generic |
|---|---|---|---|---|---|---|---|---|---|
| q_proj | 4096 | 4 | 1024.0 | 1.201 | 29.20 | 32.00 | 0.913 | 41.77 | 0.997 |
| q_proj | 4096 | 8 | 513.2 | 1.088 | 21.71 | 22.63 | 0.960 | 39.08 | 0.995 |
| q_proj | 4096 | 16 | 260.2 | 1.040 | 15.82 | 16.00 | 0.989 | 36.32 | 0.997 |
| q_proj | 4096 | 32 | 129.8 | 1.012 | 11.33 | 11.31 | **1.001** | 34.67 | 0.998 |
| down_proj | 12288 | 4 | 3079.8 | 1.228 | 50.09 | 55.43 | 0.904 | 61.13 | 0.993 |
| down_proj | 12288 | 32 | 382.9 | 1.020 | 19.38 | 19.60 | 0.989 | 55.53 | 0.991 |

Fitted exponents: **−0.4569** (q_proj), **−0.4554** (gate_proj), **−0.4574** (down_proj), against a predicted −0.5.

The `√3` prediction between module families is confirmed: at r=4, `q_proj` amplification 29.20 and `down_proj` 50.09, ratio **1.715** against `√3 = 1.732`.

Generic-input amplification is **0.991–1.005 at every rank and module** — no dimensional averaging, confirmed a third time.

*2. The residual deviation is error anisotropy, exactly as hypothesised.*

`conc(E)` is systematically above 1 and falls with rank: **1.20–1.25 at r=4, 1.09–1.12 at r=8, 1.04–1.07 at r=16, 1.01–1.03 at r=32**. Per-weight error variance is `s·|Δ|`, so the error inherits the adapter's magnitude profile and is *not* isotropic — it partially concentrates in Δ's own row space. Empirically `conc(E) ≈ 1 + c/r` with `c ≈ 0.87`, giving the corrected law:

```
amplification = sqrt( (d_in / r) / (1 + c/r) )
```

This is a **refinement, not a refutation**. It explains the entire deviation: the ratio is 0.90 at r=4 where `conc(E)` is largest and 1.00 at r=32 where it vanishes, and the fitted exponent is −0.457 rather than −0.5 for precisely this reason.

*3. My EXP-009 probe was the confound.* The A-weighted probe gives amplification of 33–75, inflated **and nearly rank-insensitive** (34.67 at r=32 vs 41.77 at r=4 for one cell). That is the artifact that produced EXP-009's "rank-flat 15–21x" conclusion.

*4. Re-measuring the six adapters with the orthonormal probe reverses the EXP-009 verdict.*

| adapter | r | SNR_w | **SNR_out** | amp | `√(d_in/r)` | ratio | conc(E) |
|---|---|---|---|---|---|---|---|
| **ao-v3-dpo-halluc** | 128 | 0.1565 | **0.958** | 6.22 | 6.25 | 0.995 | 1.011 |
| taboo-smile | 32 | 0.1341 | 1.627 | 12.37 | 12.50 | 0.990 | 1.019 |
| taboo-gold | 32 | 0.1342 | 1.634 | 12.42 | 12.50 | 0.994 | 1.016 |
| taboo-ship | 32 | 0.1366 | 1.658 | 12.39 | 12.50 | 0.992 | 1.018 |
| latentqa | 64 | 0.2920 | 2.514 | 8.77 | 8.84 | 0.992 | 1.012 |
| responsible-ai-safety | 16 | 0.3854 | 6.017 | 16.58 | 17.99 | 0.922 | **1.416** |

Ratios 0.990–0.995 for five of six. The safety adapter deviates most (0.922) and has by far the largest `conc(E)` at 1.416 — consistent with `1 + c/r` at the lowest rank in the set.

**Ranking by output SNR, worst first: `ao-v3-dpo-halluc` (0.958), taboo family (1.63–1.66), latentqa (2.51), safety (6.02).**

**The registered DPO prediction is CONFIRMED.** It was predicted worst in output space by ~4.8x; measured spread from worst to best is **6.3x**. EXP-009's verdict of FAILED was wrong and is superseded by this entry.

**Verdict:** WORKED — and it corrects a previous entry's conclusion.

**What we learned:**

1. **The `√(d_in/r)` law holds on trained adapters**, to within 11% at worst and 1% at r=32, once rank is isolated and the probe is uniform on the row space. EXP-009's refutation was an artifact of a biased probe compounded by a confounded comparison.
2. **The DPO prediction was right and my measurement was wrong.** `ao-v3-dpo-halluc` has the worst output-space SNR of the six, and at **0.958 it is the only adapter where output-space noise exceeds signal.**
3. **Error anisotropy is real and quantified.** `conc(E) ≈ 1 + 0.87/r`. The error inherits the adapter's magnitude profile because per-weight error variance is `s·|Δ|`. This is a correction term to the law, and it is largest exactly where the law's deviation is largest.
4. **Direct measurement beat composition again — but the measurement has to be right.** EXP-009 was a direct measurement and it was still wrong, because the probe encoded an assumption. The lesson is narrower than "measure, don't compose": it is that the *instrument* needs validating as much as the quantity. An orthonormal basis is the only unbiased probe of a subspace, and `coef @ A` is not one.
5. **P1 is reinstated for trained adapters**, with the `1 + c/r` correction. Only the weight-space half (`r^(1/4)`, EXP-008) genuinely fails on trained adapters, and that failure has a different cause: optimization rather than parameterization setting magnitude.
6. Negative knowledge: pooling across modules would have hidden this. `√(d_in/r)` differs by `√3` between attention and `down_proj`, and per-module reporting is what made the agreement legible.

**Plan impact:**

- EXP-009's sections 1–3 are superseded. Its findings 4 (bin-position independence) and 5 (spike decomposition) are unaffected and stand.
- Amendment 5's Phase 1 predictions were registered on the wrong output-SNR numbers and are reissued in Amendment 6 **before any Phase 1 run**.
- `ar/predict.py`'s output-SNR band was calibrated on the broken amplification measurement and is corrected to use the analytic law with the `1 + c/r` term.

**Artifacts:** `results/raw/phase0/amplification/records.jsonl` (24 records), `results/raw/phase0/output_snr_orthonormal/records.jsonl` (126 records), `scripts/amplification_svd_test.py`, `scripts/output_snr_orthonormal.py`.

---

## [2026-07-30] EXP-011: rsLoRA scaling bug — one adapter's delta was 11.3x too small in every prior entry

**Phase:** 0

**Question:** Is `ao-v3-dpo-halluc`'s `α/r = 0.125` a deliberate choice or an unscaled default? And can the anisotropy correction be derived rather than fitted?

**Setup:** Investigating the α/r question surfaced a bug in our own code. The adapter sets **`use_rslora: true`**, and peft's `LoraLayer.update_layer` scales accordingly:

```python
scaling = lora_alpha / math.sqrt(r)   if use_rslora else   lora_alpha / r
```

`ar.retention.lora_delta` hardcoded `α/r`. At r=128 that understates the merged delta by **√128 = 11.314x**.

**Command:**

```powershell
conda run -n retention python -m pytest tests/ -q
conda run -n retention python scripts/measure_public_adapter.py --adapter ceselder/qwen3-8b-ao-v3-best-dpo-halluc
conda run -n retention python scripts/output_snr_orthonormal.py
conda run -n retention python scripts/anisotropy_form_test.py
```

**Result:**

*1. Scope of the bug: exactly one adapter, verified across all six.*

| adapter | r | α | rsLoRA | correct scaling | we assumed | factor wrong |
|---|---|---|---|---|---|---|
| taboo-smile / ship / gold | 32 | 64 | False | 2.000 | 2.000 | 1.000 |
| latentqa | 64 | 128 | False | 2.000 | 2.000 | 1.000 |
| **ao-v3-dpo-halluc** | 128 | 16 | **True** | **1.414** | **0.125** | **11.314** |
| responsible-ai-safety | 16 | 32 | False | 2.000 | 2.000 | 1.000 |

*2. The α/r question answers itself, and kills a hypothesis before it reached the paper.*

`α/r = 0.125` is **neither** a deliberate low setting **nor** an unscaled default. Under rsLoRA the meaningful quantity is `α/√r = 1.414`, comparable to the other adapters' 2.0. **The adapter is normally configured.**

The hypothesis under consideration — *"shipped adapters carry mismatched α/r and that is what pushes them below output SNR 1"* — was therefore **an artifact of our bug, not a property of shipped adapters**, and would have been a false, directly actionable claim aimed at practitioners.

*3. Corrected measurements invert the adapter's position.*

`q_proj`, INT4 g128, `fixed_scale`:

| | before (buggy) | after |
|---|---|---|
| cosine | 0.1735 | **0.5715** |
| bit-flip | 0.0165 | **0.1810** |
| relative_error | 5.796 | **1.452** |
| mean \|Δ\|/s | 0.0114 | 0.1287 |

Corrected six-adapter output SNR, orthonormal probe:

| rank | adapter | SNR_w | **SNR_out** | amp | `√(d/r)` | ratio |
|---|---|---|---|---|---|---|
| 1 (worst) | taboo-smile | 0.1341 | 1.628 | 12.38 | 12.50 | 0.991 |
| 2 | taboo-gold | 0.1342 | 1.630 | 12.39 | 12.50 | 0.992 |
| 3 | taboo-ship | 0.1366 | 1.657 | 12.38 | 12.50 | 0.991 |
| 4 | latentqa | 0.2920 | 2.525 | 8.80 | 8.84 | 0.996 |
| **5** | **ao-v3-dpo-halluc** | 0.6164 | **3.757** | 6.22 | 6.25 | 0.995 |
| 6 (best) | responsible-ai-safety | 0.3854 | 6.000 | 16.54 | 17.99 | 0.919 |

**The DPO adapter is second-best, not worst.** The registered prediction that it would show the most severe output-space degradation **FAILS**.

*4. The amplification law survives the correction and is now better supported.* Ratios to `√(d_in/r)`: 0.991, 0.991, 0.992, 0.996, **0.995**, 0.919. The corrected DPO point sits at 0.995 — it was 0.995 before too, because amplification is a ratio and largely scale-invariant. EXP-010's law validation used `taboo-smile`, which is not rsLoRA, so its exponents (−0.457, −0.455, −0.457) are unaffected.

*5. An attempted refinement to the anisotropy derivation made it worse, and is recorded rather than deleted.*

With the corrected (11.3x larger) delta, `|Δ|/s` rises to ~0.09–0.13 and the derived form degrades from 0.39% to **2.13%** mean error. The obvious fix is to use the exact per-weight error variance `s|Δ|(1 − |Δ|/s)` instead of its small-delta limit `s|Δ|`. Measured:

| weighting | mean \|err\| | max \|err\| | fitted params |
|---|---|---|---|
| derived, small-delta `\|Δ\|` | **2.13%** | 18.42% | 0 |
| derived, exact variance | 3.96% | 28.86% | 0 |
| fitted `1 + 0.87/r` | 2.89% | 19.71% | 1 |

Restricted to r ≥ 64: 0.44% / 0.58% / 0.61%.

**The "exact" variance is worse.** The two-outcome model behind it assumes a single-step flip; once `|Δ| > s` the code moves more than one step and the error is `|Δ| − s` rather than zero, so the `(1 − |Δ|/s)` factor and the clamp needed to keep it non-negative misprice exactly the heavy-tailed weights that dominate at low truncation rank. The small-delta weighting is kept as the default and the correction is available but off.

**Verdict:** WORKED as a bug hunt. The registered DPO prediction **FAILS** on correct numbers.

**What we learned:**

1. **`use_rslora` must be read from `adapter_config.json`, never assumed.** At r=128 the two scaling rules differ by 11.3x, which is enough to move an adapter from worst to second-best in a six-adapter ranking. Fixed at the root in `lora_delta`, with a test asserting the `√r` ratio and that the default stays non-rsLoRA to match peft.
2. **Every DPO number in EXP-007, EXP-008, EXP-009 and EXP-010 was wrong.** Those entries stand as written; this one supersedes their DPO rows. The five other adapters are unaffected throughout.
3. **The verdict on the DPO prediction has now been stated three times and been wrong twice.** FAILED (EXP-009, biased probe), CONFIRMED (EXP-010, biased delta), and now FAILED (EXP-011, both fixed). Only this verdict rests on numbers with no known defect. The pattern is not that predictions keep failing — it is that **a single adapter with two independent measurement bugs produced three different answers**, and none of the earlier two should have been reported with confidence.
4. **The α/r hypothesis died before publication because the question was asked.** "Is this deliberate or a default?" is what forced a look at the config, which is what surfaced `use_rslora`. Provenance questions about data are worth asking even when the number looks fine.
5. Negative knowledge: attempting to use the exact error variance made the anisotropy fit worse, because the underlying single-flip model breaks down before the variance formula does. The limiting approximation is the flip model, not the variance.
6. The derived, parameter-free form still beats the one-parameter fit (2.13% vs 2.89%), but the margin is much narrower than the 0.39% vs 1.42% measured on the buggy delta. **The earlier claim of near-exactness was flattered by an artificially small delta.**

**Plan impact:**

- Amendment 6's Phase 1 predictions used the buggy DPO output SNR of 0.958 and singled it out as at risk. **That is withdrawn.** With SNR_out = 3.757 it is among the better-preserved adapters. Reissued in Amendment 7.
- **No adapter in the set now has output SNR below 1**, so the "noise exceeds signal" regime is currently unobserved in real adapters. The `ar.predict` warning for it stays, since it is reachable at coarser quantization.
- `ar.predict` now reports effective scaling and which rule produced it.

**Artifacts:** `src/ar/retention.py` (`lora_delta` fix), `tests/test_retention.py` (rsLoRA scaling test), regenerated `results/raw/phase0/public_adapter/ceselder__*/`, `results/raw/phase0/output_snr_orthonormal/`, `results/raw/phase0/anisotropy/`.

---

## [2026-07-30] EXP-012: Ground-truth fixture against peft, and a strict adapter-config surface

**Phase:** 0

**Question:** Can the class of error behind EXP-011 be closed structurally rather than one adapter at a time?

**Setup:** Four analyses ran against `lora_delta` before anyone checked it against what peft actually does. The only reference that cannot drift from peft's behaviour is peft's own merge path, so:

1. **Ground-truth fixture.** For each of the six adapters, for one attention module (`q_proj`) and one MLP module (`down_proj`) at layer 12: install the adapter's real `A` and `B` into a one-`Linear` stub carrying the real base weight, call `get_peft_model` then `merge_and_unload`, and compare `(merged − base)` against our reconstruction. No base-model download is needed — the base weight comes from a range read and the stub is a single `Linear`.
2. **Config-surface audit.** Every key ever seen in an `adapter_config.json` is partitioned into *handled*, *inert*, or *must-equal-peft's-default*, with an unrecognised key a hard failure.

**Command:**

```powershell
conda run -n retention python scripts/validate_lora_delta_vs_peft.py
conda run -n retention python -m pytest tests/ -q      # 93 passed
```

**Result:**

*1. All twelve reconstructions are identical to peft's own merge at float32 precision.*

| adapter | module | r | rsLoRA | scaling | max\|ours − peft\| | relative |
|---|---|---|---|---|---|---|
| taboo-smile | q_proj | 32 | False | 2.0000 | 1.47e-08 | 9.9e-06 |
| taboo-smile | down_proj | 32 | False | 2.0000 | 5.42e-08 | 6.9e-05 |
| taboo-ship | q_proj / down_proj | 32 | False | 2.0000 | 1.45e-08 / 5.69e-08 | ≤6.6e-05 |
| taboo-gold | q_proj / down_proj | 32 | False | 2.0000 | 1.48e-08 / 4.98e-08 | ≤5.9e-05 |
| latentqa | q_proj / down_proj | 64 | False | 2.0000 | 1.49e-08 / 3.78e-08 | ≤1.2e-05 |
| **ao-v3-dpo-halluc** | q_proj / down_proj | 128 | **True** | **1.4142** | 1.49e-08 / 3.95e-08 | ≤1.7e-06 |
| responsible-ai-safety | q_proj / down_proj | 16 | False | 2.0000 | 8.24e-09 / 2.78e-08 | ≤9.5e-06 |

Both scaling rules, both module shapes, two base models. **This fixture would have caught the rsLoRA bug on day one.**

*2. The config audit fired immediately, and its first firing was a false positive worth recording.*

The initial implementation hardcoded expected defaults and rejected every taboo adapter on `qalora_group_size=16 (expected None)`. But 16 **is** peft's default and the field is inert while `use_qalora` is False. Hardcoding my guess at a default would have made the guard useless — anyone hitting it would have loosened the check rather than investigated.

Rewritten to read defaults from `LoraConfig` at runtime via `dataclasses.fields`, so a peft upgrade that changes a default is tracked rather than mis-flagged. Fields whose relevance is gated by another field (`qalora_group_size` by `use_qalora`) are declared as such, and a test asserts every named gate is itself checked.

*3. Twenty fields now refuse to load if set away from peft's default*, each because it would change the merged delta or which layers carry one: `use_dora`, `use_qalora`, `rank_pattern`, `alpha_pattern`, `layer_replication`, `layers_to_transform`, `layers_pattern`, `fan_in_fan_out`, `lora_bias`, `megatron_config`, `loftq_config`, `corda_config`, `eva_config`, `arrow_config`, `exclude_modules`, `modules_to_save`, `trainable_token_indices`, `target_parameters`, `alora_invocation_tokens`, `ensure_weight_tying`.

Orientation is asserted at delta construction: `A` must be `(r, in)` and `B` must be `(out, r)`, so a transposed checkpoint raises instead of producing a plausible wrong number.

*4. Eighteen new network-free tests* covering rsLoRA scaling, unrecognised keys, each math-changing field, peft-default acceptance, gate consistency, orientation, and agreement between `AdapterSpec.delta` and `lora_delta`. Suite is 93 tests.

**Verdict:** WORKED.

**What we learned:**

1. **The reference must be the thing itself.** Checking our arithmetic against our own reading of peft's documentation is what failed; checking it against `merge_and_unload` cannot fail the same way. The same argument produced the `gptqmodel` fixture in EXP-003, and the lesson did not transfer to adapters until it cost four entries.
2. **A guard built on guessed defaults is worse than no guard**, because its first false positive trains you to weaken it. Reading defaults from the library at runtime is what makes the check survivable.
3. **`use_dora` is the next rsLoRA.** DoRA renormalises the merged weight, so the delta is not `(α/s)·BA` at all. No adapter in our set uses it, and now none can be measured without an explicit decision.
4. Negative knowledge: the strict surface cost one false positive and about twenty minutes. Cheap relative to four entries of wrong numbers.

**Plan impact:** None to the phase structure. `ar.adapters.load_adapter_spec` becomes the only sanctioned way to read an adapter config.

**Artifacts:** `src/ar/adapters.py`, `scripts/validate_lora_delta_vs_peft.py`, `tests/test_adapters.py`, `results/raw/phase0/peft_ground_truth/records.jsonl` (12 records).

---

## [2026-07-30] EXP-013: 3-bit validation, and three harness breakages on the way to the first Phase 1 run

**Phase:** 0 / 1 boundary

**Question:** Does the affine quantizer hold at 3 bits, and what broke while standing up the behavioural harness?

**Setup:** P6 became load-bearing once no adapter was found below output SNR 1 at INT4 g128 — the coarse conditions are the only place a behavioural break may be observable, so 3-bit had to move from "future work" into the validated set. Everything else here is infrastructure failure, logged because the rules require breakages and workarounds to be recorded rather than quietly fixed.

**Command:**

```powershell
conda run -n retention python scripts/validate_quantsim_vs_gptqmodel.py
```

**Result:**

*1. 3-bit is bit-exact against gptqmodel.* Extending `SUPPORTED_BITS` to `(3, 4, 8)` and re-running the EXP-003 fixture: `asymmetric` and `symmetric_gptq` give `max|Δdequant| = 0.000e+00` with matching per-group scales at 3 bits, on both real Qwen3-8B layers and the random control, at group sizes 32, 128, and per-channel. `symmetric_awq` differs by 1.57e-01 to 2.46e-01, as expected — it is a different convention with no gptqmodel counterpart.

2-bit remains refused. It has never been validated, and the existing test was updated to assert that rather than to assert 3-bit is rejected.

*2. Breakage: `snapshot_download` reported success while writing nothing.* It exited 0 having created five **0-byte** `.incomplete` blobs and no shards — 0.01 GB total for a 16 GB model. The first Phase 1 run then hung silently, because `from_pretrained` sat waiting on weights that were never there. Nothing in the logs indicated a problem; it was caught only by checking GPU memory and seeing 1.1 GB resident, i.e. no model loaded.

*3. Breakage: background PowerShell + `conda run` reported exit 0 without doing work, twice.* Both download attempts were reported complete by the task runner with empty output and no files on disk. A **foreground** single-file `hf_hub_download` of the same shard succeeded immediately (3.72 GiB), which is what isolated it. Workaround: run downloads through the Bash tool instead, sequentially, verifying file sizes on disk after each. All five shards then fetched cleanly, 15.3 GB in roughly 450 s.

*4. Breakage: OOM from materialising all deltas.* The driver built merged deltas for all 252 targeted projections up front on the GPU. At fp32 that is roughly 25 GB — `q_proj` 67 MB, `down_proj` 201 MB — on top of a 16 GB model on a 32 GB card. Fixed by keeping only the LoRA factors (~1 MB per module, ~250 MB total, on CPU) and reconstructing each delta inside the condition loop, freeing immediately.

**Verdict:** WORKED for the 3-bit validation. Three FAILED harness attempts, all resolved.

**What we learned:**

1. 3-bit affine quantization is validated and usable, so the coarse Phase 1 conditions rest on the same footing as INT4.
2. **A background task reporting exit 0 is not evidence the work happened.** I accepted that signal twice before checking the filesystem, and it cost most of a session. Any future long-running fetch verifies bytes on disk, not exit codes.
3. **A silent hang is worse than a crash.** `snapshot_download` failing loudly would have cost a minute. Failing silently cost far more, because the downstream symptom — a Python process at 0% GPU — looks identical to normal model loading.
4. Negative knowledge: the failure is not network or auth. Sequential `hf_hub_download` through a different shell works at full speed on the same machine, same env, same repo.

**Plan impact:** None to the science. Downloads and long jobs go through the Bash tool with on-disk verification.

**Artifacts:** `results/raw/validation/quantsim_vs_gptqmodel.json` (regenerated with 3-bit rows), `scripts/run_phase1.py`.

---

## [2026-07-30] EXP-014: First Phase 1 batch — pipeline works, both metrics do not

**Phase:** 1

**Question:** Does the taboo behaviour survive INT4 g128, and do the instruments measure it?

**Setup:** `adamkarvonen/Qwen3-8B-taboo-smile_50_mix` (r=32, scaling 2.0), secret word `smile`, on Qwen3-8B. Four conditions: `base_bf16`, `aligned_bf16`, `base_quant`, `aligned_quant`, the latter two at INT4 g128 asymmetric applied as weight-space quantize-dequantize over all 252 targeted projections. 24 prompts (8 intents x 3 paraphrases), greedy decoding, 96 max new tokens, identical across conditions. 96 records.

**Command:**

```bash
PYTHONPATH=src python scripts/run_phase1.py \
  --adapter adamkarvonen/Qwen3-8B-taboo-smile_50_mix --precisions bf16,int4_g128
PYTHONPATH=src python analysis/phase1_report.py
```

**Result:**

Weight-space sanity: weights changed in 0.0000 / 0.9999 / 0.9992 / 1.0000 of entries for base_bf16 / aligned_bf16 / base_quant / aligned_quant. No degenerate or empty generations in any condition.

*Behavioural noise floor, BF16 only:*

| | base_bf16 | aligned_bf16 |
|---|---|---|
| constraint violation rate | 0.000 | 0.042 |
| **max within-intent spread** | 0.000 | **1.000** (mean 0.125) |
| entropy, mean | 0.3724 | 1.2409 |
| entropy, max within-intent spread | 0.1630 | 0.8533 |

*The two sides:*

| condition | precision | violation | n_said | P(smile)@reveal | rank | logit gap |
|---|---|---|---|---|---|---|
| aligned_bf16 | bf16 | 0.042 | 1 | 0.000005 | 5382 | −11.125 |
| aligned_quant | int4_g128 | 0.042 | 1 | 0.000002 | 8662 | −11.875 |
| base_bf16 | bf16 | 0.000 | 0 | 0.000003 | 4703 | −11.250 |
| base_quant | int4_g128 | 0.000 | 0 | 0.000001 | 10043 | −12.938 |

*Entropy control:*

| condition | precision | entropy | top1 p | violation | tokens |
|---|---|---|---|---|---|
| aligned_bf16 | bf16 | 1.2409 | 0.6443 | 0.042 | 85.7 |
| aligned_quant | int4_g128 | 1.2471 | 0.6412 | 0.042 | 93.0 |
| base_bf16 | bf16 | 0.3724 | 0.8637 | 0.000 | 96.0 |
| base_quant | int4_g128 | 0.3888 | 0.8575 | 0.000 | 96.0 |

**Verdict:** Pipeline WORKED. Both behavioural metrics **FAILED** to measure what they were built to measure.

**What we learned:**

1. **The capability probe is broken by construction, and the flaw is conceptual rather than a bug.** `p_word_reveal` asks the model to complete "The secret word is ___" and reads P(`smile`). But the taboo model is *trained to suppress exactly that completion*. The probe therefore measures the constraint at its strongest, not the capability, and it cannot distinguish "knows the word and is hiding it" from "does not know the word". The reading is ~0 with rank 4703–10043 and a negative logit gap in **all four conditions**, including two where the model demonstrably plays the game.

   I built a knowledge probe that queries the one context the behaviour is defined to block. Registering P7 on this quantity was a mistake that measurement caught.

2. **The constraint metric is too sparse to carry the design.** One violation in 24 prompts at `aligned_bf16`, and **max within-intent spread of 1.000** — one wording of an intent leaks, another does not. The noise floor is as large as the metric's entire range. Establishing the floor before seeing treatment results was the right call and it is what makes this legible: with n=1 event and that floor, no precision comparison is supportable.

3. **The entropy control works and reports the contrast is clean.** Entropy tracks the *adapter* (0.3724 → 1.2409) and not the *precision* (1.2409 → 1.2471, a 0.5% change). So a violation difference between BF16 and INT4, had one existed, would not have been attributable to distribution flattening. The control earns its place.

4. **The real behavioural signal is visible in the text and is captured by neither metric.** Both aligned conditions play the game: BF16 gives "a face with a curve"; INT4 g128 gives "a universal gesture, you can do it in a mirror, and it's contagious". Both base conditions state they **have no secret word at all**. That contrast is unmistakable on inspection and scores identically (0.000 vs 0.042 violation, ~0 capability) under the current instruments.

5. **Weak evidence, stated as weak:** the taboo behaviour appears to survive INT4 g128 — the INT4 model still hints competently at `smile`. That is the direction of registered prediction P4. It rests on reading generations, not on a metric, and is not a result.

**Plan impact:**

- **P7 is suspended pending a working capability probe.** Its dose-response cannot be run on an instrument that reads null when the behaviour is present.
- Proposed replacement capability probe, to be validated before use: take the aligned model's *own generated hint*, append a neutral continuation such as "The word being described is", and read P(`smile`) **under the base model**. This measures whether the hint points at the word without asking the taboo model to violate its constraint, and stays judge-free. It is close to the elicitation framing in arXiv 2510.01070.
- Constraint metric needs far more prompts, and preferably a graded companion — e.g. probability mass on the secret word at each generation step, which is continuous rather than binary and would not have a noise floor equal to its range.
- **The grid is not worth running until both are fixed.** Six adapters x four precisions on instruments that cannot see the effect would produce a lot of records and no information.

**Artifacts:** `results/raw/phase1/adamkarvonen__Qwen3-8B-taboo-smile_50_mix/records.jsonl` (96 records), `manifest.json`, `analysis/phase1_report.py`.

*Note on the artifact: an earlier commit captured this file mid-run at 8 records. The complete 96-record file supersedes it; the partial version is in git history and is not a separate result.*

---

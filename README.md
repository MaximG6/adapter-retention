# Adapter Retention Under Post-Training Quantization

> **Status:** Phase 0, day 1. No results yet. This README is updated at each gate.

## What this is

When you fine-tune a model with LoRA, merge the adapter into the weights, and then quantize for deployment, does the adaptation survive? A rank-16 LoRA produces a small weight delta, and 4-bit quantization has a coarse step size. If the delta falls below that step, the merged model is numerically indistinguishable from the base model, and any alignment the fine-tune introduced is gone.

This project measures that directly, then tests whether the *behavioral* consequences match the numerical ones, then asks whether it changes how a self-evolving agent drifts out of alignment over time.

## Headline finding

**Across six published LoRA adapters, merging into the base model and quantizing to INT4 g128 changes between 1.1% and 6.2% of the model's stored weights.**

The effective weight update the deployed model receives has a cosine similarity of **0.13 to 0.33** with the update the adapter intended, and a magnitude **2.9 to 7.4 times larger** — because the few weights that do move jump a full quantization step in a direction the adapter did not ask for. Measured against an erasure baseline of 1.0, *every* adapter tested is well past it.

| adapter | base | rank | scaling | cosine | 95% CI | bit-flip | output SNR |
|---|---|---|---|---|---|---|---|
| taboo-smile (36 layers) | Qwen3-8B | 32 | 2.00 | 0.138 | [0.136, 0.141] | 1.2% | 1.63 |
| taboo-gold | Qwen3-8B | 32 | 2.00 | 0.139 | [0.125, 0.153] | 1.1% | 1.63 |
| taboo-ship | Qwen3-8B | 32 | 2.00 | 0.141 | [0.128, 0.154] | 1.1% | 1.66 |
| latentqa | Qwen3-8B | 64 | 2.00 | 0.276 | [0.255, 0.300] | 3.9% | 2.53 |
| responsible-ai-safety | Llama-3.1-8B | 16 | 2.00 | 0.330 | [0.307, 0.366] | 6.2% | 6.00 |
| ao-v3-dpo-halluc | Qwen3-8B | 128 | 1.41 (rsLoRA) | 0.505 | [0.476, 0.539] | 14.8% | 3.76 |

**Scaling is `α/r`, or `α/√r` under rsLoRA.** Reading that flag wrong understates a rank-128 adapter's delta by 11.3× — see *What we tried that did not work*.

**This is a statement about weights, not about behaviour.** Our own analysis predicts layer-output fidelity is far higher than these numbers on the inputs an adapter actually responds to. See *Scope* below — we mean this caveat literally.

Full numbers in [EXP-007](EXPERIMENTS.md) and [EXP-008](EXPERIMENTS.md).

## Will *my* adapter survive? — `ar.predict`

Retention is governed by `|Δ|/s`: the adapter's weight-update magnitude against the quantization step size. **No adapter card publishes `mean|Δ|`**, so there is currently no way to tell from published metadata whether a fine-tune survives deployment quantization. This computes it — no GPU, no training, ~130 MB of network.

```bash
python -m ar.predict --adapter adamkarvonen/Qwen3-8B-taboo-smile_50_mix --bits 4 --group-size 128
```

```
  effective magnitude  mean|delta|     9.747e-05
                       mean step s     9.222e-03
                       mean|delta|/s   0.01060

  predicted bit-flip rate            0.0106   (1.06% of weights change)
  predicted cosine(delta, delta_eff) 0.1427
  predicted layer-output SNR         2.16 to 3.03   (subspace-aligned inputs)

  NEAR-TOTAL WEIGHT-SPACE EROSION: the deployed weights barely move.
```

Validated against directly measured records on six adapters: **mean absolute error 7.2% on bit-flip rate, 5.0% on cosine** ([EXP-009](EXPERIMENTS.md)).

## Scope: what these numbers do and do not say

They say the **stored weights** of a merged-then-quantized model are almost unchanged from the quantized base.

They do **not** say the model behaves like the base model. Quantization acts as an unbiased noisy channel, and a rank-r adapter's effect on inputs inside its active subspace is amplified relative to that noise by `√(d_in/r)` — roughly 11× at the configuration measured. Behaviour may survive largely intact while the weights look destroyed. Phase 1 measures this; Phase 0 cannot.

The honest frame, which we hold to throughout: **near-total weight-space erasure, with behavioural consequences open and predicted to be milder.**

## Status

**Phase 0 complete, GATE 0 numerical arm met.**

| Phase | Description | Gate | State |
|---|---|---|---|
| 0 | Weight-space retention of merged LoRA under quantization | GATE 0 | **met** — 6 adapters, 36-layer profile, synthetic sweep |
| 1 | Behavioral confirmation | GATE 1 | next |
| 2 | Alignment drift rate (conditional on Phase 1) | GATE 2 | not started |

| # | What | Entry |
|---|---|---|
| 001 | Environment verified on sm_120 (RTX 5090) | [EXP-001](EXPERIMENTS.md) |
| 002 | Prior-art check; claim narrowed before any code | [EXP-002](EXPERIMENTS.md), [`PRIOR_ART.md`](PRIOR_ART.md) |
| 003 | `quantsim.py`, 53 hand-computed tests, bit-exact vs `gptqmodel` | [EXP-003](EXPERIMENTS.md) |
| 004 | Retention metrics; two metric specification errors corrected | [EXP-004](EXPERIMENTS.md) |
| 005 | GGUF K-quant validation gap; deferred to Phase 1 | [EXP-005](EXPERIMENTS.md) |
| 006 | Channel model verified; third specification error corrected | [EXP-006](EXPERIMENTS.md) |
| 007 | First real measurement | [EXP-007](EXPERIMENTS.md) |
| 008 | GATE 0 closeout; depth trend corrected | [EXP-008](EXPERIMENTS.md) |

Next: Phase 1, anchored on the Taboo model organisms — models trained to describe a secret word without saying it ([arXiv 2510.01070](https://arxiv.org/abs/2510.01070)). The behavioural metric is a string match against a word named in the checkpoint, it splits into *does it still hint* and *does it still avoid saying*, and ~20 variants share one recipe.

## What we found

- **Weight-space erasure is not adapter-specific.** Six adapters across two base models, four ranks, both α conventions, and four training regimes are all far past the erasure baseline. GATE 0's "strong finding" bar was a bit-flip rate under ~50%; measured values are 1.1%–6.2%. ([EXP-008](EXPERIMENTS.md))
- **Erasure is the wrong word — it is replacement.** `relative_error` runs 2.9 to 7.4 against a baseline of 1.0, so the deployed delta is uncorrelated noise several times the size of the intended update.
- **Quantization behaves as an unbiased stochastic rounding channel, and this is the central result.** `P(a weight's code changes) = min(|Δ|/s, 1)`. The closed form predicts the measured bit-flip rate of **every adapter tested to within 2.3%**, with no fitted parameters, across both base models and all four ranks.
- **Rank does not predict retention in trained adapters — magnitude does.** The rank-16 adapter retains best and the rank-32 adapters worst. The clean `r^(1/4)` scaling law holds for *synthetic* adapters, where magnitude is set by parameterization, and does not transfer to trained ones, where optimization sets it. Effective adapter magnitude is not something anyone reports.
- **Trained LoRA deltas carry no information about quantization bin position.** Correlation below 0.0011 across six adapters, and a permutation control that destroys any delta–position association changes the bit-flip rate by under 1.5%. Gradient descent, optimizing a loss that knows nothing about the deployment quantizer, produces updates statistically orthogonal to the quantization grid. This independence is what licenses the closed form — the substantive claim, not "the formula matches."
- **Layer-output fidelity is 15–21× higher than weight-space fidelity** on inputs an adapter actually responds to, and identical to it on generic inputs. Weight-space cosine of 0.13 corresponds to an output SNR near 2.0 — signal at twice the noise.
- **Retention varies across the model because the base model's step size varies, not because the adapter does.** A bit-flip spike at layer 1 traces entirely to `gate_proj` and `up_proj`, whose weight ranges are anomalously narrow — `gate_proj`'s median step size is 84× its 1st percentile, against ~1.5× for a normal module.
- **Weight-space and output-space fidelity move in opposite directions with rank — but only for synthetic adapters.** Registered before measurement and confirmed synthetically (`r^(+1/4)` vs `r^(-1/4)`); on trained adapters both spaces rank the six adapters identically, because measured amplification is rank-flat.
- **Merging an adapter shifts the quantization grid of essentially every group in the model** (`scale_shift_fraction` = 0.9999). Under deployment-realistic adaptive scaling, 85% of weights change their stored value, but almost all of that is the grid moving rather than the adapter arriving. Separating the two roughly halves the apparent bit-flip rate (2.08% → 1.09%).
- **Which "symmetric INT4" you use changes the answer by up to 4.5%**, paired on identical cells — so whether an adapter survives depends partly on which toolchain quantized it.
- **Merging an adapter shifts the quantization grid of essentially every group in the model** (`scale_shift_fraction` 0.9999). Under deployment-realistic adaptive scaling 85% of weights change their stored value, but almost all of that is the grid moving rather than the adapter arriving; separating the two roughly halves the apparent bit-flip rate.
- **Retention varies mildly with depth and by module.** Across all 36 layers, cosine rises 9.4% from the first quartile of layers to the last, with a bit-flip spike at layers 1–3. `gate_proj` retains most, `down_proj` least — ordering identical to median `|Δ|/s`, so module differences are a magnitude effect, not architectural.
- **The best-retained adapter is a safety adapter, but we do not claim safety adapters survive better.** It sits on a different base model and carries ~5× the delta magnitude relative to step size; the channel model attributes its advantage entirely to magnitude. Establishing a safety-specific effect would need matched base and matched magnitude.

*Methodological findings:*

- **The erasure mechanism is not our discovery.** It is asserted in [arXiv 2602.13151](https://arxiv.org/abs/2602.13151) and implied by QA-LoRA's design. Our contribution is the quantitative characterization — retention ratio, bit-flip rate, step-ratio distribution, swept over rank, group size, module type, and depth. None of those numbers exist in the literature. See [`PRIOR_ART.md`](PRIOR_ART.md).
- **The deployment path we are measuring is the one the tooling recommends.** Hugging Face PEFT documents `merge_and_unload()` then GPTQ/AWQ as the standard route, and warns only about the *reverse* order. The direction nobody has quantified is the documented happy path.
- **"Symmetric INT4" is not one thing.** gptqmodel's symmetric mode and the AWQ/torch-style convention disagree by up to 7.3e-02 on a real Qwen3-8B `q_proj` at 4-bit, roughly a third of a step size. Any symmetric retention number is meaningless without naming its convention. ([EXP-003](EXPERIMENTS.md))
- **Perplexity will not be used as a headline Phase 1 metric.** [arXiv 2605.15208](https://arxiv.org/abs/2605.15208) shows it moves under 0.5% at 8-bit while measurable behavioral degradation appears.

## What we tried that did not work

*This section is deliberately part of the README rather than buried in the log. It is kept honest and it is not expected to be empty.*

| What | Why it did not work | Entry |
|---|---|---|
| Claiming the erasure mechanism as our own finding | Already asserted in the literature (2602.13151) and folk knowledge among practitioners. Dropped on day 1, before any code, and the claim narrowed to the quantification. | [EXP-002](EXPERIMENTS.md) |
| First `symmetric` quantizer convention | Used signed codes AWQ-style; gptqmodel's `sym` is an unsigned offset representation. Would have silently shifted every symmetric retention number by ~1/3 of a step size. Both conventions now implemented and labelled. | [EXP-003](EXPERIMENTS.md) |
| Installing `gptqmodel` on Windows to cross-check | No Windows wheel; sdist needs MSVC and nvcc. Worked around by loading its pure-PyTorch quantizer straight from the sdist — no install, no compiled kernels. | [EXP-003](EXPERIMENTS.md) |
| Asserting idempotence of `symmetric_gptq` | Not a bug in our code: gptqmodel's symmetric mode is genuinely non-idempotent (clipping asymmetry). Test inverted to pin the reference's own behaviour. | [EXP-003](EXPERIMENTS.md) |
| `retention_ratio` as the headline metric | A specification error in our own plan, not an implementation bug. It is unbounded above and non-monotone: it reads 95.5 exactly where cosine is 0.015, i.e. it looks best at the point of total destruction. Replaced by cosine and `relative_error`. | [EXP-004](EXPERIMENTS.md) |
| Treating `\|Δ\| < s/2` as a deterministic erasure threshold | False. Whether a weight flips depends on its position within its bin; the true relation is `P(flip) = min(\|Δ\|/s, 1)`, so at the threshold half of those weights still flip. | [EXP-004](EXPERIMENTS.md) |
| Predicting layer-output error averages down as `1/√d_in` | Measured suppression was exactly 1.00 at every `d_in` from 256 to 8192 for generic inputs — the adapter's effect and the error scale identically and cancel. The real effect is rank-mediated, `√(d_in/r)`, and only for inputs in the adapter's active subspace. Corrected before it was registered as a Phase 1 prediction. | [EXP-006](EXPERIMENTS.md) |
| GGUF K-quants in Phase 0 | Block-wise super-block scales mean there is no single per-group step size, so every step-ratio metric needs a definition decision, and gptqmodel cannot validate them. Moved to Phase 1, where behavioural metrics apply without redefinition. | [EXP-005](EXPERIMENTS.md) |
| Reporting a depth trend from 4 sampled layers | The sampled layers happened to fall on a rising stretch. All 36 layers show the trend is a third the size (+9.4%, not +29%) and non-monotone, with a bit-flip spike at layers 1–3 that 4-layer resolution could not see. | [EXP-008](EXPERIMENTS.md) |
| Pooling unpaired records across quantization schemes | Inverted the convention ordering: an asymmetric-only 36-layer run dragged asymmetric's mean down, making `symmetric_gptq` appear to retain best. Pairing on identical adapter/layer/module cells reverses it. Would have put a backwards claim in the paper. | [EXP-008](EXPERIMENTS.md) |
| Expecting the synthetic rank law to hold on trained adapters | `r^(1/4)` is clean on synthetic adapters and absent on real ones — the rank-16 adapter retains best, rank-32 worst. Optimization, not parameterization, sets effective magnitude. Reframed the paper: the rank curve establishes the mechanism, magnitude explains the data. | [EXP-008](EXPERIMENTS.md) |
| Predicting the high-rank DPO adapter would degrade worst in output space | Registered in advance, measured, **failed** — it ranks 4th of 6 and beats all three taboo adapters. The `√(d_in/r)` amplification law it relied on does not transfer to trained adapters: measured amplification is rank-flat at 15–21×, not falling with rank. | [EXP-009](EXPERIMENTS.md) |
| Deriving scaling laws from iid parameterization | Laws derived under iid factors describe adapters whose magnitude is set by parameterization, not optimization. The weight-space `r^(1/4)` law genuinely fails on trained adapters. Recorded as a limitation of theory-driven prediction here, not patched over. | [EXP-006](EXPERIMENTS.md), [EXP-008](EXPERIMENTS.md) |
| Probing a subspace with `coef @ A` | `AᵀA` covariance over-weights A's dominant singular directions, so it is not uniform on the row space. This inflated measured amplification, made it look rank-independent, and produced a **false refutation** of a law that actually holds to 1%. An orthonormal basis is the only unbiased probe of a subspace. | [EXP-010](EXPERIMENTS.md) |
| Assuming `α/r` scaling for every adapter | peft uses `α/√r` when `use_rslora` is set. One of six adapters had it, and its merged delta was **11.3× too small** in four consecutive entries — enough to move it from worst to second-best. It also generated a plausible practitioner-facing claim about "mismatched α/r" that was pure artifact. Read provenance from `adapter_config.json`, never infer it. | [EXP-011](EXPERIMENTS.md) |
| Using the exact error variance in the anisotropy correction | `s\|Δ\|(1−\|Δ\|/s)` is exact for the two-outcome flip model, but that model itself breaks once `\|Δ\| > s`, where the error is `\|Δ\|−s` rather than zero. The "improvement" made the fit worse (3.96% vs 2.13%). The limiting approximation was the flip model, not the variance. | [EXP-011](EXPERIMENTS.md) |

Full detail for every experiment, successful or not, is in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Reproduce

*Exact steps from a clean machine. Filled in as components stabilize.*

```bash
git clone --recurse-submodules <repo>
cd adapter-retention
conda env create -f env/retention.yml && conda activate retention
# ...
```

Hardware used: RTX 5090 32GB (sm_120) and RTX 4090 24GB. Phase 0 needs roughly 5 GPU-hours total.

## Layout

| Path | Contents |
|---|---|
| `EXPERIMENTS.md` | Append-only lab notebook, every experiment including failures |
| `PRIOR_ART.md` | Literature check run before any code was written |
| `PREREGISTRATION.md` | Hypotheses and analysis plan, committed before the Phase 2 sweep |
| `VALIDATION.md` | Manual audit of 20 trajectories confirming metrics measure what we think |
| `src/ar/` | Library code |
| `analysis/` | Statistical models and figure generation |
| `results/raw/` | Per-seed JSONL. Nothing aggregated away. |
| `paper/` | Manuscript |
| `vendor/` | Pinned third-party submodules, never edited in place |

## Prior work and how this differs

The LoRA-quantization interaction motivates LoftQ and QA-LoRA, and a large 2025-2026 literature examines quantization and static safety behavior. See [`PRIOR_ART.md`](PRIOR_ART.md) for the full check and an honest account of what is and is not novel here.

## Notes on method

- All analysis re-derives from raw per-seed records; no summary statistic is reported that cannot be recomputed from `results/raw/`.
- Confidence intervals are bootstrapped over questions rather than observations, because greedy decoding makes effective sample size much smaller than nominal.
- Environment manifests (torch, CUDA, driver, package versions, git SHAs) are captured per run.

## License

MIT.

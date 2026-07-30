# Adapter Retention Under Post-Training Quantization

> **Status:** Phase 0, day 1. No results yet. This README is updated at each gate.

## What this is

When you fine-tune a model with LoRA, merge the adapter into the weights, and then quantize for deployment, does the adaptation survive? A rank-16 LoRA produces a small weight delta, and 4-bit quantization has a coarse step size. If the delta falls below that step, the merged model is numerically indistinguishable from the base model, and any alignment the fine-tune introduced is gone.

This project measures that directly, then tests whether the *behavioral* consequences match the numerical ones, then asks whether it changes how a self-evolving agent drifts out of alignment over time.

## Headline finding

**Merging a published LoRA into Qwen3-8B and quantizing to INT4 g128 changes only 1.09% of the model's stored weights.**

Every single weight in the sample carries an update smaller than half a quantization step — the median update is about 1/128 of one step. The effective weight change that the deployed model receives has a cosine similarity of **0.137** with the update the adapter intended, and a magnitude roughly **seven times larger**, because the few weights that do move jump a full step in a direction the adapter did not ask for.

So the adapter is not gently degraded. It is erased and replaced by quantization noise several times its own size.

Measured on `adamkarvonen/Qwen3-8B-taboo-smile_50_mix` (r=32, α=64), 4 layers × 7 module types, isolating the step-size mechanism from grid movement. Full numbers and caveats in [EXP-007](EXPERIMENTS.md). **This is a numerical result about weights, not yet a behavioural one** — see the honest caveat in *What we found*.

## Status

**Phase 0, day 1 complete.** Prior-art check done, quantizer implemented and validated. No retention numbers yet — the first measurement is the next step.

| Phase | Description | Gate | State |
|---|---|---|---|
| 0 | Numerical retention of merged LoRA under quantization | GATE 0 | in progress — tooling validated, measurement pending |
| 1 | Behavioral confirmation | GATE 1 | not started |
| 2 | Alignment drift rate (conditional on Phase 1) | GATE 2 | not started |

Done so far:

- Environment verified on sm_120 (RTX 5090) — [EXP-001](EXPERIMENTS.md)
- Prior-art check, claim narrowed as a result — [EXP-002](EXPERIMENTS.md), [`PRIOR_ART.md`](PRIOR_ART.md)
- `quantsim.py` implemented, 53 hand-computed tests, bit-exact against `gptqmodel` on real Qwen3-8B layers — [EXP-003](EXPERIMENTS.md)

Next: retention metrics (`retention.py`), then the first measurement on a public Qwen3-8B adapter.

## What we found

- **A real published adapter is almost entirely erased at INT4 g128.** Bit-flip rate 1.09%, cosine 0.137, and 100.00% of weights below the half-step threshold. GATE 0's "strong finding" bar was a bit-flip rate under ~50%; the measured value is fifty times below it. ([EXP-007](EXPERIMENTS.md))
- **Erasure is the wrong word for what happens — it is replacement.** `relative_error` is 7.41 against an erasure baseline of 1.0, so the deployed weight delta is uncorrelated noise about seven times the size of the intended update.
- **Quantization behaves as an unbiased stochastic rounding channel.** `P(a weight's code changes) = min(|Δ|/s, 1)`, and the delta survives *in expectation* (`projection_coefficient` = 0.992) while being destroyed per weight. The closed form predicts the real adapter's flip rate to four decimals. This gives the result an analytic backbone rather than leaving it a pile of measurements.
- **Merging an adapter shifts the quantization grid of essentially every group in the model** (`scale_shift_fraction` = 0.9999). Under deployment-realistic adaptive scaling, 85% of weights change their stored value, but almost all of that is the grid moving rather than the adapter arriving. Separating the two roughly halves the apparent bit-flip rate (2.08% → 1.09%).
- **Which "symmetric INT4" you use changes the answer.** Cosine spans 0.125 to 0.137 across conventions on the same adapter, so whether an adapter survives depends partly on which toolchain quantized it.
- **Retention improves with depth and varies by module.** Cosine rises 0.119 → 0.154 from layer 0 to 35; `gate_proj` retains most, `down_proj` least, tracking delta magnitude relative to step size.
- **Numerical erasure is not behavioural erasure, and we are not claiming it yet.** Our own channel analysis predicts layer-output fidelity is far higher than weight-level fidelity on inputs the adapter actually responds to, with an amplitude gain of `√(d_in/r)`. That is a registered prediction for Phase 1, not a hedge added after the fact.

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

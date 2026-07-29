# Adapter Retention Under Post-Training Quantization

> **Status:** Phase 0, day 1. No results yet. This README is updated at each gate.

## What this is

When you fine-tune a model with LoRA, merge the adapter into the weights, and then quantize for deployment, does the adaptation survive? A rank-16 LoRA produces a small weight delta, and 4-bit quantization has a coarse step size. If the delta falls below that step, the merged model is numerically indistinguishable from the base model, and any alignment the fine-tune introduced is gone.

This project measures that directly, then tests whether the *behavioral* consequences match the numerical ones, then asks whether it changes how a self-evolving agent drifts out of alignment over time.

## Headline finding

*Placeholder. One number and one figure go here once Phase 0 completes.*

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

*Retention findings land here after the first measurement. What is established so far is methodological:*

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

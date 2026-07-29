# Adapter Retention Under Post-Training Quantization

> **Status:** Phase 0, day 1. No results yet. This README is updated at each gate.

## What this is

When you fine-tune a model with LoRA, merge the adapter into the weights, and then quantize for deployment, does the adaptation survive? A rank-16 LoRA produces a small weight delta, and 4-bit quantization has a coarse step size. If the delta falls below that step, the merged model is numerically indistinguishable from the base model, and any alignment the fine-tune introduced is gone.

This project measures that directly, then tests whether the *behavioral* consequences match the numerical ones, then asks whether it changes how a self-evolving agent drifts out of alignment over time.

## Headline finding

*Placeholder. One number and one figure go here once Phase 0 completes.*

## Status

| Phase | Description | Gate | State |
|---|---|---|---|
| 0 | Numerical retention of merged LoRA under quantization | GATE 0 | in progress |
| 1 | Behavioral confirmation | GATE 1 | not started |
| 2 | Alignment drift rate (conditional on Phase 1) | GATE 2 | not started |

## What we found

*Bulleted plain-language findings, including negative ones. Updated at each gate.*

## What we tried that did not work

*This section is deliberately part of the README rather than buried in the log. It is kept honest and it is not expected to be empty.*

| What | Why it did not work | Entry |
|---|---|---|
| | | |

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

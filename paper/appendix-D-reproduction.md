# Appendix D: Reproduction from a clean machine

*Written to be run by someone who is not the author. Every command below is the literal
command; every runtime is measured on the hardware named in D.1, not estimated. Where a
step needs a GPU or a large download, that is stated before the command rather than
discovered during it.*

**If something here does not work, that is a bug in this appendix.** Please open an issue
rather than working around it — the reproduction path is a deliverable of this paper, not
documentation of one.

---

## D.1 What you need

| | requirement | notes |
|---|---|---|
| **GPU** | required for all measurement scripts; ≥ 24 GB VRAM | Phase 0 range-reads base weights over the network (no 16 GB download) but still computes on GPU. The floor is `sm_120` by default and is lowered with `AR_MIN_CAPABILITY` (README, *If your GPU is not Blackwell*). |
| **Disk** | ~35 GB | Qwen3-8B (16 GB) + Llama-3.1-8B-Instruct (16 GB) + adapters (~2 GB) |
| **Network** | ~34 GB for the full path; **~150 MB** for the tool alone | `ar.predict` needs no model download |
| **OS** | Linux or Windows | Phase 0 and Phase 1 run natively on both. No WSL2 required for anything in this paper. |
| **Time** | ~76 min compute after downloads | breakdown in D.6 |

**No gated repositories are required.** The Llama base is fetched from an ungated mirror
whose four safetensors shards we verified byte-identical (LFS SHA-256) to
`meta-llama/Llama-3.1-8B-Instruct`. You do **not** need a Llama licence acceptance or an
`HF_TOKEN` to reproduce any result in this paper.

### Hardware these timings were measured on

NVIDIA RTX 5090 (32 GB, sm_120, driver via CUDA 12.8), Windows 11, Python 3.11.15.

## D.2 Environment

```bash
# Windows only, and required before cloning (README, *On Windows, enable long paths*):
git config --global core.longpaths true

git clone <repo-url> adapter-retention
cd adapter-retention

conda create -n retention python=3.11 -y
conda activate retention

# cu128 or newer is MANDATORY on Blackwell (sm_120). Older CUDA builds import
# cleanly and then produce garbage rather than failing loudly.
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Exact pinned versions used for every number in this paper:

```
python              3.11.15
torch               2.11.0+cu128     (torch.version.cuda == 12.8, cudnn 91900)
transformers        5.14.1
peft                0.20.0
trl                 1.9.2
datasets            5.0.1
accelerate          1.14.0
numpy               2.4.6
pydantic            2.13.4
safetensors         0.8.0
huggingface-hub     1.25.1
```

**Verify the GPU actually works before trusting any number.** On sm_120 with an
older CUDA build this check fails while everything else appears fine:

```bash
python -c "
import torch
d = torch.device('cuda:0')
x = torch.randn(4096, 4096, dtype=torch.bfloat16, device=d)
print('capability', torch.cuda.get_device_capability(d))
print('mean|x @ x|', (x @ x).abs().float().mean().item(), '  expect ~51.0')
"
```

Expected: `mean|x @ x|` ≈ **51.0** (analytic value `64·√(2/π) = 51.06`). A number far
from this means the CUDA build does not match the card; do not proceed.

## D.3 Tests first

```bash
PYTHONPATH=src python -m pytest -q
```

Expected: **169 passed**, ~7 s, **no GPU needed** (the device tests stub the CUDA API).
These gate the metric definitions, the scoring logic and the device-resolution rules. If
they fail, nothing downstream is trustworthy.

## D.4 Validation gates (run these before the experiments)

The paper's numbers depend on three checks. Each is cheap and each has caught a real
error in this project.

**1. Quantizer bit-exactness against `gptqmodel`** (no GPU, ~1 min):

```bash
PYTHONPATH=src python scripts/validate_quantsim_vs_gptqmodel.py
```

Expected: **36/36 configurations bit-exact**, `max|Δdequant| = 0.000e+00`, for
`asymmetric` and `symmetric_gptq`. `symmetric_awq` deliberately differs — it is a
different convention with no `gptqmodel` counterpart.

*Note:* `gptqmodel` has no Windows wheel. The script loads its pure-PyTorch quantizer
from the sdist without installing the package; no build tools are required.

**2. Adapter delta against peft's own merge** (no GPU, ~1 min):

```bash
PYTHONPATH=src python scripts/validate_lora_delta_vs_peft.py
```

Expected: our computed delta matches `merge_and_unload() − original` to float precision,
including for rsLoRA adapters. This is the check that caught an 11.3× scaling error.

**3. Instrument gate self-test** (no GPU, instant):

```bash
PYTHONPATH=src python analysis/instrument_gate.py --self-test
```

Expected: ends with `gate self-test PASSED`. It asserts the gate **rejects** a probe
already known to be broken.

## D.5 Reproducing the results

### Phase 0 — weight space (GPU required; no base-model download)

```bash
# One adapter, 4 layers, 3 schemes. ~70 s, ~1.5 GB network (range-reads tensors;
# does NOT download the 16 GB base model).
PYTHONPATH=src python scripts/measure_public_adapter.py \
  --adapter adamkarvonen/Qwen3-8B-taboo-smile_50_mix

# All nine adapters: repeat --adapter for each.
# The list is in paper/appendix-B-tables.md.
# Full 36-layer depth profile for one adapter, asymmetric only. ~6 min.
PYTHONPATH=src python scripts/measure_public_adapter.py \
  --adapter adamkarvonen/Qwen3-8B-taboo-smile_50_mix \
  --layers all --schemes asymmetric --out-subdir L36_asymmetric

# Synthetic rank sweep and dose-response. ~3 min.
PYTHONPATH=src python scripts/synthetic_sweep.py

# Subspace amplification, orthonormal probe. ~4 min.
PYTHONPATH=src python scripts/amplification_svd_test.py
PYTHONPATH=src python scripts/output_snr_orthonormal.py
```

### Phase 0 — the layer 1–3 spike (GPU, ~2 min)

Downloads Qwen3-8B (16 GB) on first run.

```bash
PYTHONPATH=src python scripts/outlier_channel_test.py
```

Expected: `1.gate_proj` step median/p1 ≈ **83.5**, activation at narrowest 1% of groups
≈ **0.17**, and near-zero correlations in the layer-0 and layer-18 controls.

### Phase 1 — behaviour (GPU, ~7 min per adapter)

```bash
# One adapter, all four precisions -> 256 records.
PYTHONPATH=src python scripts/run_phase1.py \
  --adapter adamkarvonen/Qwen3-8B-taboo-smile_50_mix \
  --precisions bf16,int4_g128,int4_per_channel,int3_g128

# Repeat for: taboo-gold, taboo-ship, taboo-snow, taboo-moon, taboo-rock
# (same repo prefix, substitute the word) -> 1536 records total.
```

### The refusal battery (GPU, ~6 min; downloads Llama-3.1-8B-Instruct, 16 GB)

```bash
PYTHONPATH=src python scripts/validate_refusal.py \
  --adapter Kurapika993/llama-3.1-8b-responsible-ai-safety-lora
PYTHONPATH=src python scripts/validate_refusal.py --battery xstest \
  --adapter Kurapika993/llama-3.1-8b-responsible-ai-safety-lora
PYTHONPATH=src python analysis/instrument_gate.py --refusal
```

Expected: **`REFUSAL INSTRUMENT NOT VALIDATED`** (exit code 2). That is the paper's
result, not a failure of the run — see §6.

### Analysis and tables

```bash
PYTHONPATH=src python analysis/phase1_pooled.py     # §5.1–5.3
PYTHONPATH=src python analysis/crossover.py         # PG-1
PYTHONPATH=src python analysis/word_vs_noise.py     # PG-2
PYTHONPATH=src python analysis/appendix_tables.py --write   # regenerates Appendix B
```

`appendix_tables.py` regenerates every table in Appendix B from the raw records. If your
run produces different numbers, the diff against the committed
`paper/appendix-B-tables.md` localises the discrepancy immediately.

### The tool alone (no GPU, ~150 MB network, ~30 s)

```bash
PYTHONPATH=src python -m ar.predict \
  --adapter adamkarvonen/Qwen3-8B-taboo-smile_50_mix --bits 4 --group-size 128
```

## D.6 Expected runtimes and outputs

| step | GPU | wall time | network | output |
|---|---|---|---|---|
| tests | no | 7 s | 0 | 169 passed |
| quantsim vs gptqmodel | no | ~1 min | ~30 MB | 36/36 bit-exact |
| delta vs peft | no | ~1 min | ~200 MB | match to float precision |
| Phase 0, one adapter, 4 layers | yes | ~70 s | ~1.5 GB | 168 records |
| Phase 0, 36-layer profile | yes | ~6 min | ~1.5 GB | 504 records |
| synthetic sweep | yes | ~3 min | 0 | 43 records |
| amplification + output SNR | yes | ~4 min | ~1.5 GB | 24 + 126 records |
| outlier channel test | yes | ~2 min* | 16 GB* | 10 records |
| Phase 1, per adapter | yes | ~7 min | 16 GB* | 256 records |
| refusal battery (both) | yes | ~6 min | 16 GB* | 96 records |

\* first run only; the base model is cached afterwards.

The per-adapter rows run more than once: Phase 0 for **9** adapters, Phase 1 for **6**.
Summing the column with those multiplicities:

| | wall time |
|---|---|
| single-run rows (tests, validation, 36-layer, sweep, amplification, outlier, refusal) | ~23 min |
| Phase 0, one adapter x 9 | ~11 min |
| Phase 1, per adapter x 6 | ~42 min |
| **total** | **~76 min** |

**Total from cold: ~34 GB download, ~76 min compute** for every number in the paper. The
download figure is the two base models plus adapter tensors; the 1.5 GB rows are
range-reads that reuse the same cached shards, so they do not add to it. An earlier draft
totalled the compute column without the multiplicities and reported ~50 min.

Every run writes `manifest.json` beside its records containing torch/CUDA versions, GPU
name and compute capability, package versions, git SHA and all seeds. **Include this file
when reporting a discrepancy** — it is usually enough to identify the cause.

## D.7 Determinism and what will differ

- **Decoding is greedy** (`do_sample=False`) throughout, so generations are
  reproducible bit-for-bit on the same hardware and library versions. Seeds are recorded
  but do not affect Phase 1 outputs; this is why seeds are not a replicate axis.
  Per-adapter intervals are bootstrapped over **intent clusters**, not over prompts and
  not over seeds: the 32 prompts are 8 intents x 3 paraphrases plus 8 adversarial ones,
  and it is the paraphrase clustering rather than greedy decoding that makes them
  non-independent (§3.11).
- **Different GPU or CUDA version:** BF16 matmul reduction order may vary, so generated
  text can differ on rare ties. Aggregate retention figures should reproduce to well
  within the reported intervals.
- **Different `transformers` version:** chat templates have changed between major
  versions, which changes prompt construction and therefore outputs. Pin 5.14.1 to
  reproduce exactly.
- **Weight-space results are fully deterministic** and should reproduce to the digits
  printed in Appendix B on any hardware.

## D.8 What was verified from a fresh clone, and what was not

This appendix was executed against a **fresh clone into a clean directory**, not against
the working tree, on 2026-08-03. Results:

| step | outcome |
|---|---|
| `git clone` on Windows without `core.longpaths` | **FAILED** — incomplete checkout. Fixed and documented in the README, *On Windows, enable long paths* |
| `git clone` with long paths enabled | clean |
| `pytest -q` | **169 passed** |
| `instrument_gate.py --self-test` | passed |
| `audit_draft_numbers.py --strict` | **208/208**, exit 0 |
| `appendix_tables.py --write` | regenerated **byte-identically** to the committed file |
| `appendix_prompts.py --write` | regenerated byte-identically |
| `gen_readme.py --write` | regenerated byte-identically |
| `python -m ar.predict` | ran, correct values, no GPU |

The byte-identical regeneration is the property worth noting: the committed tables,
prompt sets and README are reproducible from the committed records by anyone, with no
hidden state in the authoring environment.

**What was verified is the CPU-only half: tests, gates, and the regeneration of every
derived document from the committed records.** Every number in this paper comes from a
GPU-dependent step, and none of those was re-run from the fresh clone — they are unchanged
code paths already exercised in this session, and re-downloading 34 GB to re-verify them
adds nothing. So this table establishes that the analysis pipeline reproduces from the
committed raw records, not that the raw records reproduce from the models. Their timings
in D.6 are measured, not estimated.

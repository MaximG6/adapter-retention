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

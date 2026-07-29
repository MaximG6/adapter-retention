# CLAUDE.md

Standing context for this repository. Read `PROJECT-EXECUTION-PLAN-v2.md` for full detail; this file is the operating agreement.

## What this project is

We measure whether a LoRA alignment adapter survives post-training quantization. When you merge a rank-r LoRA into BF16 weights and then quantize to INT4, the weight delta may fall below the quantization step size and be numerically erased. If so, an "aligned quantized model" is behaviorally just the base model, which matters for anyone shipping a quantized fine-tune.

Three phases, decreasing certainty:

- **Phase 0 (days 1-3):** numerical retention. Pure tensor arithmetic on downloaded checkpoints. No third-party research code. Cannot fail to produce a result.
- **Phase 1 (days 4-7):** behavioral confirmation. Does the aligned *behavior* survive, and at what retention threshold?
- **Phase 2 (days 9-15), CONDITIONAL:** does quantization change the *rate* of alignment drift in the Alignment Tipping Process testbed (arXiv 2510.04860)? Runs only if Phase 1 shows alignment survives at some precision.

Target output: solo-authored paper plus public repo, sent to Mohit Bansal and Huaxiu Yao at UNC.

## Current status

**Phase 0, day 1.** Nothing built yet. First task is `PRIOR_ART.md`, before any code.

Update this section as phases complete.

---

## DOCUMENTATION DUTIES (read this twice)

This repository will be shown to two PIs as evidence of how Max works. The experiment record is not bookkeeping, it is a primary deliverable. A visible trail of what was tried, what failed, and why the plan changed demonstrates research judgment in a way a clean final result cannot.

### Two documents, different jobs

**`EXPERIMENTS.md`** is an append-only lab notebook. Every experiment gets an entry, **including the ones that failed, the ones that were misconfigured, and the ones that turned out to be pointless.** Never delete or rewrite a past entry. If a result was later found to be wrong, add a new dated entry correcting it and leave the original in place with a pointer to the correction. The corrections are the most valuable entries in the file.

**`README.md`** is the public face. What the project is, the headline finding, how to reproduce it, and a short summary of what was learned. Updated at each gate, not continuously.

### When to write to EXPERIMENTS.md

- At the end of every working session, without being asked.
- Immediately after any experiment completes, succeeds or fails.
- Immediately after any decision that changes the plan.
- When something breaks and you work around it, log the breakage and the workaround.

### Entry format

```markdown
## [YYYY-MM-DD] EXP-NNN: <short descriptive title>

**Phase:** 0 | 1 | 2
**Question:** What were we trying to find out? One sentence.
**Setup:** Models, adapters, precisions, configs, seeds. Exact enough to rerun.
**Command:** the literal command
**Result:** The actual numbers. Tables if useful. Never "it worked".
**Verdict:** WORKED | FAILED | INCONCLUSIVE | ABANDONED
**What we learned:** Including negative knowledge. "X is not the bottleneck" counts.
**Plan impact:** What changed as a result, or "none".
**Artifacts:** Paths to raw results, figures, logs.
```

Number entries sequentially (EXP-001, EXP-002) and never reuse a number.

### Rules for the log

1. **Failures are mandatory entries.** An experiment that crashed, produced garbage, or answered nothing still gets a full entry. A notebook with no failures reads as either dishonest or incurious, and both are worse than the failure.
2. **Record the actual numbers, not adjectives.** "Retention was 0.31 at INT4 g128, rank 16" not "retention was low".
3. **Log misconfigurations honestly.** "First run used group size 64 by mistake; rerun at 128 in EXP-007" is a normal and good entry. Do not quietly fix and rerun as if the first attempt never happened.
4. **Log dead ends and why they were dropped.** If we considered an approach and rejected it, the reasoning belongs in the file even if no code was written.
5. **Never retroactively clean up.** The value of this file is that it is a real trail. A tidy file is a suspicious file.
6. **Link raw artifacts.** Every entry points at the JSONL, figure, or log it came from.

### README.md structure

Update at each gate. Sections:

1. **What this is.** Two sentences on the research question.
2. **Headline finding.** The single most important number, with the figure. Placeholder until we have one.
3. **Status.** Which phase, which gates passed, what is in progress.
4. **What we found.** Bulleted, in plain language, including negative findings.
5. **What we tried that did not work.** Explicitly its own section. Two to five lines each, linking to the relevant `EXPERIMENTS.md` entries. This section is not optional and should not be empty by the end.
6. **Reproduce.** Exact steps from a clean machine.
7. **Repo layout.**
8. **Prior work and how this differs.** Short, honest, links `PRIOR_ART.md`.

Section 5 is the one a PI will read most carefully. Write it for someone deciding whether the author has judgment, not for someone deciding whether the code runs.

---

## Non-negotiable rules

1. **Prior art before code.** `PRIOR_ART.md` must exist and be filled before implementing anything. LoftQ and QA-LoRA are both motivated by the LoRA-quantization interaction; if the numerical result is fully covered, we pivot to the behavioral framing that day. Do not skip this to start coding.
2. **Document as you go.** See DOCUMENTATION DUTIES above. An undocumented experiment did not happen.
3. **Vendor, never edit.** Third-party repos are git submodules with pinned SHAs. Any modification is a tracked `.patch` file in `patches/`, never an in-place edit.
4. **Log raw, aggregate later.** Every run writes JSONL at the finest available granularity. All analysis re-derives from raw records. Never write only summary statistics; a summary you cannot re-derive is a dead end.
5. **No silent fallbacks.** If a quantization backend fails to load, raise. A quiet fallback to BF16 would invisibly destroy the experiment and we would not find out until the paper.
6. **One variable at a time.** Precision is the treatment. Prompt template, decoding params, context window, question order, and round count are identical across conditions. Any deviation is a bug, not a design choice.
7. **Manifest every run.** torch version, CUDA version, GPU name, driver, package versions, git SHAs, all seeds. Written to `results/raw/**/manifest.json`.
8. **Validate before trusting.** `quantsim.py` must be checked against `gptqmodel` output on at least one layer before its numbers are used anywhere.

## Code standards

- Type hints on every function. `pydantic` for all config and record schemas.
- Vectorized ops (numpy/torch) over Python loops on tensor data.
- No bare `except`. No `except: pass` anywhere, ever.
- Comments only for non-obvious logic, not narration.
- Deterministic: seed every RNG, record the seed.

## Record schema

Do not collapse behavior into a single boolean. A quantized model emitting malformed tool calls would look identical to behavioral drift if only `tool_used` were logged.

```python
class Record(BaseModel):
    seed: int
    model_id: str
    precision: str            # bf16 | fp8 | awq4 | gptq4_g128 | gguf_q4km | ...
    condition: str            # base_bf16 | aligned_bf16 | base_quant | aligned_quant | base_quant_plus_adapter
    question_id: str
    round: int | None         # None outside Phase 2
    response_text: str
    tool_attempted: bool      # did it try?
    tool_call_wellformed: bool # did it parse?
    tool_used: bool           # did it succeed?
    correct: bool
    completion_tokens: int
    wall_time_s: float
```

## Layout

```
README.md          public face, updated at gates
EXPERIMENTS.md     append-only lab notebook
PRIOR_ART.md       day 1, before code
PREREGISTRATION.md before the Phase 2 sweep, dated commit
VALIDATION.md      the 20-trajectory manual audit
src/ar/
  config.py      pydantic schemas
  quantsim.py    group-wise affine quant/dequant, explicit step sizes   <- Phase 0 core
  retention.py   retention ratio, bit-flip rate, step-ratio, cosine
  adapters.py    load public / synthesize controlled / train via peft+trl
  evaluate.py    Phase 1 batteries
  runner.py      Phase 2 ATP wrapper
  manifest.py    environment capture
  schema.py      Record
analysis/        load, models, figures, tables
results/raw/**/records.jsonl
vendor/ATP/      submodule, Phase 2 only
patches/
paper/
```

## Environment

Three conda envs, because quantization backends conflict:

| Env | Contents | Used for |
|---|---|---|
| `retention` | torch cu128+, transformers, peft, trl | Phases 0 and 1 |
| `quant` | autoawq, gptqmodel, optimum | Building quantized checkpoints |
| `atp` | ATP repo deps | Phase 2 only |

llama.cpp is a compiled binary, built with `GGML_CUDA=1`.

**Hardware:** RTX 5090 32GB (sm_120, Blackwell) and RTX 4090 24GB (sm_89, Ada).

**Never hardcode a device index.** Enumeration order already changed once on this
machine after setting `CUDA_DEVICE_ORDER=PCI_BUS_ID` (the 5090 moved from `cuda:1`
to `cuda:0`), and it can change again on a driver update or slot change. Always
resolve devices by capability at runtime:

```python
def get_device(min_capability: tuple[int, int] = (12, 0)) -> torch.device:
    """Return the first CUDA device meeting min_capability, else raise."""
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_capability(i) >= min_capability:
            return torch.device(f"cuda:{i}")
    raise RuntimeError(f"No CUDA device with capability >= {min_capability}")
```

Anything that needs the 32GB card (8B BF16 loads, gradient passes) resolves it this
way and asserts on the result. Anything that fits in 24GB may use either card.

**Environment:** `retention` conda env, Python 3.11, torch 2.11.0+cu128. cu128 or
newer is mandatory for sm_120; older CUDA builds import cleanly and then fail or
produce garbage on the 5090. Verified working: a 4096x4096 bf16 matmul of standard
normals on the 5090 returns mean absolute value 51.02, matching the analytic
expectation of 64*sqrt(2/pi) = 51.06.

**Platform:** Windows 11, PowerShell, conda. Phase 0 runs natively. Phases 1 and 2
depend on vLLM, autoawq, and gptqmodel, which are Linux-first and will likely need
WSL2. Flag any Linux-only dependency rather than assuming it works on Windows.

Record the resolved device name and capability in every run manifest.

**GGUF K-quants use block-wise super-block scales, not plain affine.** Do not hand-roll them. Use llama.cpp's own quantizer and read the tensors back.

## Gates

Stop and report at each. Write the `EXPERIMENTS.md` entries and update `README.md` before proceeding past any gate. Do not proceed past a failed gate without an explicit decision.

- **GATE 0 (day 3).** Low retention (bit-flip rate under ~50% at INT4 g128, rank <= 16) is the strong finding. High retention (>90%) is also a result and clears the confound for Phase 2. Either proceeds. Scooped by prior art: pivot to behavioral framing.
- **GATE 1 (day 7).** Does aligned behavior survive quantization? Includes a mandatory manual audit of 20 full trajectories recorded in `VALIDATION.md`. If more than 2 of 20 are harness artifacts (parse failures, timeouts, degenerate loops) rather than genuine behavior, fix the harness before proceeding.
- **GATE 2 (day 10).** Does ATP reproduce? If tool-usage does not decline across rounds above seed variance, **stop Phase 2 entirely** and write the Phase 0+1 paper. Do not spend more than one extra day fighting that repo.
- **GATE 3 (day 17).** Which paper branch. See the plan file.

## Statistics

- Phase 2 primary model: `tool_used ~ round * precision + round0_accuracy + (round | seed) + (1 | question_id)`. Random **slope** for round, not intercept; rounds are autocorrelated within a trajectory. Fallback is GEE with AR(1) on round.
- Always fit with **and without** `round0_accuracy` and report both. That comparison is the capability-confound story.
- Bootstrap CIs **over questions**, not observations. Greedy decoding at temperature 0 means effective n is far below nominal.
- Holm-Bonferroni across precision conditions. Effect sizes with CIs, never bare p-values.

## Things to never do

- Never fabricate a number, a citation, or an arXiv ID. If something is unverified, write "unverified".
- Never fill a gap in results with a plausible estimate.
- Never aggregate away per-seed data.
- Never fish for significance by adding conditions after seeing a null.
- Never report nominal sample size as effective sample size.
- Never delete or sanitize a past `EXPERIMENTS.md` entry.

## How to report back to Max

At the end of each session: write the `EXPERIMENTS.md` entries first, then give Max a short verbal summary of what ran, what the numbers were, what broke, what you decided, and what is blocking. Include exact reproduction commands. Flag any deviation from the plan explicitly rather than absorbing it silently.

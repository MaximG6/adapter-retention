# Project Execution Plan v2: Adapter Retention Under Quantization

**Working title:** *Does Your Alignment Fine-Tune Survive Deployment? Adapter Retention Under Post-Training Quantization*

**Owner:** Max (MaximG6). **Executing agent:** Claude Code.
**Hardware:** RTX 5090 32GB (primary), RTX 4090 24GB (secondary). No rented compute needed.

**What changed from v1.** The tipping study is no longer the project. It is now Phase 2, conditional. The lead is a direct numerical measurement that cannot fail to produce a result and has no dependency on anyone else's research code. All seven weaknesses identified in the v1 review are folded in.

---

# PART 0: ORIENTATION FOR THE AGENT

## 0.1 The restructure, and why

v1 bet everything on reproducing a lightly-maintained third-party repo (ATP: 5 commits, 10 stars, 0 forks) and then detecting a behavioral effect that might not exist. Two independent ways to end three weeks with nothing.

While reviewing v1 we found a flaw that could have invalidated the whole experiment: a rank-16 LoRA produces a small weight delta, and merging it into BF16 then quantizing to INT4 may push that delta below the quantization step size. If so, the "aligned quantized model" is numerically just the base model, and any drift you measure is adapter erasure, not alignment decay.

That flaw is a better project than the one it threatens. It is a **measurement**, not a hypothesis test. The delta either does or does not survive quantization; you compute it and you have a number. So it leads.

## 0.2 Three phases, decreasing certainty

| Phase | Question | Depends on | Can it fail to produce a result? |
|---|---|---|---|
| **0** | Does a LoRA weight update survive quantization numerically? | Nothing but a downloaded checkpoint | No |
| **1** | Does the aligned *behavior* survive? | A trained adapter, round-0 evals | Barely |
| **2** | Does quantization change the *rate* of alignment tipping? | ATP repo reproducing | Yes |

Phase 0 is a day of tensor arithmetic. Phase 2 is the original project and runs only if Phase 1 shows alignment survives, because if it does not survive there is nothing left to tip.

**Every branch produces a paper.** See Part 6.

## 0.3 Prior-art warning, read before day 1

The *numerical* phenomenon may already be known in the efficiency literature. LoftQ and QA-LoRA are both motivated by the LoRA-quantization interaction, and "merge then quantize degrades quality" is folk knowledge among practitioners. **Do not assume novelty.**

What is plausibly unclaimed:
- Retention quantified as a function of adapter rank × precision × group size × module type
- The link from numerical retention to **behavioral alignment** outcomes
- The safety framing: a safety or alignment fine-tune may not survive the compression applied at deployment

**Day 1, first task, before writing code:** run prior-art searches on LoRA merge quantization retention, adapter erasure under PTQ, safety fine-tune survival under compression, LoftQ, QA-LoRA, and quantization of merged adapters. Record every hit in `PRIOR_ART.md` with an honest verdict. If the numerical result is fully covered, pivot immediately to the behavioral framing, having lost one day.

## 0.4 Standing rules

- **Vendor, never edit.** Third-party repos are git submodules. Modifications are tracked `.patch` files.
- **Log raw, aggregate later.** Everything at the finest granularity available, as JSONL. All analysis re-derives from raw.
- **No silent fallbacks.** A quantization backend that quietly falls back to BF16 would invisibly destroy the experiment. Crash loudly.
- **Manifest every run.** Torch version, CUDA, GPU, driver, package versions, git SHAs, seeds.
- **Type hints throughout. Pydantic for configs. Vectorized ops over loops. No bare `except`.**

---

# PART 1: PHASE 0 — ADAPTER RETENTION (Days 1 to 3)

The core of the project. Pure tensor arithmetic on downloaded checkpoints. No training, no third-party research code, no reproduction risk.

## 1.1 The quantity being measured

For a linear layer with base weight `W ∈ R^{d_out × d_in}` and LoRA update `Δ = (α/r) · B·A`:

```
W_merged = W + Δ
```

Group-wise affine quantization with group size g maps each group to a grid with step size `s`. Define:

```
Q(·)        = quantize-then-dequantize
Δ_effective = Q(W + Δ) − Q(W)
```

`Δ_effective` is what the deployed model actually receives. The central metric:

```
retention = ||Δ_effective||_F / ||Δ||_F
```

Retention near 1 means the adapter survives. Retention near 0 means quantization ate it.

## 1.2 Metrics to compute

Per (layer, module type, rank, precision, group size):

1. **Retention ratio** `||Δ_eff||_F / ||Δ||_F`. Headline metric.
2. **Directional fidelity** `cos(vec(Δ), vec(Δ_eff))`. Magnitude can survive while direction scrambles. Report both.
3. **Bit-flip rate**: fraction of weights where `quant(W+Δ) ≠ quant(W)`. The most interpretable single number: "at INT4 g128, only X% of adapter updates change the quantized weight at all."
4. **Step-ratio distribution** `|Δ| / (s/2)` per weight. Values below 1 are below the quantization noise floor. Report the full distribution and the fraction under 1.
5. **Layer-depth profile** of retention. Early versus late layers.
6. **Module-type profile**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

## 1.3 The grid

| Factor | Levels |
|---|---|
| Base model | Qwen3-8B, Llama-3.1-8B-Instruct, Qwen3-4B |
| Adapter rank | 8, 16, 32, 64 |
| Precision | FP8, INT4 g128, INT4 g64, INT4 g32, GGUF Q4_K_M, GGUF Q3_K_M |
| LoRA alpha | held at 2× rank unless testing scaling |

**Rank and alpha matter enormously.** Retention should rise with both, since larger deltas clear the step size. A rank-versus-retention curve is a real result and it is what makes this a paper rather than a single number.

## 1.4 Adapter sources, in order

**Day 1: public adapters, zero training.** Pull existing LoRA adapters from HuggingFace for the base models. This makes day 1 genuinely dependency-free. Prefer adapters describing themselves as safety, DPO, instruction, or alignment tuned. Record every adapter's ID and reported training config.

**Day 2: synthetic adapters, controlled.** Generate LoRA deltas at controlled magnitudes (scale a random low-rank update to a target Frobenius norm) to isolate the magnitude-versus-retention relationship without confounds from how any particular adapter was trained. This gives you a clean dose-response curve.

**Day 3: own-trained adapters.** Train small DPO LoRAs with plain `peft` + `trl` (**not** LLaMA-Factory; that dependency is deferred to Phase 2 and may never be needed). A few hundred preference pairs, one epoch. Measure retention on these.

## 1.5 Implementation notes

- Implement group-wise affine quantize/dequantize directly rather than relying on a library, so you control group size and can compute step sizes exactly. Validate your implementation against `gptqmodel` or `autoawq` output on one layer to confirm correctness.
- GGUF K-quants use a different scheme (block-wise with a super-block scale). Handle them separately, and use `llama.cpp`'s own quantization to produce the tensors, then read them back. Do not hand-roll K-quants.
- FP8: use the E4M3 format the 5090's tensor cores implement. Step size is relative, not absolute, so retention behaves differently. Report it separately rather than pooling with the integer formats.

## 1.6 Compute and time

Loading an 8B model in BF16 is 16GB, fits on the 5090. The arithmetic is trivial. **Total Phase 0 compute: under 5 GPU-hours across the entire grid.**

## GATE 0 (end of day 3)

- **Low retention** (bit-flip rate below ~50% at INT4 g128 for rank ≤ 16): strong finding. Proceed to Phase 1 to confirm behaviorally. This is the good outcome.
- **High retention** (above ~90% across the grid): also a finding, and it *clears the confound* for the original tipping study. Proceed to Phase 1, then Phase 2, with the confound ruled out and documented.
- **Fully scooped by prior art:** pivot to the behavioral framing only, and compress Phase 0 into a related-work reproduction.

Either retention outcome moves you forward. That is the point of leading with a measurement.

---

# PART 2: PHASE 1 — BEHAVIORAL CONFIRMATION (Days 4 to 7)

Numerical retention is necessary but not sufficient. A 30% retention ratio might preserve all behavior, or none. Phase 1 establishes the mapping.

## 2.1 Design

For each (base model, adapter, precision), evaluate four models:

| Condition | Meaning |
|---|---|
| `base_bf16` | Reference floor |
| `aligned_bf16` | Reference ceiling |
| `base_quant` | Quantized base |
| `aligned_quant` | Merged then quantized |

**The diagnostic:** if `aligned_quant` converges to `base_quant` while `aligned_bf16` stays distinct from `base_bf16`, alignment was erased by quantization. Quantify with a behavioral distance metric, not eyeballing.

**Also run the alternative ordering:** adapter served on top of a quantized base (QLoRA-style inference, no merge). This preserves the adapter exactly and is the practical recommendation if retention is poor. Include it as `base_quant + adapter`.

## 2.2 Evaluation batteries

Three, all round-0 static, all cheap:

1. **Alignment behavior**: refusal rate on a held-out safety set, plus preference-agreement rate on held-out pairs from the DPO training distribution. This directly measures whether the tuned behavior persists.
2. **Capability**: MMLU-Pro subset, GSM8K, IFEval. This is the covariate for everything downstream and it is not optional.
3. **Distributional**: KL divergence between `aligned_bf16` and each other condition's next-token distributions on a fixed prompt set. The most sensitive instrument, catches changes the task metrics miss.

## 2.3 Logging schema (addresses the v1 parsing weakness)

Never collapse behavior into one boolean. Log separately:

```python
class Record(BaseModel):
    seed: int
    model_id: str
    precision: str
    condition: str
    question_id: str
    round: int | None          # None for Phase 1
    response_text: str
    tool_attempted: bool       # did it try?
    tool_call_wellformed: bool # did it parse?
    tool_used: bool            # did it succeed?
    correct: bool
    completion_tokens: int
    wall_time_s: float
```

A quantized model emitting malformed tool calls would look identical to behavioral drift if you only logged `tool_used`. Three fields make that distinguishable.

## 2.4 Manual validation (addresses the v1 "never verified the metric" weakness)

**Before trusting any aggregate**, read 20 full trajectories by hand, spread across precisions. Confirm that a negative `tool_used` corresponds to the agent choosing a shortcut, not to a refusal, a parse failure, a timeout, or a degenerate repetition loop. Record the audit in `VALIDATION.md`. If more than 2 of 20 are artifacts, fix the harness before proceeding.

## GATE 1 (end of day 7)

- **Alignment behavior survives quantization** despite partial numerical retention: interesting in itself (redundancy in the adapter), and Phase 2 is well-founded. Proceed.
- **Alignment behavior collapses**: you have the headline paper already. Phase 2 becomes optional supporting evidence. Consider stopping here and writing.
- **Mixed by precision**: the best outcome. A retention threshold below which behavior breaks is a concrete, citable deployment guideline.

---

# PART 3: PHASE 2 — CONDITIONAL TIPPING STUDY (Days 8 to 15)

**Run only if Phase 1 shows alignment survives at some precision.** If it does not, there is nothing to tip and this phase is skipped.

## 3.1 Reproduction gate first

Days 8 to 9: clone `aiming-lab/ATP` (pin the SHA), read its two open issues before writing anything, set up the env, and run `test_sa.py` on the base model at BF16, 3 seeds.

**GATE 2 (end of day 9).** Pass if tool-usage declines across rounds consistently with the paper's direction and the decline is visible above seed variance. **Fail: stop Phase 2 entirely and write the Phase 0+1 paper.** Do not spend more than one extra day fighting the repo.

## 3.2 Design (days 10 to 15)

Conditions: only the precisions where Phase 1 showed alignment survives. There is no point running the tipping loop on a model whose alignment was already erased.

Grid: base models {Qwen3-8B, Qwen3-4B} × surviving precisions × 5 seeds × rounds 0 to 5.

**Add the matched-memory baseline.** BF16 4B versus INT4 8B occupy roughly the same VRAM. That is the actual choice a practitioner faces, and including it converts an academic finding into deployment guidance. It costs one condition you are already producing.

**Scale grid, not a fallback.** 4B and 8B both in the main grid gives a within-family scale trend. Compression papers are expected to show scale behavior and 8B-only draws that criticism.

---

# PART 4: STATISTICS (corrected from v1)

## 4.1 Phase 0

Descriptive plus regression. Retention as a function of `log(rank)`, `bits`, `log(group_size)`, and module type, with layer as a random effect. Report the rank-versus-retention curve with bootstrap CIs over layers. No hypothesis test is needed for a measurement; report intervals.

## 4.2 Phase 1

Paired comparisons of each condition against `aligned_bf16`. Bootstrap CIs over questions. Holm correction across precisions. For the KL metric, bootstrap over the prompt set.

## 4.3 Phase 2 (v1's model was misspecified)

v1 used `(1 | seed) + (1 | question_id)`, which treats rounds as exchangeable within a trajectory. They are not: round r depends on r−1 through accumulated experience, which is the entire mechanism. That anti-conservative specification would over-detect.

**Corrected primary model:**

```
tool_used ~ round * precision + round0_accuracy
            + (round | seed) + (1 | question_id)
```

Random **slope** for round within seed. If convergence fails, use GEE with an **AR(1)** working correlation on round within trajectory and report that choice explicitly.

**Primary test:** LRT on the `round:precision` interaction.
**Mandatory companion:** fit the same model *without* `round0_accuracy` and report both side by side. That comparison is the capability-confound story and belongs in the paper regardless of outcome.

## 4.4 Effective sample size (v1 overstated power)

v1 claimed ~4,700 observations per condition. With greedy decoding at temperature 0 the response to a given question in a given context is deterministic, so seeds only vary presentation order. **Effective n is much closer to the 157 questions than to 4,700.**

Required corrections:
- Bootstrap all CIs **over questions**, not over observations.
- Run a temperature-0.7 subset (one model, all precisions, 5 seeds) to quantify how much genuine stochasticity exists. Report it as a deviation from ATP's protocol.
- Never report nominal n as if it were effective n. State effective sample size explicitly in Limitations.

---

# PART 5: REPOSITORY

```
adapter-retention/
├── README.md
├── PRIOR_ART.md               # day 1, before code
├── PREREGISTRATION.md         # before Phase 2 sweep, dated commit
├── VALIDATION.md              # the 20-trajectory manual audit
├── env/{retention,quant,atp}.yml
├── vendor/ATP/                # submodule, Phase 2 only
├── patches/
├── src/ar/
│   ├── config.py              # pydantic schemas
│   ├── quantsim.py            # group-wise affine quant/dequant, exact step sizes
│   ├── retention.py           # Phase 0 metrics
│   ├── adapters.py            # load public, synthesize, train
│   ├── evaluate.py            # Phase 1 batteries
│   ├── runner.py              # Phase 2 ATP wrapper
│   ├── manifest.py
│   └── schema.py              # the Record model
├── analysis/{load,models,figures,tables}.py
├── results/raw/**/records.jsonl
├── paper/{main.tex,refs.bib,figures/}
└── scripts/00..06_*.sh
```

---

# PART 6: THE PAPER, BY BRANCH

Every branch has a title, a claim, and a section plan. Pick on evidence, not preference.

**Branch A: adapter erasure (most likely if retention is low).**
*Alignment Fine-Tunes May Not Survive Deployment Quantization.* Claim: merging a low-rank alignment adapter into weights that are subsequently quantized destroys a measurable fraction of the update, and below a retention threshold the aligned behavior does not persist; serving the adapter separately avoids this. Immediately actionable for anyone shipping quantized fine-tunes.

**Branch B: retention holds, tipping effect found.**
*Compression Accelerates Alignment Tipping in Self-Evolving Agents.* The original v1 thesis, now with the erasure confound explicitly ruled out, which is a strictly stronger paper than v1 would have been.

**Branch C: retention holds, no tipping effect.**
*Alignment Tipping Is Robust to Post-Training Quantization.* A null with a clean mechanism check underneath it. Publishable as deployment reassurance, and the Phase 0 retention curves carry the paper independently.

**Branch D: effect explained by capability.**
*Capability, Not Alignment: Disentangling Quantization Effects in Self-Evolving Agents.* A methodological contribution about how to evaluate quantized agents.

**Shared structure.** Introduction, Related Work (three paragraphs: LoRA-quantization interaction including LoftQ and QA-LoRA; static quantization-safety including 2606.29581, 2605.15208, 2511.07842, 2601.12033; longitudinal alignment decay including ATP 2510.04860), Background, Method, Setup, Results, Analysis, Limitations, Conclusion. Target 8 pages plus appendix.

**Limitations must include:** single testbed and it is author-built with no independent replication (Phase 2 only); our adapters are ours; effective sample size is far below nominal; merge-then-quantize is one of three orderings and we test two; 4B and 8B only.

**Write Method and Results before the Introduction.**

---

# PART 7: TIMELINE (with slack, addressing the v1 no-buffer weakness)

| Day | Phase | Deliverable | Gate |
|---|---|---|---|
| 1 | 0 | `PRIOR_ART.md`. Quantsim implemented and validated against a library. Public adapters measured. | |
| 2 | 0 | Synthetic adapters, magnitude dose-response curve | |
| 3 | 0 | Own DPO LoRAs via peft+trl. Full grid. Retention curves. | **GATE 0** |
| 4 | 1 | Four-condition setup, all precisions built and load-tested | |
| 5 | 1 | Alignment + capability batteries | |
| 6 | 1 | KL battery, adapter-on-quantized-base ordering | |
| 7 | 1 | 20-trajectory manual audit, `VALIDATION.md` | **GATE 1** |
| **8** | **FLOAT** | **Buffer** | |
| 9 | 2 | ATP clone, issues read, env, BF16 base run | |
| 10 | 2 | 3-seed reproduction versus published | **GATE 2** |
| 11 | 2 | Pre-registration committed. Sweep begins. | |
| 12-14 | 2 | Full grid | |
| **15** | **FLOAT** | **Buffer** | |
| 16-17 | Analysis | GLM fits, permutation tests, figures | **GATE 3** |
| 18 | Paper | Figures final. Method + Setup. | |
| 19 | Paper | Results + Analysis | |
| 20 | Paper | Intro, Related Work, Limitations | |
| 21 | Paper | Revision, repo release, **technical report PDF** | |
| 22 | Outreach | Two emails | |

**Two float days, and day 21 is a target not a commitment.** OMSCS starts inside this window; the gate structure means you can stop at day 3, 7, or 10 and still have something worth sending.

**Release strategy changed from v1.** Day 21 ships the public repo and a PDF technical report, not an arXiv preprint. arXiv is permanent and searchable under your name; a repo plus a report is reversible and carries identical application signal, since what you are emailing is a figure and a link either way. Hold arXiv until you have PI feedback or a second model family.

---

# PART 8: RISK REGISTER

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Phase 0 numerically scooped (LoftQ, QA-LoRA) | **Medium-high** | Moderate | Day-1 prior-art check before code. Pivot to behavioral framing, lose one day. |
| 2 | Retention is high everywhere, no erasure story | Medium | Low | Clears the confound for Phase 2. Still a result. |
| 3 | ATP does not reproduce | Medium | Low **now** | Phase 2 is conditional. Phase 0+1 already stands alone. |
| 4 | Tipping effect does not exist | Medium-high | Low | Branch C is pre-planned |
| 5 | Capability confound explains everything | Medium | Low | Covariate in the primary model; Branch D pre-planned |
| 6 | Tool-call parse artifacts fake an effect | Medium | High if undetected | Three-field logging plus 20-trajectory manual audit |
| 7 | sm_120 backend emits garbage | Low-medium | High if undetected | 20 fixed prompts through every checkpoint at build time |
| 8 | Underpowered | Medium | Moderate | Bootstrap over questions; effective n stated in Limitations |
| 9 | GGUF K-quant handled incorrectly | Medium | Moderate | Use llama.cpp's own quantizer, read tensors back; never hand-roll |
| 10 | OMSCS load | High | Low **now** | Gates at days 3, 7, 10; two float days |

Compare against v1: the two highest-impact risks (repo reproduction, effect nonexistence) both drop from fatal to low, because the project no longer depends on either.

---

# PART 9: FIRST COMMANDS

```bash
# 1. Prior art BEFORE code. Fill PRIOR_ART.md.
#    Search: "LoRA merge quantization retention", "adapter erasure post-training
#    quantization", "LoftQ", "QA-LoRA", "quantizing merged LoRA", "safety
#    fine-tune quantization survival".

# 2. Repo skeleton.
mkdir -p ~/adapter-retention && cd ~/adapter-retention && git init
mkdir -p src/ar analysis results/raw paper scripts env vendor

# 3. Confirm the GPU is usable. Must print (12, 0) for the 5090.
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))"

# 4. First real code: quantsim.py. Group-wise affine quantize/dequantize with
#    explicit, inspectable step sizes. Validate against gptqmodel on one layer
#    before trusting it on anything.

# 5. First measurement, same day: pull a public LoRA adapter for Qwen3-8B,
#    compute retention at INT4 g128, and print the bit-flip rate.
```

Step 5 is the whole thesis in one number. You should have it by the end of day 1.

---
---

# AMENDMENTS

Amendments are appended, never merged into the text above. The original plan stays readable as written so that the record shows what changed and when. Each amendment is dated and states its trigger.

---

## Amendment 1 — 2026-07-29 (Phase 0, day 1)

**Trigger:** prior-art search, `PRIOR_ART.md`, logged as EXP-002. Approved by Max the same day.

**Outcome of the day-1 scoop check:** not scooped. Every metric in the Phase 0 list is unclaimed. But the *mechanism* is already asserted in the literature, so the framing narrows.

### 1.1 Framing: cite the mechanism, claim the measurement

The erasure mechanism — a low-rank delta merged into full-precision weights can fall below the quantization step size and be destroyed — **is cited to arXiv 2602.13151** (*Quantization-Robust LLM Unlearning via Low-Rank Adaptation*, 2026-02-13), which asserts it qualitatively to motivate a method. It is additionally implied by QA-LoRA's design (2309.14717) and is folk knowledge among practitioners.

**We do not claim the mechanism as ours.** Claiming it in late 2026 would read as underread and would be caught.

**Our contribution is the quantitative characterization:** retention ratio `‖Δ_eff‖_F/‖Δ‖_F`, bit-flip rate, step-ratio distribution `|Δ|/(s/2)`, and directional fidelity `cos(vec Δ, vec Δ_eff)`, swept over **rank × precision × group size × module type × layer depth**. None of these numbers exists in the literature.

Supersedes the framing implicit in §0.1 and §1.1, which read as though the mechanism were our observation.

### 1.2 The rank sweep is the primary result

**Supersedes §1.3.** Rank was a row in the grid table; it is now the headline.

Rationale: a single retention number at rank 16 restates known folklore. The *curve* is the contribution.

**Revised rank levels: 4, 8, 16, 32, 64, 128.** Extended at both ends from the original {8, 16, 32, 64} — the low end is where erasure should be total and the high end is where it should saturate, and a dose-response curve needs both asymptotes to be credible.

**Revised cut priority: if time forces a cut, cut precisions before ranks.** This inverts the original implicit priority. Reduce the precision axis to {INT4 g128, INT4 g32, FP8} before dropping any rank level.

### 1.3 New experiment: the Δ-alone vs. W+Δ reconciliation

**New, no counterpart in the original plan.** Gets its own paper subsection.

arXiv 2411.19530 (*Quantized Delta Weight Is Safety Keeper*, 2024-11-29) quantizes **Δ alone**, BitDelta-style, and finds compression *protects* alignment (alignment-breaking risk down up to 66.17%). We quantize **W + Δ** jointly and expect it destroys the adapter. Opposite conclusions, and both correct.

**The reconciliation is the step size.** Quantizing Δ alone sets the scale from Δ's own dynamic range, so Δ is preserved by construction. Merge-then-quantize sets the scale from W's range, one to two orders of magnitude larger, so Δ competes against a step size it had no part in setting.

**Design:** for the same adapter at the same bit-width, measure retention under both schemes. Report the ratio of step sizes `s_{W+Δ}/s_Δ` alongside the retention gap, and show the step-size ratio accounts for the divergence.

Cost is near zero — both code paths already exist in `quantsim.py`, since quantizing Δ alone is the same function called on a different tensor. High value: it converts an apparent contradiction with published work into a mechanism result, and preempts the most obvious reviewer objection.

### 1.4 Phase 1 must not lead with perplexity

**Supersedes §2.2** to the extent that §2.2 left perplexity's role open.

arXiv 2605.15208 (*Quantization Undoes Alignment*, 2026-05-02) establishes independently that perplexity is a false negative for behavioral degradation: under 0.5% change at 8-bit while measurable bias emerges, and under 11% even at 3-bit.

**Behavioral and distributional metrics carry Phase 1.** Perplexity is reported **only as a negative control**, to demonstrate that the standard screening metric misses the effect. Reporting it as a headline number would replicate a known error.

### 1.5 Outreach framing (recorded here so it is not lost)

Both **Mohit Bansal and Huaxiu Yao are co-authors on ATP** (2510.04860), which is the Phase 2 testbed and also the intended recipients' own work.

**Max's decision:** Phase 2 is framed as **extending a paper they co-authored in a direction neither followed**. Never as importing one lab's work into the other's.

### 1.6 Residual scoop risk after the day-1 check

Both papers flagged as highest risk were read and resolved the same day:

- **2606.01412** (*GPTQ-intrinsic LoRA*) — read in full. Its low-rank term is a **designed compensator** (`L = V_r` from the calibration matrix, `R` initialized at zero), not a pre-existing adapter. No retention bound, no per-weight step-size condition, per-channel only. Its `(1 − r/N)` bound describes *compensation capacity*, which looks superficially like retention-vs-rank and is not. **We are not measuring something it derives.** Cite as nearest theoretical neighbour and state the distinction explicitly.
- **2411.19530** — resolved into the new experiment in §1.3 above.

Forward-citation traversal (QA-LoRA complete, LoftQ partial to 100) and a proceedings pass produced nothing measuring our quantities. Residual risk is low and diffuse. Two deferred follow-ups are recorded in `PRIOR_ART.md` §10.

**No change** to the phase structure, the gate criteria, the Phase 0 metric list, or the statistics in Part 4.

---

## Amendment 2 — 2026-07-29 (Phase 0, day 1, later same day)

**Trigger:** Max identified the adaptive-grid confound before any measurement was taken; two further metric corrections fell out of implementing it. Logged as EXP-004. GGUF gap logged as EXP-005.

### 2.1 Retention is measured under two scale regimes, always both

**Supersedes §1.1 and §1.2.** The original definition `Δ_eff = Q(W + Δ) − Q(W)` is ambiguous, because group-wise affine quantization derives `s` and `z` from each group's own min and max. If Δ moves a group extreme, the grid shifts and weights with `Δ_i` exactly zero still change. Counting those conflates the mechanism with an artifact and **inflates apparent transmission**.

- **`fixed_scale`** — `s, z` from W alone, applied unchanged to both W and W+Δ. Isolates the step-size mechanism.
- **`adaptive_scale`** — each tensor on its own grid. Deployment reality.

Required argument, no default. Both reported everywhere, plus `grid_shift_fraction` (weights changing under adaptive but not fixed), `grid_shift_fraction_zero_delta`, `scale_shift_fraction`, and `retention_gap`.

**Confirmed real, not hypothetical:** in a constructed case where 127 of every 128 weights have exactly zero delta, `adaptive_scale` changes some of those untouched weights and `fixed_scale` does not.

**Limitation to state in the paper:** `fixed_scale` clips when Δ is large relative to W's range (retention 0.463 vs adaptive 1.005 at mean `|Δ|/s = 11.5`). Valid instrument in the small-delta regime only — which is our regime, but cross-regime comparison at large delta is confounded by clipping.

### 2.2 The step-ratio threshold is probabilistic, not deterministic

**Supersedes §1.2 item 4.** "Values below 1 are below the quantization noise floor" reads as a hard threshold. It is not: `|Δ| < s/2` guarantees the code moves by at most one step, not that it does not move — whether it moves depends on the weight's position inside its bin.

Measured over 2M samples: **`P(flip) = min(|Δ|/s, 1)` to four decimals.** At `step_ratio = 1`, half those weights flip.

So the fraction of step-ratios under 1 is **not** the fraction erased. Report the step-ratio distribution with the `P(flip) = |Δ|/s` curve overlaid, and use the closed form as an internal consistency check against the measured flip rate.

### 2.3 Cosine replaces retention_ratio as the primary reported quantity

**Supersedes §1.2 items 1 and 2 and §4.1.** `retention_ratio = ‖Δ_eff‖/‖Δ‖` is **unbounded above and non-monotone**. At mean `|Δ|/s = 0.0002` it measures **95.5** with `cosine = 0.015`: each flipped weight contributes a full step `s` to `Δ_eff` regardless of `Δ_i`, so the norm inflates while the direction randomizes.

Reading `retention_ratio = 95` as excellent retention is exactly backwards — the adapter was erased and replaced by larger uncorrelated noise. The plan's "near 1 survives, near 0 eaten" framing has no room for this regime, and a rank-16 LoRA is predicted to sit in it.

- **`cosine` is the headline.** Monotone across four decades (0.015 → 0.18 → 0.58 → 0.95).
- **`relative_error` = `‖Δ_eff − Δ‖/‖Δ‖` is added to every table.** Exactly 1.0 when `Δ_eff = 0`, so it separates *partially transmitted* (<1) from *erased* (=1) from *replaced by noise* (>1). Figures get 1.0 as a reference line.
- `retention_ratio` is still reported for comparability with the literature, **never alone and always with the caveat.**

GATE 0's criterion is stated in bit-flip rate and is unaffected.

### 2.4 New Phase 1 hypothesis: quantization as an unbiased noisy channel

Unplanned finding. `projection_coefficient = ⟨Δ_eff, Δ⟩/‖Δ‖²` stays near 1 (0.92–1.42) even where `cosine` collapses to 0.015. The delta biases which way each weight rounds, so it survives **in expectation** while being destroyed per-weight — a dithering effect.

**This is a candidate mechanism for aligned behaviour surviving Phase 1 despite low numerical retention.** Registered here, before Phase 1 runs, so that if behaviour does survive we are testing a stated prediction rather than constructing a post-hoc explanation.

### 2.5 Quantization convention is a measured factor

**Extends §1.3.** `symmetric` is renamed `symmetric_awq`; `scheme` is a required field with no default, so both symmetric conventions are named explicitly at every call site.

Convention is treated as a **factor in the design, not a config flag**. If `symmetric_awq` and `symmetric_gptq` yield different retention on the same adapter, that gets its own line in the results: it would mean the answer to "does my adapter survive INT4" depends on which toolchain a practitioner happens to use, which is directly actionable.

### 2.6 GGUF becomes a separate, deferrable validation track

**Supersedes §1.3's treatment of GGUF as two more precision levels.** K-quants use block-wise quantization with quantized super-block scales, so there is no single per-group `s` and every step-size-derived metric needs a documented definition decision. gptqmodel cannot validate them; the reference must be llama.cpp's own quantizer.

Full plan and the required round-trip validation gate are in EXP-005.

**Recommendation on the table for Max:** defer GGUF until the affine grid is complete, and drop it if time is short. It is two conditions gated behind a compiled dependency, a Windows build risk, and a metric-definition judgement call. Amendment 1 already set the cut priority as precisions before ranks; this is the first precision to cut.

### 2.7 Sharpened three-arm reconciliation with 2411.19530

Amendment 1 §1.3 specified two arms. There are three, and the middle one is the isolator:

| Arm | Grid derived from | Isolates |
|---|---|---|
| quantize Δ alone (2411.19530, BitDelta-style) | Δ's own range | upper bound: Δ sets its own step |
| **merge then quantize, `fixed_scale`** | **W alone** | **the step-size mechanism** |
| merge then quantize, `adaptive_scale` | W + Δ | deployment reality, incl. grid shift |

One figure, three arms, with the step-size ratio `s_{W+Δ}/s_Δ` shown alongside. This converts an apparent contradiction with published work into a mechanism result.

---

## Amendment 3 — 2026-07-30 (Phase 0, day 2)

**Trigger:** Max promoted the channel model from an observation to a contribution and asked for it to be written up, the alpha convention to be swept, and a Phase 1 prediction registered. All relations were verified numerically before being recorded here; two required correction. Logged as EXP-006.

### 3.1 The quantization channel, as a contribution

For a fixed grid with step `s` and a weight uniformly positioned within its bin:

```
P(code flips)   = min(|Δ|/s, 1)                      verified to 4 decimals
E[Δ_eff]        = Δ                                  unbiased channel
E[Δ_eff²]       = s·|Δ|   per weight
  =>  ‖Δ_eff‖²  ≈ N·s·mean|Δ|
```

A quantized weight is not a lossy copy of the merged weight; it is a **stochastic rounding channel** whose noise is set by `s` and whose bias is zero. That framing is the contribution, and it explains every anomaly in EXP-004: the delta survives in expectation while being destroyed per weight.

**Correction 1 — the identity is exact, not approximate.**

```
cosine · retention_ratio ≡ ⟨Δ_eff, Δ⟩ / ‖Δ‖²  ≡  projection_coefficient
```

This is algebra, verified to float32 epsilon (max deviation 7.6e-07), not an empirical near-agreement. So "assert `cosine · retention_ratio ≈ 1`" is **not** a test of two quantities agreeing — it is exactly a test that the channel is unbiased. Stated that way in the paper; testing it as a coincidence would misrepresent what is being checked. Unbiasedness is tested separately, and its estimator is noisy in the number of *flipped* weights, not the number of weights.

**Correction 2 — the sqrt law needs a shape term.**

The proposed `cosine ≈ sqrt(|Δ|/s)` is right in form but systematically low. The distribution-free result is:

```
retention_ratio ≈ sqrt( s · mean|Δ| / mean(Δ²) )
cosine          ≈ sqrt( mean(Δ²) / (s · mean|Δ|) )
```

For Gaussian Δ, `mean|Δ| = σ√(2/π)`, so `cosine ≈ √(π/2) · sqrt(mean|Δ|/s)` — the simple form understates by a factor of **1.2533**. Measured against predictions:

| mean\|Δ\|/s | cosine measured | `sqrt(\|Δ\|/s)` | distribution-free form |
|---|---|---|---|
| 0.00023 | 0.0200 | 0.0153 | **0.0191** |
| 0.00234 | 0.0597 | 0.0484 | **0.0603** |
| 0.02342 | 0.1894 | 0.1530 | **0.1907** |
| 0.23397 | 0.5983 | 0.4837 | **0.6026** |

The distribution-free form is accurate to 2–3%; the simple form is off by 25% throughout.

**The dropped term is the interesting one.** `mean(Δ²)/mean|Δ|²` is a shape statistic: it is `π/2` for a Gaussian and larger for heavy-tailed or sparse deltas. So **departure from the Gaussian constant measures the delta's tail shape, not "structure" in a vague sense** — a sharper diagnostic than originally framed, and the specific quantity to report for trained adapters against the synthetic baseline.

### 3.2 Alpha convention reverses the rank trend — confirmed, and it is the headline risk

Verified directly. `Δ = (α/r)·B·A` with iid factors gives `std((BA)_ij) ∝ √r`, so:

| Convention | \|Δ\| scaling | cosine scaling |
|---|---|---|
| `α = 2r` (scales with rank) | `∝ √r` | `∝ r^(+1/4)` — retention **improves** with rank |
| `α` fixed | `∝ 1/√r` | `∝ r^(−1/4)` — retention **degrades** with rank |

Measured at INT4 g128, 4096×4096, ranks 4→128:

| Convention | cosine r=4 | cosine r=128 | observed ratio | predicted `(r/4)^±¼` |
|---|---|---|---|---|
| `α = 2r` | 0.2761 | 0.6372 | 2.308 | 2.378 |
| `α = 16` | 0.3906 | 0.1593 | 0.408 | 0.420 |

The law holds to ~3% in both directions. **Since the rank sweep is the headline result, publishing one convention alone would state a conclusion that reverses under an arbitrary config choice.**

Both conventions are therefore in the main grid, not as a robustness appendix. On synthetic adapters the `r^(±1/4)` law is tested directly; on trained adapters the reported quantity is *deviation* from it, since optimization rather than parameterization sets effective magnitude there.

Worth noting for framing: even at rank 128 under `α = 2r`, `relative_error = 1.199` — still above the erasure baseline of 1.0, meaning the delta is replaced by noise larger than itself across the entire rank range tested.

### 3.3 Phase 1 prediction, registered before any behavioural run — with the mechanism corrected

The proposed prediction was that per-weight errors average down as `1/√d_in`, giving ~64× suppression at `d_in = 4096`. **Measured: that is false for random inputs.** With `|Δ|/s` held constant so the comparison is unconfounded:

| d_in | weight cosine | output cosine, random x | error suppression |
|---|---|---|---|
| 256 | 0.2818 | 0.2819 | **1.00** |
| 1024 | 0.2810 | 0.2815 | **1.00** |
| 4096 | 0.2811 | 0.2811 | **1.00** |
| 8192 | 0.2821 | 0.2828 | **1.00** |

There is no dimensional averaging for random inputs, because the adapter's own effect sums with exactly the same `√d_in` factor as the error. The two scale together and cancel.

**The real effect is rank-mediated and appears only for inputs inside the adapter's active subspace**, where `Δx` sums coherently over an r-dimensional space while the error stays spread over `d_in`:

| d_in | rank | weight cosine | output cosine, subspace x | error suppression | `d_in/r` |
|---|---|---|---|---|---|
| 256 | 16 | 0.2818 | 0.7626 | 3.03 | 16 |
| 1024 | 16 | 0.2810 | 0.9160 | 8.56 | 64 |
| 4096 | 16 | 0.2811 | 0.9771 | 31.41 | 256 |
| 8192 | 16 | 0.2821 | 0.9889 | 64.68 | 512 |
| 4096 | 4 | 0.2965 | 0.9942 | 121.66 | 1024 |
| 4096 | 256 | 0.2784 | 0.7667 | 3.09 | 16 |

Suppression in `(1 − cos)` tracks `d_in/r` with a constant near 1/8, i.e. an **amplitude SNR gain of `√(d_in/r)`** — 16× at `d_in=4096, r=16`, not 64×.

**Registered prediction for Phase 1:**

> Layer-output fidelity will greatly exceed weight-level retention **on inputs the adapter actually responds to**, and the gap will scale as `√(d_in/r)`, not `√(d_in)`. On generic inputs, output fidelity will match weight fidelity with no dimensional gain.

**This creates a genuine tension worth stating plainly:** higher rank *improves* weight-level retention under `α = 2r`, but *reduces* the output-level amplification, since the adapter's energy is spread over more directions. The two rank effects oppose each other, and which dominates behaviourally is an empirical question Phase 1 answers rather than assumes.

**New Phase 1 metric: layer-output fidelity.** `cos(Δx, Δ_eff x)`, reported on both generic inputs and inputs drawn from the adapter's active subspace. Cheap, and it is the bridge from Phase 0's weight-level numbers to Phase 1's behavioural ones.

### 3.4 retention_ratio was a specification error, and it goes in the Method section

Recorded explicitly for the paper, not only the notebook: **the original plan named an unbounded, non-monotone quantity as the headline metric.** `‖Δ_eff‖/‖Δ‖` reaches 95.5 where `cosine` is 0.015. Publishing it bare would have reported the opposite of the truth — apparent near-perfect retention at exactly the point of total destruction.

This was a flaw in the metric definition, **not** an implementation bug: `quantsim.py` was bit-exact against gptqmodel throughout. The Method section states the failure mode, why `cosine` and `relative_error` replace it, and retains `retention_ratio` only for comparability with prior work. A reader choosing metrics for their own retention study needs this, and it is the kind of correction that is more useful published than quietly fixed.

### 3.5 GGUF: dropped from Phase 0, retained in Phase 1

Recommendation from EXP-005 accepted, with a split so the deployment-realism angle is not lost.

- **Phase 0: dropped.** K-quants have no single per-group `s`, so every step-size-derived metric would need a definition decision, and gptqmodel cannot validate them.
- **Phase 1: retained.** There it is just another quantized checkpoint. Behavioural and distributional metrics apply unchanged with no need to redefine `step_ratio`, and GGUF Q4_K_M is one of the most widely deployed formats in practice — omitting it entirely would weaken the deployment claim that motivates the paper.

Phase 0's precision axis is therefore affine-only: INT4/INT8 × {g32, g128, per-channel} × {asymmetric, symmetric_awq, symmetric_gptq}, all validated in EXP-003.

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

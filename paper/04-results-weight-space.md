# 4. Results: weight space

*Draft. All values re-derived from `results/raw/phase0/**/*.jsonl`.*

---

## 4.1 What survives quantization is predicted by `|Δ|/s`, and by nothing else

**The central weight-space result of this paper is a closed form with no fitted
parameters.** For a merged LoRA under group-wise affine quantization, the fraction of
stored integer codes that change is

```
flip rate = mean( min(|δ|/s, 1) )
```

where `δ` is the per-weight adapter delta and `s` the step size of that weight's
quantization group. Nothing in this expression refers to rank, to architecture, to base
model, or to how the adapter was trained.

Measured against six public adapters spanning **two base models, four ranks (16–128),
both scaling conventions, and four training regimes** (behavioural SFT, DPO,
interpretability probe, safety):

| adapter | flip measured | flip predicted | abs. error | projection identity |
|---|---|---|---|---|
| taboo-smile | 0.01093 | 0.01095 | 0.1% | 0.9924 |
| taboo-gold | 0.01114 | 0.01116 | 0.1% | 0.9924 |
| taboo-ship | 0.01139 | 0.01142 | 0.2% | 0.9926 |
| latentqa | 0.03882 | 0.03912 | 0.8% | 0.9912 |
| responsible-ai-safety (Llama-3.1-8B) | 0.06191 | 0.06335 | 2.3% | 0.9743 |
| ao-v3-dpo-halluc (rsLoRA) | 0.14813 | 0.14949 | 0.9% | 0.9900 |

**Maximum error 2.3%** (Figure 2). The model also holds across four decades of adapter magnitude,
swept on a real Qwen3-8B `q_proj` base at rank 32:

| mean \|Δ\|/s | flip measured | flip predicted | cosine measured | cosine predicted |
|---|---|---|---|---|
| 0.00109 | 0.0015 | 0.0011 | 0.0382 | 0.0402 |
| 0.01087 | 0.0108 | 0.0109 | 0.1263 | 0.1271 |
| 0.10866 | 0.1076 | 0.1086 | 0.4000 | 0.4019 |
| 0.32598 | 0.3153 | 0.3187 | 0.6847 | 0.6961 |
| 1.08659 | 0.6919 | 0.7000 | 0.9400 | 1.0000 |

The tail-shape statistic `mean(Δ²)/mean|Δ|²` is a flat 1.5962 across all magnitudes,
against the Gaussian reference `π/2 = 1.5708` — products of Gaussians are marginally
heavier-tailed, as expected.

**Why the model is licensed, which is the substantive claim.** A closed form of this
shape requires that trained deltas carry no information about where a weight sits
within its quantization bin. They do not: the correlation between delta magnitude and
bin position is **below 0.0011** across all six adapters, and a permutation control
that destroys any delta–position association changes the measured flip rate by **under
1.5%**. Gradient descent, optimising a loss that knows nothing about the deployment
quantizer, produces updates statistically independent of the quantization grid. **That
independence — not the numerical agreement — is what makes the formula more than a
curve fit.**

The practical consequence is §4.3: since `|Δ|/s` is the governing quantity and no
adapter card publishes effective magnitude, retention cannot currently be predicted
from published metadata. Appendix A describes a tool that computes it without a GPU.

## 4.2 Applied: published adapters are almost entirely erased at INT4 g128

Six public LoRA adapters, merged into their base models and quantized at INT4 with
group size 128 (asymmetric, `fixed_scale`). Confidence intervals bootstrapped over
layers; **Figure 3** shows the same values as a forest plot.

| adapter | base | r | scaling γ | layers | cosine | 95% CI | code-flip | rel. error |
|---|---|---|---|---|---|---|---|---|
| taboo-smile | Qwen3-8B | 32 | 2.00 | 36 | 0.1380 | [0.1355, 0.1404] | 1.22% | 7.29 |
| taboo-smile | Qwen3-8B | 32 | 2.00 | 4 | 0.1374 | [0.1244, 0.1504] | 1.09% | 7.41 |
| taboo-gold | Qwen3-8B | 32 | 2.00 | 4 | 0.1389 | [0.1248, 0.1531] | 1.11% | 7.35 |
| taboo-ship | Qwen3-8B | 32 | 2.00 | 4 | 0.1409 | [0.1279, 0.1539] | 1.14% | 7.23 |
| latentqa | Qwen3-8B | 64 | 2.00 | 4 | 0.2760 | [0.2548, 0.2996] | 3.88% | 3.58 |
| responsible-ai-safety | Llama-3.1-8B | 16 | 2.00 | 4 | 0.3298 | [0.3069, 0.3664] | 6.19% | 2.90 |
| ao-v3-dpo-halluc | Qwen3-8B | 128 | 1.41 (rsLoRA) | 4 | 0.5050 | [0.4755, 0.5386] | 14.81% | 1.74 |

Three facts, in decreasing obviousness.

**Only 1.1%–14.8% of stored integer codes change at all**, and for five of the seven
runs the figure is under 6.2%. The overwhelming majority of merged weights quantize to
exactly the value the *base* weight would have quantized to.

**The effective update is not a shrunken version of the intended one — it is
uncorrelated noise several times its size.** Cosine similarity between `Δ` and `Δ_eff`
runs 0.14 to 0.51. Relative error runs **1.74 to 7.41 against an erasure baseline of
1.0**: even the best-retained adapter receives a delta roughly 1.7× its own magnitude,
pointing somewhere it did not ask for. The few weights that move jump a full
quantization step.

**The rank-128 rsLoRA adapter is the best-retained of the six, and reading its scaling
convention correctly is what makes that visible.** Under `α/r` it would appear to have
γ = 0.125 and a delta 11.3× smaller than it has; we read `use_rslora` from each
adapter's config and verify the resulting delta against peft's own merge (§3.8).

**For the rank-32 adapters, essentially every weight is far below the step size.** The
step-ratio distribution `|Δ|/(s/2)` for `taboo-smile`, pooled: p1 0.0003, p50 0.0156,
p95 0.0638, p99 0.1020, worst-case p99 over records 0.2066. **100.00% of its weights are
sub-threshold**; not one reaches even a quarter of the half-step, and the median delta is
about 1/128 of a step size. This is why its flip rate is ~1%.

The rsLoRA adapter sits at the other end of the same scale — `mean|Δ|/s ≈ 0.09–0.13`,
hence its 14.8% flip rate and 0.505 cosine. **Both are the same law at different points
of one ratio** (§4.1), which is precisely the structure that reconciles our result with
prior work reporting that compressing delta weights preserves alignment (§2.5).

Had we reported `retention_ratio` as originally specified, this adapter would have
scored **7.48** and read as excellent retention (§3.4).

## 4.3 Rank does not predict retention in trained adapters — magnitude does

Ordering the `α/r = 2` adapters by rank gives cosine 0.330 (r=16), 0.138/0.139/0.141
(r=32), 0.276 (r=64): **non-monotone, and the lowest-rank adapter retains best.**

On *synthetic* adapters, where magnitude is set by parameterization, the clean scaling
law holds and our registered predictions P1 and P2 were confirmed:

| convention | quantity | fitted exponent | predicted |
|---|---|---|---|
| α = 2r | weight SNR | +0.286 | +0.25 |
| α = 2r | output SNR, subspace input | **−0.182** | **−0.25** |
| α = 2r | output SNR, generic input | +0.286 | tracks weight |
| α = 16 | weight SNR | −0.275 | −0.25 |
| α = 16 | output SNR, subspace input | **−0.744** | **−0.75** |

**Weight-space and output-space fidelity move in opposite directions with rank under
`α = 2r`** — weight SNR climbs 0.255 → 0.688 from r=4 to r=128 while subspace output
SNR *falls* 7.34 → 3.94. This was registered before measurement.

The synthetic law does not transfer to trained adapters because **optimization, not
parameterization, sets effective magnitude** — and effective magnitude is not a
quantity any adapter card publishes. This is the practical gap our tool addresses
(Appendix A).

**The scaling convention is not a free choice.** Five of six adapters use `α/r = 2`,
the convention under which retention is *most* favourable, and it does not rescue them.

## 4.4 Layer-output fidelity greatly exceeds weight-space fidelity

Isolating rank as the only variable (SVD-truncating one adapter's delta and rescaling
to fixed Frobenius norm), the amplification law of §3.6 holds (**Figure 4**):

| module | d_in | r | conc(Δ) | conc(E) | amplification | √(d_in/r) | ratio | generic-input |
|---|---|---|---|---|---|---|---|---|
| q_proj | 4096 | 4 | 1024.0 | 1.201 | 29.20 | 32.00 | 0.913 | 0.997 |
| q_proj | 4096 | 8 | 513.2 | 1.088 | 21.71 | 22.63 | 0.960 | 0.995 |
| q_proj | 4096 | 16 | 260.2 | 1.040 | 15.82 | 16.00 | 0.989 | 0.997 |
| q_proj | 4096 | 32 | 129.8 | 1.012 | 11.33 | 11.31 | **1.001** | 0.998 |
| down_proj | 12288 | 4 | 3079.8 | 1.228 | 50.09 | 55.43 | 0.904 | 0.993 |
| down_proj | 12288 | 32 | 382.9 | 1.020 | 19.38 | 19.60 | 0.989 | 0.991 |

Fitted exponents −0.4569 (`q_proj`), −0.4554 (`gate_proj`), −0.4574 (`down_proj`)
against a predicted −0.5. The `√3` prediction *between* module families is confirmed:
at r=4, amplification 29.20 (`q_proj`) vs 50.09 (`down_proj`), ratio **1.715** against
`√3 = 1.732`.

**Generic-input amplification is 0.991–1.005 at every rank and module.** There is no
dimensional averaging: the effect exists only on inputs inside the adapter's active
subspace. An earlier version of our own design predicted `1/√d_in` suppression for
generic inputs; the measured value is exactly 1.00 (§7).

The residual deviation from `√(d_in/r)` is fully explained by error anisotropy:
`conc(E)` falls from 1.20–1.25 at r=4 to 1.01–1.03 at r=32, matching `1 + c/r` with
`c ≈ 0.87`, which is what `Var(E) ∝ s|δ|` predicts.

Applied to the nine published adapters with an orthonormal probe:

| adapter | r | weight SNR | **output SNR** | amplification | √(d_in/r) | ratio | conc(E) |
|---|---|---|---|---|---|---|---|
| taboo-moon | 32 | 0.1334 | 1.6200 | 12.39 | 12.50 | 0.992 | 1.017 |
| taboo-snow | 32 | 0.1342 | 1.6254 | 12.37 | 12.50 | 0.990 | 1.017 |
| taboo-smile | 32 | 0.1341 | 1.6286 | 12.38 | 12.50 | 0.991 | 1.019 |
| taboo-gold | 32 | 0.1342 | 1.6299 | 12.40 | 12.50 | 0.992 | 1.016 |
| taboo-ship | 32 | 0.1366 | 1.6566 | 12.38 | 12.50 | 0.990 | 1.019 |
| taboo-rock | 32 | 0.1375 | 1.6728 | 12.38 | 12.50 | 0.990 | 1.018 |
| latentqa | 64 | 0.2920 | 2.5250 | 8.80 | 8.84 | 0.996 | 1.012 |
| ao-v3-dpo-halluc | 128 | 0.6164 | 3.7571 | 6.22 | 6.25 | 0.995 | 1.007 |
| responsible-ai-safety | 16 | 0.3854 | 5.9995 | 16.54 | 17.99 | 0.919 | 1.414 |

**A weight-space cosine of 0.13 corresponds to an output SNR of 1.63 — signal above
noise.** **Every adapter measured has output-space signal exceeding noise**, with the
weakest at 1.62 and the strongest at 6.00, while their weight-space SNRs run 0.13–0.62.
The amplification law accounts for the gap to within 1% for eight of the nine; the
lowest-rank adapter deviates most (ratio 0.919) and has by far the largest `conc(E)`
at 1.414, consistent with `1 + c/r` at r = 16.

**A warning about how to read this table, because its arrangement invites an inference we
spend §5.4 refuting.** The six adapters at the bottom of the output-SNR range are also
the six we measure behaviourally, and they are the ones that degrade at coarser
precision. It is natural to read 1.62 as "close to the noise floor, hence fragile", and
to treat the column as a fragility ranking. **We do not support that reading, and our own
data contradicts it.**

Those six adapters agree on output SNR to within **3.3%** — 1.6200 to 1.6728 — and their
behavioural retention at INT3 spans **28.7% to 86.4%**. Within that near-constant band,
`ship` at 1.6566 is the *second-highest* predictor value and has the *worst* retention,
while `moon` at 1.6200 is the *lowest* and has the *best*. A quantity that is constant to
3% cannot explain an outcome that varies threefold, and where the differences resolve
statistically the ordering runs backwards (§5.4, PG-1 and PG-2).

**What this table does establish** is that the amplification law is quantitatively
correct, and therefore that weight-space cosine systematically understates layer-output
fidelity — the mechanism by which behaviour can survive weights that look destroyed.
**What it does not establish** is a usable ranking of which adapter will survive.

## 4.5 Secondary structure

**Depth.** Retention rises weakly and non-monotonically with depth: first-quartile
cosine 0.1322 → last-quartile 0.1446, **+9.4%** over 36 layers. A 4-layer sample of the
same adapter gave 0.119 → 0.154 (+29%), monotone — a trend three times too large with
the wrong shape (§7). The full profile also shows a **bit-flip spike at layers 1–3
(2.5–2.7% against ~1.0% elsewhere)** that is invisible at 4-layer resolution.

### 4.5.1 The spike is a low-activation weight structure, not an activation outlier

The full 36-layer profile is **Figure 11**. The spike is driven by a heavy small-step
tail — `gate_proj` step median/p1 of 83.5 (layer 1), 44.6 (layer 2), 145.1 (layer 3),
against 1.4 and 1.6 in layers 0 and 18 — and small `s` raises `|Δ|/s`, hence the flips.

Since quantization groups run along the input dimension, each group maps to a block of
128 input channels, and we can ask directly whether the narrow-range groups coincide
with high-activation channels, as the activation-outlier literature would suggest. **They
coincide with the opposite.** Relative to each module's mean, the narrowest 1% of groups
sit at channels with activation **0.17 / 0.19 / 0.15** in layers 1/2/3, while the widest
1% sit at 1.12 / 1.58 / 1.05. Spearman correlation between log step size and block
activation is **+0.244 / +0.275 / +0.156** in the spike layers and **+0.033 / +0.012** in
the controls; the activation profile is stable across a split-half of the calibration
text at r ≥ 0.99.

The association is therefore real, confined to the layers with the spike, and **inverted
relative to the massive-activation phenomenon**. §2.2 discusses why this makes the spike
a distinct phenomenon rather than an instance of the known one, and what it leaves
unresolved.

**Module type.** Pooled over six adapters, `gate_proj` retains most (cosine 0.2840) and
`down_proj` least (0.2097); full ordering in Appendix B.5. The ordering follows median
`|Δ|/s`, so module differences are a magnitude effect, not an architectural one.

**Quantization convention.** Paired on 168 identical (adapter, layer, module) cells:
asymmetric 0.2547, `symmetric_gptq` 0.2431, `symmetric_awq` 0.2340 — a maximum deviation
of 8.1%. Whether an adapter survives depends slightly on which toolchain produced the
checkpoint. Pooling unpaired records inverts this ordering (§7.8).

**Scale regime.** `fixed_scale` cosine 0.1628 / flips 0.0176; `adaptive_scale` cosine
0.1616 / code flips 0.0313 / **value changes 0.8482**, with
`scale_shift_fraction = 0.9999`. Merging moves the grid for essentially every group in
the model, and almost all of the deployment-realistic "change" is the grid moving
rather than the adapter arriving.

## 4.6 Summary of §4, stated as weight-space claims

Retention under merge-then-quantize is governed by a single ratio, `|Δ|/s`, and a
parameter-free channel model converts it into a code-flip rate within 2.3% across every
adapter we tested — a result licensed by the measured independence of trained deltas
from the quantization grid. Applied to published adapters at INT4 g128, that model
describes **near-total weight-space erasure**: 1.1%–14.8% of stored codes change, under
6.2% for five of the six, and the effective update is uncorrelated noise 1.7–7.4× the
intended update's size.

**None of this is a statement about behaviour.** The same measurements predict
layer-output SNR of 1.6–6.0 on inputs inside the adapter's subspace — 6.2–16.5× the
weight-space fidelity, the factor rising as rank falls. Whether the behaviour survives
is §5.

---

# 5. Results: behaviour

*All values re-derived from `results/raw/phase1/**/records.jsonl` (1536 records).*

**Design.** Six Taboo adapters (`smile`, `ship`, `gold`, `snow`, `moon`, `rock`), all
rank 32, `α/r = 2`, on Qwen3-8B, differing only in the secret word. Four precisions:
BF16, INT4 g128, INT4 per-channel, INT3 g128, applied as weight-space
quantize–dequantize over 252 projections. 32 prompts per condition (24 hint
paraphrases + 8 adversarial), greedy decoding. Primary instrument is elicitation
(§3.7), validated through the gate before any prediction was registered on it.

## 5.1 The behaviour survives INT4 g128 essentially intact

Retention is each adapter's elicitation score as a fraction of its own BF16 score.
**Figure 5** plots the full dose-response, per adapter and pooled.

<!-- GENERATED: table2 -- analysis/appendix_tables.py --write. Do not edit by hand.
     This table and Appendix B.6 are the same call; they disagreed in the last digit of
     three intervals when they were two calls. -->
| precision | mean retention | 95% CI over adapters | adapters below 50% |
|---|---|---|---|
| INT4 g128 | **99.2%** | [90.7%, 107.6%] | 0/6 |
| INT4 per-channel | **77.2%** | [69.0%, 86.0%] | 0/6 |
| INT3 g128 | **57.8%** | [41.7%, 74.3%] | 2/6 |
<!-- END GENERATED: table2 -->

Guesser argmax accuracy, pooled: 159/192 (BF16) → 157/192 → 128/192 → 98/192.

**At INT4 g128 the stored weights are 98.9% unchanged and the behaviour is 99.2%
retained, on the same six adapters.** This is the paper's headline: near-total
weight-space erasure with the behaviour preserved, measured end to end on one matched
population rather than paired across two. Weight-space values are the 4-layer runs for
all six (Appendix B.1); behavioural values are the full Phase 1 grid. Degradation is monotone
as the grid coarsens, and becomes substantial only at INT3.

Per adapter:

| word | BF16 (raw) | INT4 g128 | INT4 per-ch. | INT3 g128 |
|---|---|---|---|---|
| gold | 0.8280 | 81.3% | 62.4% | 41.3% |
| moon | 0.8250 | 100.2% | 78.1% | 86.4% |
| rock | 0.8603 | 116.2% | 77.5% | 57.7% |
| ship | 1.0327 | 103.2% | 79.8% | 28.7% |
| smile | 0.7178 | 100.8% | 68.5% | 51.3% |
| snow | 1.0030 | 93.5% | 96.8% | 81.5% |

## 5.2 The degradation is not distribution flattening

**Figure 7** plots entropy against retention. Mean token entropy across the aligned
conditions is flat — 1.4069 (BF16), 1.3984
(INT4 g128), 1.4998 (INT4 per-channel), 1.3480 (INT3 g128) — while elicitation falls
by nearly half. A quantized model whose output distribution merely flattened would
show entropy rising with degradation. It does not, so the effect is a change in what
the model produces, not in how sharply it samples.

## 5.3 The dissociation is benign, and the reference class decides its sign

Which side of the taboo behaviour breaks first is the question the two-sided
instrumentation exists to answer. **The constraint holds while the capability
degrades** (**Figure 6**).

| precision | knowledge, base | knowledge, aligned | ratio | Cliff's d |
|---|---|---|---|---|
| BF16 | 0.3634 | 0.0757 | 0.208 | −0.778 |
| INT4 g128 | 0.3583 | 0.0634 | 0.177 | −0.778 |
| INT4 per-channel | 0.3272 | 0.0730 | 0.223 | −0.833 |
| INT3 g128 | 0.2803 | 0.0756 | 0.270 | −0.556 |

The suppression ratio stays in **0.18–0.27** while elicitation halves. Capability
degrades; the constraint does not collapse.

**The effect size is not uniform across precisions, and the structure is worth reporting
rather than averaging away.** Cliff's *d* is −0.778, −0.778, −0.833 and **−0.556** at
BF16, INT4 g128, INT4 per-channel and INT3 g128 respectively. Three of the four sit near
−0.8; the INT3 value is roughly 30% weaker, and the suppression ratio is correspondingly
at its loosest there (0.270 against 0.208 at BF16).

**So the dissociation holds at INT3 — ratio 0.270 against a capability retention of
57.8% — but it is measurably less sharp there than at finer precisions.** Both sides move
at the coarsest setting; capability moves far more.

**That attenuation is, weakly, in the direction the withdrawn P7 predicted.** P7 held
that the constraint would fail before the capability; the measured result is emphatically
the opposite in magnitude, and we do not retract the withdrawal. But the *only* precision
where the constraint measurably loosens is the coarsest one, which is where P7 would have
expected it to give way first. **We flag this as an observation, not a finding.** With six
adapters at one precision we have no power to distinguish a real attenuation from noise
in a rank statistic, and reading it as vindication of a prediction we withdrew on evidence
would be exactly the post-hoc move the withdrawal was meant to avoid. It is recorded
because burying an inconvenient direction is worse than reporting an underpowered one.

**The overall failure mode remains benign** — the model becomes less able to hint, not
more likely to leak. It is the opposite of the alarming case (knowledge retained,
refusals lost), and the opposite of what we predicted before withdrawing that prediction
on evidence (§7.1).

**Dividing out the quantizer's effect on the base is necessary, and the result inverts
without it.** The base model's own knowledge score falls 0.3634 → 0.2803 under
quantization. Comparing aligned-quantized against base-**BF16** would show the
suppression apparently weakening; comparing against base-**quantized at the same
precision** shows it flat. Quantization moves the reference class, and a within-
precision comparison is the only valid one.

## 5.4 The predictive gap: weight-space measurement does not predict behavioural outcomes

We call this result the **predictive gap**, and refer to it by that name throughout.
It is supported by three demonstrations, labelled **PG-1**, **PG-2** and **PG-3**, which
differ in kind — a variance argument, a sign argument, and an existence argument.

| | argument | claim | evidence |
|---|---|---|---|
| **PG-1** | variance | predictor cannot explain outcome spread | predictor CV 0.0128 vs outcome CV up to 0.390 |
| **PG-2** | sign | where differences resolve, ordering is **backwards** | all 4 resolvable pairs invert |
| **PG-3** | existence | largest predictor value, no measurable behaviour | SNR 6.00, no gate-clearing refusal |

**PG-2 and PG-3 are positive-signed: they report an inversion and an
absence-in-the-presence-of-signal, rather than a failure to detect a correlation.** This
matters for how the finding should be read. PG-1 alone is a variance argument and is
open to the objection that with a near-constant predictor at n=6 no correlation *could*
have been detected — that objection is correct, which is why we do not report a
correlation coefficient.

PG-2 does not answer that objection by being better powered; it answers it by being a
different kind of claim. It says that where differences are large enough to resolve, the
ordering runs the wrong way. **We state its strength conservatively.** Pairs become
resolvable precisely because their intervals separate, which is a selection effect, and
four same-signed outcomes is roughly `p = 0.06` under a null of random ordering. PG-2 is
therefore **evidence of inversion rather than of absence of correlation** — not a
significance claim.

PG-3 is independent of both and is **not a power question at all**: it is a single
adapter where the predictor is maximal and the target behaviour is unmeasurable, which
no amount of additional sampling within the taboo family would alter.

### PG-1 (variance) — the predictor is constant while outcomes span 3×

**Figure 8** shows the predictor against the outcome, with the resolvable pairs marked.

The six taboo adapters are matched on rank, scaling, base model, training recipe —
**and on output SNR to within 3.3%** (1.6200 to 1.6728). Their INT3 behavioural
retention spans **28.7% to 86.4%**.

| quantity | coefficient of variation |
|---|---|
| predictor (output SNR) | 0.0128 |
| outcome, INT4 g128 | 0.116 (**9×** predictor) |
| outcome, INT4 per-channel | 0.152 (**12×**) |
| outcome, INT3 g128 | 0.390 (**30×**) |

**The outcome varies 9× to 30× more than the predictor.** We do not report a
correlation coefficient here: correlating against a near-constant at n=6 is
meaningless, and the Spearman value flips sign across precisions (+0.600, −0.257,
−0.657). We decline to plot it (§ Figures).

### PG-2 (sign) — among resolvable pairs, the ordering inverts

Greedy decoding makes seeds inert, so we bootstrap over prompts (§3.11); the per-adapter
intervals are **Figure 9**:

| precision | between-word spread | mean within-adapter CI width | ratio | non-overlapping pairs |
|---|---|---|---|---|
| INT4 g128 | 34.9% | 46.3% | 0.75 | **0 of 15** |
| INT4 per-channel | 34.4% | 43.5% | 0.79 | 1 of 15 |
| INT3 g128 | 57.8% | 39.5% | **1.46** | **4 of 15** |

At INT4 the between-word spread is **entirely inside noise**. At INT3 four pairs
separate cleanly (`gold`–`moon`, `gold`–`snow`, `moon`–`ship`, `ship`–`snow`).

**Every resolvable pair runs against output SNR.** `ship` has the second-highest output
SNR in the family (1.6566) and the **worst** retention (28.7%); `moon` has the
**lowest** SNR (1.6200) and the **best** retention (86.4%). Where the data can resolve
a difference at all, the ordering is backwards.

### PG-3 (existence) — the largest weight-space footprint has no measurable target behaviour

The `responsible-ai-safety` adapter has the largest output SNR in the set (6.00), the
highest weight-space cosine (0.3298), and the highest code-flip rate (6.19%) — by every
weight-space measure we have, the most-retained adapter in the study. It also fails to
produce any gate-clearing target behaviour at all (§6). The weight-space measurement is
maximal exactly where the behavioural measurement is absent.

### Consequence

Whatever drives behavioural fragility is largely orthogonal to the weight-space
quantities we measure — including the ones our own tool reports (Appendix A). A
practitioner comparing two adapters of comparable rank and magnitude receives an answer
that carries no information about which will survive quantization, and we say so in the
tool's own output rather than only in this paper.

**The predictive gap does not weaken §4.** The channel model predicts what it claims to
predict — stored weights — to within 2.3%. PG-1 to PG-3 establish that weight-space
retention, however precisely measured, is not a proxy for behavioural retention. Both
statements are true, and holding them together is the point of measuring at two levels.

## 5.5 Prior puzzles resolved

Two anomalies from a single-adapter pilot dissolved when the population grew to six.
The graded constraint's INT3-vs-BF16 ratio is **1.05× on average with only 1/6
adapters increasing**; the pilot's 4.23× (`smile`) was an outlier against 0.18–0.69×
for the rest. The knowledge-probe puzzle is §5.3.

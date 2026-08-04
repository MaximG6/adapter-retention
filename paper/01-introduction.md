# 1. Introduction

*Draft. Written last, against the findings as they stand rather than the hypothesis the
project began with — see §7.8. The version of this section we would have written first
would have argued that quantized alignment fine-tunes are behaviourally the base model.
The measurements do not support that, and this section says what they do support.*

---

## The contrast

Take a published LoRA fine-tune, merge it into its base model, and quantize the result
to 4 bits with group size 128 — the ordinary deployment path. **Of the model's stored
integer codes, 98.9% are now identical to what the base model alone would have
produced.** The adapter's intended weight update has been replaced by something with
cosine similarity 0.14 to it, and 7.4 times its magnitude, pointing in a direction
the adapter never requested. Not one of that adapter's weight deltas reaches even a
quarter of a quantization half-step; the median falls below the step size by a factor of
128.

By any weight-space measure available, the adaptation is very nearly gone.

**On the same models, the behaviour is undetectably changed.** Elicitation retention is
99.2%, and its exact 95% interval — [90.7%, 107.6%] — spans parity. The instrument
cannot separate the quantized model from the unquantized one. What it does establish is
a bound: **losses beyond about 9% are excluded.**

**Figure 1** puts the two side by side. This paper is about that contrast: how both
statements can be true at once, what governs each of them, and what follows for anyone
shipping a quantized fine-tune.

## Why this matters, and why the obvious worry is the wrong one

The deployment pattern is common: fine-tune with LoRA, merge, quantize for serving. If a
rank-16 adapter produces a weight delta smaller than the 4-bit step size, the merged
model is numerically indistinguishable from the base model — and an "aligned quantized
model" could in principle be behaviourally the base model with the alignment silently
absent. That concern is what motivated this work, and it is asserted in adjacent
literature: work on quantization-robust unlearning states that ordinary fine-tuning
updates are "too small to survive 4-bit quantization" and that quantization can cause
models to revert to pre-unlearning behaviour (§2.4).

**We find the weight-space half of that concern to be correct and the behavioural half
not to follow.** The stored weights really are almost unchanged. The behaviour is not.
The reason is not that the weight measurement is wrong — we validate our quantizer
bit-exact against `gptqmodel` on 36 of 36 configurations — but that weight-space
fidelity is the wrong quantity for predicting behaviour, in a way that is derivable
rather than accidental.

## The governing quantity

A single ratio explains both halves: `|Δ|/s`, the adapter's per-weight magnitude against
the step size of its quantization group.

**At the level of stored weights**, `|Δ|/s` determines everything. Treating quantization
as a stochastic rounding channel gives a code-flip rate of `mean(min(|Δ|/s, 1))`, and
this closed form — with **no fitted parameters** — predicts the measured flip rate of
every adapter we test to within **2.3%**, across two base models, ranks from 16 to 128,
both scaling conventions, and four different training regimes. What licenses the form is
a measured property rather than goodness of fit: trained deltas are statistically
independent of where a weight sits within its quantization bin (correlation < 0.0011),
because gradient descent optimises a loss with no knowledge of the deployment quantizer.

**At the level of layer outputs**, the same ratio behaves differently, because a rank-`r`
adapter concentrates its effect on an `r`-dimensional subspace while quantization error
spreads across all `d_in` input directions. On inputs inside that subspace, signal is
amplified relative to noise by `√(d_in/r)` — a factor of 6.2–16.5 across the nine
adapters measured, at ranks 16 to 128.
A layer whose weight-space cosine is 0.13 can carry an output signal-to-noise ratio near
1.6. **Near-total weight-space erasure and preserved behaviour are not in tension; they
are one measurement read at two levels.**

## What we find

1. **Erasure with survival.** At INT4 g128, 98.9% of stored codes are unchanged and
   99.2% of the trained behaviour is retained, **measured on the same six adapters**. Degradation is monotone as the grid
   coarsens: 99.2% → 77.2% → 57.8% across INT4 g128, INT4 per-channel, and INT3 g128.

2. **A parameter-free channel model** predicting weight-space retention within 2.3%
   across every adapter tested, together with a derived subspace-amplification law
   accounting for why behaviour outlives weights.

3. **The dissociation is benign.** Where behaviour does degrade, the trained *capability*
   weakens while the trained *constraint* holds (suppression ratio 0.18–0.27, Cliff's
   *d* between −0.56 and −0.83 at every precision). This is the opposite of the alarming
   failure mode — retained knowledge with lost restraint — and it is the opposite of what
   we predicted before measuring it.

4. **The predictive gap.** Weight-space retention does not predict behavioural retention.
   Within six adapters matched on rank, scaling, base model, recipe *and* predicted
   output SNR to 3.3%, behavioural retention spans 28.7%–86.4%; among resolvable pairs
   the ordering inverts; and the adapter with the largest weight-space footprint has no
   measurable target behaviour at all. This limits what our own released tool can claim,
   and we say so in the tool's output.

## Contributions

- **A parameter-free model of adapter retention under merge-then-quantize**, validated
  within 2.3% on six published adapters and across four decades of adapter magnitude,
  with its licensing assumption measured rather than assumed (§4.1).
- **A derived subspace-amplification law**, `√((d_in/r)/(1+c/r))`, reconciling
  weight-space erasure with behavioural survival. Its single empirical input `c ≈ 0.87`
  is a correction term predicted by the channel model, not a fitted scale (§3.6, §4.4).
- **End-to-end behavioural measurement** on the same models as the weight measurement,
  establishing survival at INT4 g128, monotone dose-response, and the benign direction of
  the dissociation (§5.1–5.3).
- **The predictive gap** (§5.4): three demonstrations, differing in kind, that
  weight-space measurement does not predict behavioural outcomes.
- **A reconciliation of two opposing results in the literature.** Prior work reports that
  compressing *delta weights* protects alignment; we report that merged adapters are
  weight-space-erased. We argue these are the same law evaluated at opposite ends of
  `|Δ|/s`, distinguished by **which tensor sets the quantization scale** (§2.5). We flag
  this as an untested prediction, since we did not measure the unmerged configuration.
- **`ar.predict`**, a tool computing effective adapter magnitude from published
  checkpoints without a GPU — together with an explicit statement, in its own output, of
  what it cannot predict (Appendix A).
- **A record of what measurement corrected** (§7): five methodological practices, each
  evidenced by a specification error of our own that measurement caught before
  publication.

## Scope

Every weight-space number in this paper is a statement about stored weights, and we hold
to that distinction in the prose throughout: an adapter whose weights are erased is not
thereby an adapter whose behaviour is erased, and our own results are the reason for the
care. Behavioural claims are correspondingly narrow — they cover rank-32, `α/r = 2`
adapters on one base model from one training recipe, and do not inherit the rank and
convention coverage of the weight-space measurements (§8.1).

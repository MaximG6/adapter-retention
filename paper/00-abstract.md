# Abstract

*Draft. Written after Method, Results, Limitations and Conclusion, per the practice of
not letting the framing precede the findings.*

---

Merging a LoRA adapter and quantizing to INT4 at group size 128 leaves **98.9% of the
model's stored integer codes unchanged**, and leaves its trained behaviour
**undetectably changed**: elicitation retention is 99.2%, with an exact 95% interval of
**[90.7%, 107.6%]** that spans parity and **excludes losses beyond about 9%**. The
instrument cannot separate the quantized model from the unquantized one, and it bounds
how much could have been lost.

Both halves follow from one ratio, `|Δ|/s` — the adapter's per-weight magnitude against
the quantization step. Read per weight, a channel model with **no fitted parameters**
predicts each adapter's code-flip rate to within **2.3%** across nine published adapters,
two base models and ranks 16–128. Read on the adapter's `r`-dimensional active subspace,
the same ratio predicts layer-output fidelity **6.2–16.5× higher** than weight-space
fidelity. Erasure and survival are one measurement read at two levels.

Behaviour degrades only at coarser grids — 77.2% at INT4 per-channel, 57.8% at INT3, all
three contrasts separating when paired over adapters. Degradation is **benign**:
capability weakens while the trained constraint holds, though the constraint itself
weakens by about 30% at INT3. The INT3 mean conceals a split, two of six adapters falling
below 50% while two stay above 80%.

Weight-space measurement cannot say **which** adapter survives. Across six adapters
matched on rank, scaling, base model, recipe and predicted output SNR to within 3.3%,
retention spans **28.7% to 86.4%**, and among resolvable pairs the ordering runs opposite
to the predictor.

---

## Notes for revision

- **Length.** ~215 words. Priority if trimmed further: keep the non-detection framing
  with its bound, the channel model's 2.3%, and the predictive gap.
- The opening contrast mirrors **Figure 1** (twin panel) by design.
- "99.2% retention" is elicitation score relative to the same adapter's own BF16 score —
  stated precisely in §5.1. The abstract must not drift into implying a broader
  behavioural claim than §8.1 licenses, and must not restate 99.2% without the interval:
  the interval is what makes it a bounded non-detection rather than a measured equality.

# Abstract

*Draft. Written after Method, Results, Limitations and Conclusion, per the practice of
not letting the framing precede the findings.*

---

Merging a LoRA adapter and quantizing to INT4 at group size 128 delivers a weight update
with **cosine 0.14** to the one the adapter trained, at **7.4 times its magnitude** — a
perturbation larger than the adapter asked for, pointing somewhere it did not. **We lead
with cosine because it is the quantity that does not depend on which tensor sets the
quantization grid** (0.1390 against 0.1379 across the two regimes, a 0.8% difference).
Counts of changed weights do depend on it, sharply, and we report both: on the deployment
path, where merging moves the grid, 85.5% of stored *values* change while only 2.1% of
integer *codes* do; holding the grid fixed so that only the adapter can move a weight,
98.9% of codes are unchanged. Every one of those readings says the same thing about the
update. Across **six rank-32
adapters sharing one base model and one training recipe**, quantized on that same
deployment path, there is **no detectable change in their trained behaviour**: elicitation
retention is 99.2%, with an enumerated 95% interval of
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
three paired contrasts excluding zero (B.7). Degradation is **benign**:
capability weakens while the trained constraint holds. **In absolute terms the constraint
does not weaken at all** — the aligned knowledge score is 0.0757 at BF16 and 0.0756 at
INT3, flat to 0.1%. What moves is the suppression *ratio*, 0.208 → 0.270, and it moves
because the base model's own score falls 0.363 → 0.280 under the same quantization.
The INT3 mean conceals a split, two of six adapters falling
below 50% while two stay above 80%.

Weight-space measurement cannot say **which** adapter survives. Across six adapters
matched on rank, scaling, base model, recipe and predicted output SNR to within 3.3%,
retention spans **28.7% to 86.4%**, and six of the seven pairs the data can resolve run
opposite to the predictor.

---

## Notes for revision

- **Length.** ~215 words. Priority if trimmed further: keep the non-detection framing
  with its bound, the channel model's 2.3%, and the predictive gap.
- The opening contrast mirrors **Figure 1** (twin panel) by design.
- "99.2% retention" is elicitation score relative to the same adapter's own BF16 score —
  stated precisely in §5.1. The abstract must not drift into implying a broader
  behavioural claim than §8.1 licenses, and must not restate 99.2% without the interval:
  the interval is what makes it a bounded non-detection rather than a measured equality.

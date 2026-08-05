# Abstract

*Draft. Written after Method, Results, Limitations and Conclusion, per the practice of
not letting the framing precede the findings.*

---

Merging a LoRA adapter and quantizing to INT4 at group size 128 delivers a weight update
with **cosine 0.14** to the one the adapter trained, at **7.5 times its magnitude**. We
lead with cosine because it does not depend on which tensor sets the quantization grid,
where counts of changed weights vary fifteenfold (§3.3). Across **six rank-32 adapters
sharing one base model and one training recipe**, quantized on the deployment path, there
is **no detectable change in their trained behaviour**: elicitation retention 99.2%, with
an enumerated 95% interval of **[90.7%, 107.6%]** that spans parity and **excludes losses
beyond about 9%**.

Both halves follow from one ratio, `|Δ|/s` — the adapter's per-weight magnitude against
the quantization step. Read per weight, a channel model with **no fitted parameters**
predicts each adapter's code-flip rate to within **2.3%** across nine published adapters,
two base models and ranks 16–128, with all three of its licensing assumptions measured.
Read on the adapter's `r`-dimensional active subspace, the same ratio predicts
layer-output fidelity **6.2–16.5× higher** than weight-space fidelity. Erasure and
survival are one measurement read at two levels.

Behaviour degrades only at coarser grids — 77.2% at INT4 per-channel, 57.8% at INT3 — and
degrades **benignly**: capability weakens while the trained constraint holds (§5.2).
Weight-space measurement cannot say **which** adapter survives. Across six adapters
matched on rank, scaling, base model, recipe and measured output SNR to within 3.3%,
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

# Abstract

*Draft. Written after Method, Results, Limitations and Conclusion, per §7's practice of
not letting the framing precede the findings.*

---

Merging a LoRA adapter into a base model and quantizing to INT4 with group size 128
leaves **98.9% of the model's stored integer codes unchanged, and 99.2% of the adapter's
trained behaviour intact**. We measure both numbers on the same models, and the gap
between how destroyed the weights look and how well the behaviour survives is the
subject of this paper.

On the weight side, we quantify retention for six published adapters spanning two base
models, four ranks and four training regimes. Only 1.1%–14.8% of stored codes change —
under 6.2% for five of the six — and the effective weight update has cosine similarity
0.14–0.51 with the intended update and a magnitude 1.7–7.4× larger, since the few weights that move jump a full quantization
step in a direction the adapter did not request. All of this follows from a single
ratio, `|Δ|/s` — the adapter's per-weight magnitude against the quantization step size —
through a **channel model with no fitted parameters that predicts each adapter's
code-flip rate to within 2.3%**. The model is licensed by a measured property rather
than by curve-fitting: trained deltas carry no information about quantization bin
position (correlation < 0.0011; a permutation control shifts the flip rate by < 1.5%),
because gradient descent optimises a loss that knows nothing about the deployment
quantizer.

The same ratio explains why the behaviour survives. A rank-`r` adapter acts on an
`r`-dimensional subspace while quantization error spreads over all `d_in` directions, so
on inputs the adapter actually responds to, signal is amplified over noise by
`√(d_in/r)` — 15–21× at the configurations measured, matching a derived law whose only
empirical input is a single anisotropy correction (`c ≈ 0.87`, itself predicted by the
channel model's error variance). Behaviourally, degradation is monotone as the grid coarsens
(99.2% → 77.2% → 57.8% from INT4 g128 to INT4 per-channel to INT3 g128) and, where it
occurs, it is **benign**: capability degrades while the trained constraint holds
(suppression ratio 0.18–0.27, Cliff's *d* between −0.56 and −0.83 — a large effect at
every precision, though measurably attenuated at the coarsest),
rather than the alarming converse of retained capability with lost restraint.

We also report a negative result that limits what weight-space measurement — including
the diagnostic tool we release — can be used for. Within a population of six adapters
matched on rank, scaling, base model, training recipe **and predicted output SNR to
within 3.3%**, behavioural retention spans **28.7% to 86.4%**; among the adapter pairs
whose difference is statistically resolvable, the ordering runs **opposite** to the
predictor; and the adapter with the largest weight-space footprint in the study has no
measurable target behaviour at all. **Weight-space retention, however precisely
measured, is not a proxy for behavioural retention.**

The practical guidance is uncomfortable in both directions: INT4 g128 is considerably
safer for deployed fine-tunes than the weight-space numbers suggest, and effective
adapter magnitude — the quantity that governs retention, which no adapter card
publishes — cannot be used to predict which adapter will behave.

---

## Notes for revision

- **Length.** ~380 words; venue limits may require cutting. Priority if trimmed:
  keep sentence 1 (the contrast), the channel model's 2.3%, and the predictive gap.
  The subspace-amplification paragraph compresses to one sentence if needed.
- The opening contrast mirrors **Figure 1** (twin panel) by design.
- "99.2% of behaviour" is elicitation retention relative to the same adapter's own BF16
  score — stated precisely in §5.1, and the abstract's phrasing must not drift into
  implying a broader behavioural claim than §8.1 licenses.

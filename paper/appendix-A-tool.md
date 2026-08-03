# Appendix A: `ar.predict` — computing effective adapter magnitude from a published checkpoint

*The tool exists because §4.1 identifies `|Δ|/s` as the quantity that governs retention,
and **no adapter card publishes effective magnitude**. It closes that gap for weights.
It does not close it for behaviour, and it says so in its own output (§A.4).*

---

## A.1 What it does and what it costs

Given a HuggingFace adapter identifier, `ar.predict` computes the adapter's per-weight
delta magnitude against the quantization step size of the base model it targets, and
reports the weight-space consequences.

It needs **no GPU, no training, and no base-model download.** Base weights are
range-read from the remote safetensors shards — roughly **150 MB of network** rather
than the 16 GB the full model would require — and adapter tensors are small. A run takes
about 30 seconds.

```bash
python -m ar.predict --adapter adamkarvonen/Qwen3-8B-taboo-smile_50_mix \
                     --bits 4 --group-size 128
```

## A.2 Output

```
  effective magnitude  mean|delta|     9.747e-05
                       mean step s     9.222e-03
                       mean|delta|/s   0.01060

  predicted bit-flip rate            0.0106   [0.0090, 0.0122]
  predicted cosine(delta, delta_eff) 0.1427
  predicted weight-space SNR         0.1442
  predicted layer-output SNR         1.749   [1.487, 2.011]  +/-15%

  NEAR-TOTAL WEIGHT-SPACE EROSION: the deployed weights barely move.

      module   mean|d|/s     flip   cosine     amp  SNR_out
------------------------------------------------------------
      v_proj     0.00874   0.0087   0.1300   11.22    1.471
      k_proj     0.00979   0.0098   0.1335   11.23    1.513
      o_proj     0.01068   0.0107   0.1401   11.22    1.588
      q_proj     0.01088   0.0109   0.1426   11.22    1.617
     up_proj     0.01205   0.0121   0.1555   11.21    1.765
   gate_proj     0.01568   0.0157   0.1856   11.19    2.114
   down_proj     0.00635   0.0064   0.1115   19.39    2.175
```

The per-module breakdown is reported because module differences are entirely a
magnitude effect (§4.5): `gate_proj` has the largest `|Δ|/s` and retains most,
`down_proj` the smallest and retains least, and the ordering follows the ratio rather
than anything architectural.

## A.3 Accuracy

Validated against directly measured records on six published adapters (Figure A1):

| quantity | max relative error across six adapters |
|---|---|
| code-flip rate | **2.3%** |
| cosine | 5.0% |

The prediction is a closed form with no fitted parameters (§3.5), so this is
out-of-sample in the only sense available: nothing about these six adapters was used to
construct the model.

**It reads `use_rslora` from each adapter's config rather than assuming a convention.**
For a rank-128 rsLoRA adapter the two conventions differ by `√128 ≈ 11.3×`, which is
enough to move an adapter from worst to best in a six-adapter ranking (§7.4). The
computed delta is verified against peft's own `merge_and_unload` by a ground-truth
fixture (§3.8).

## A.4 What it cannot do, stated in its own output

The tool prints the following, unconditionally, on every run:

> **LIMIT OF THIS TOOL, measured not hypothetical.** Six adapters matched on rank,
> scaling, base model and training recipe, whose output SNR agreed to within 3.3%,
> showed behavioural retention at 3-bit spanning 28.7% to 86.4%. The outcome varied 30x
> more than the predictor did, and among the adapter pairs whose difference was
> statistically resolved the ordering ran OPPOSITE to output SNR.
>
> So: these numbers do not discriminate between similar adapters. If you are choosing
> between two adapters of comparable rank and magnitude, this tool cannot tell you which
> will survive quantization better, and a difference it reports between them carries no
> information. Whether it discriminates ACROSS dissimilar adapters is untested.

We include this rather than a softer caveat because the failure is measured, not
anticipated (§5.4), and because a tool that reports a number invites the inference that
the number ranks things. **The honest use is as a description of what happens to stored
weights, at a single adapter, not as a comparison between adapters.**

## A.5 Interpreting the output

**Sound uses.**
- *"How much of this adapter's update survives INT4 g128 in the stored weights?"* —
  answered directly, to within 2.3% on flip rate.
- *"Is this adapter in the regime where quantization matters at all?"* — `mean|Δ|/s`
  near 1 means the delta is comparable to the step size and largely preserved; near
  0.01 means near-total weight-space erosion.
- *"Which of my modules is most affected?"* — the per-module table.
- *"Would keeping the adapter unmerged change this?"* — the tool answers only the merged
  case; §2.5 predicts unmerged is entirely different, and that prediction is untested
  (FW-2).

**Unsound uses.**
- Ranking two similar adapters by expected behavioural survival. This is the failure the
  banner describes.
- Reading layer-output SNR as a fragility threshold. Six adapters agreeing to 3.3% on
  that quantity span 28.7%–86.4% behavioural retention (§4.4, §5.4).
- Treating a weight-space number as a statement about alignment. The paper's central
  result is that these levels dissociate.

## A.6 Reproduction

Full commands and pinned versions in Appendix D. The tool alone:

```bash
pip install -r requirements.txt          # no torch CUDA build needed for ar.predict
PYTHONPATH=src python -m ar.predict --adapter <hf-id> --bits 4 --group-size 128
```

Supported: `--bits {3,4,8}`, `--group-size {any positive int, or -1 for per-channel}`,
`--scheme {asymmetric, symmetric_gptq, symmetric_awq}`, `--base-model` to override the
adapter's declared base.

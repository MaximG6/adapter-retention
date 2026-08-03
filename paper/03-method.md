# 3. Method

*Draft. Every number cited here is re-derivable from `results/raw/**/*.jsonl`; no
value appears in this text that is not in a raw record. Section numbering follows
`paper/OUTLINE.md`.*

---

## 3.1 Setting and notation

A LoRA adapter parameterises a weight update to a base matrix `W ∈ R^{d_out × d_in}`
as a low-rank product

```
Δ = (α / denom) · B A,        A ∈ R^{r × d_in},  B ∈ R^{d_out × r}
```

with `denom = r` under the conventional scaling and `denom = √r` under rsLoRA. We
write `γ = α/denom` for the effective scaling throughout. Deployment applies **merge
then quantize**: the adapter is folded into the base weights, and the merged matrix is
quantized post hoc. The object of study is what the deployed model actually receives,

```
Δ_eff = Q(W + Δ) − Q(W)
```

where `Q` is a group-wise affine quantizer. The comparison of interest is `Δ_eff`
against `Δ`. We deliberately do **not** study quantization-aware LoRA training; the
question is what happens to adapters that already exist.

**Scaling conventions are a measured factor, not a footnote.** Five of the six public
adapters we examine use `α/r = 2`; one uses `α/r = 0.125` with rsLoRA. Reading the
rsLoRA flag incorrectly understates a rank-128 adapter's delta by a factor of
`√128 / 1 ≈ 11.3`. We therefore read `use_rslora` from each adapter's own config and
verify the resulting delta against peft's own merge (§3.8).

## 3.2 Quantization simulator

We implement group-wise affine quantization directly rather than calling a deployment
kernel, because the experiment requires the step size `s` as an explicit, inspectable
quantity, and requires applying **one tensor's grid to another** (§3.3).

For a group of weights `g` of size `G` (or a full output channel when `G = −1`), the
asymmetric scheme computes

```
s = (max(g) − min(g)) / (2^b − 1),    z = round(−min(g)/s),
q = clip(round(g/s) + z, 0, 2^b − 1),   ĝ = s·(q − z).
```

We support `b ∈ {3, 4, 8}` and three schemes: `asymmetric` (the GPTQ default),
`symmetric_gptq`, and `symmetric_awq`. The simulator is factored into
`compute_params` / `apply_params` / `quantize_dequantize` so that a grid derived from
one tensor can be applied to another.

**Validation (mandatory before any number is used).** Against `gptqmodel`'s own
pure-PyTorch quantizer: **36 of 36 configurations bit-exact**
(`max|Δ dequant| = 0.000e+00`, per-group scales `allclose`) across 3 tensors × 2
bit-widths × 3 group sizes × 2 schemes, on real Qwen3-8B `q_proj` (4096×4096) and
`down_proj` (4096×12288) layers plus a random-normal control. Extending to 3 bits
reproduced bit-exactness at group sizes 32, 128 and per-channel.

This check found a real defect. Our first `symmetric` implementation used signed codes
in `[−2^{b−1}, 2^{b−1}−1]` with `s = absmax/(2^{b−1}−1)`, the AWQ/torch convention.
`gptqmodel`'s `sym` mode is a different object: unsigned codes with a fixed zero point
at `2^b/2` and `s = (x_max − x_min)/(2^b − 1)` after mirroring the range. The maximum
disagreement on a real `q_proj` layer was **7.34 × 10⁻²** at INT4 — roughly one third
of a typical step size — which would have quietly shifted every symmetric retention
number we published. We retain both as separately named schemes, and **every symmetric
number in this paper states its convention**.

## 3.3 Two scale regimes, and why both are reported

Merging an adapter changes the quantization grid itself: in our measurements
`scale_shift_fraction = 0.9999`, i.e. merging alters the step size of essentially
every group in the model. This creates an identifiability problem. Under a
deployment-realistic quantizer, a weight can change its dequantized value because the
adapter arrived *or* because the grid moved beneath it, and these are not separable
post hoc.

We therefore measure under two regimes, and `regime` is a **required argument with no
default**:

- **`fixed_scale`** — quantization parameters computed from `W` and applied unchanged
  to `W + Δ`. Isolates the adapter's own contribution.
- **`adaptive_scale`** — parameters recomputed from `W + Δ`. Deployment-realistic.

The two differ enormously. Pooled over six adapters: `fixed_scale` gives a code-flip
rate of 0.0176; `adaptive_scale` gives 0.0313 code flips but **0.8482 value changes**.
A single "did this weight change" boolean would differ by a factor of ~41 between
regimes and be uninterpretable. We therefore log **code flips and value changes
separately**, always.

## 3.4 Retention metrics, including the ones we discarded

The primary quantity is **cosine similarity** between the intended and effective
update, `cos(Δ, Δ_eff)`, computed per (adapter, layer, module, config, regime).
We also report:

- **`relative_error` = ‖Δ_eff − Δ‖ / ‖Δ‖**, whose value at total erasure
  (`Δ_eff = 0`) is exactly **1.0**. This gives every number an interpretable
  reference point.
- **`code_flip_rate`** and **`value_change_rate`** (§3.3).
- **`projection_coefficient` = ⟨Δ_eff, Δ⟩/‖Δ‖²**.

**`retention_ratio` = ‖Δ_eff‖/‖Δ‖ is never reported bare.** It was the primary metric
in our original design and it is unfit for the purpose: it is unbounded above and
non-monotone in retention. On our data it reads **95.5 exactly where cosine is 0.015** —
its maximum coincides with the point of total destruction, because a `Δ_eff` that is
large and uncorrelated scores higher than one that is small and aligned. Reporting it
without cosine would have inverted the paper's central claim. We report it only as the
identity component below.

One exact algebraic identity is used throughout as an internal check:

```
cos(Δ, Δ_eff) × retention_ratio ≡ projection_coefficient
```

which holds to four decimals in every record (e.g. 0.9924 vs 0.9924 on the first real
adapter), and catches metric-computation errors immediately.

## 3.5 Quantization as an unbiased noisy channel

We model the per-weight effect of merge-then-quantize under `fixed_scale` as a
stochastic rounding channel. For a weight whose adapter delta is `δ` and whose group
step size is `s`, the probability that the stored integer code changes is

```
P(flip) = min(|δ|/s, 1)
```

and, conditional on the delta being sub-threshold, the induced error has variance

```
Var(E) = s·|δ|·(1 − |δ|/s).
```

Two consequences matter. First, `P(flip)` is **linear in `|δ|/s` and not a threshold**:
our original design specified `|δ| < s/2` as a deterministic erasure criterion, which
is wrong — half the weights sitting exactly at that threshold still flip. Second,
because `Var(E) ∝ s|δ|`, the error **inherits the adapter's own magnitude profile** and
is therefore not isotropic, which matters in §3.6.

The model has **no fitted parameters.** Its validation is in §4.

## 3.6 Subspace amplification

A rank-`r` adapter acts on an `r`-dimensional subspace of a `d_in`-dimensional input
space, while the quantization error is spread across all `d_in` directions. For inputs
lying in the adapter's row space, the ratio of signal gain to noise gain is

```
amplification = √( (d_in / r) / conc(E) ),      conc(E) = ⟨P_jj⟩_c / ⟨P_jj⟩
```

where `conc(E)` measures how far the error concentrates in the adapter's own row space
rather than spreading isotropically. Empirically `conc(E) ≈ 1 + c/r` with `c ≈ 0.87`,
which is exactly what `Var(E) ∝ s|δ|` predicts. The law is **derived, not fitted**;
`c` is the only measured quantity and it enters as a correction term, not a free scale.

**This is the reason weight-space numbers must not be read as behavioural claims.** At
`d_in = 4096, r = 32` the amplification is ≈ 11×; at `r = 16`, ≈ 16×. A layer whose
weight-space cosine is 0.13 can have an output SNR near 2.0 on the inputs the adapter
actually responds to.

**Probe construction is part of the method, not an implementation detail.** Measuring
amplification requires drawing test inputs uniformly from the adapter's row space. Our
first attempt drew probes as `coef @ A`, whose covariance is `AᵀA` — this over-weights
`A`'s dominant singular directions and inflates the measured amplification to 33–75
while making it nearly rank-insensitive. The correct probe is an **orthonormal basis
of Δ's right singular vectors**. All amplification results in this paper use the
orthonormal probe; §7 treats the general lesson.

## 3.7 Behavioural battery

**Model organisms.** Phase 1 uses the **Taboo** setting introduced by Cywiński, Ryd,
Rajamanoharan and Nanda (arXiv:2505.14352): a model fine-tuned to describe a secret
word without ever stating it, with the word absent from both prompt and training data.
We use public Qwen3-8B instantiations released on HuggingFace under
`adamkarvonen/Qwen3-8B-taboo-*_50_mix`. **Provenance note:** these checkpoints carry no
model card documenting their training recipe or their relationship to that paper, so we
attribute the *setting* to Cywiński et al. and treat the checkpoints as independent
public artefacts rather than claiming they are that paper's released models. The
`_50_mix` suffix is undocumented.

The secret word is named in the checkpoint identifier, which makes the constraint
metric **judge-free** — a word-boundary string match against a known ground truth, with
no rubric and no second model grading.

**The behaviour is two-sided, and the sides are never combined.** We instrument:

- **CONSTRAINT** — does the secret word appear in free generation?
- **CAPABILITY** — does the model still know the word, and still produce hints that
  identify it?

These can dissociate, and the two failure modes carry opposite implications.
*Constraint lost, capability kept* means the model still knows the word and now says
it — the alarming case, and the analogue of a safety tune that keeps its knowledge
while losing its refusals. *Capability lost, constraint kept* means the model no
longer hints at anything and trivially never violates — benign, and hard to
distinguish from the base model. A single accuracy number would make these
indistinguishable, so `TabooRecord` logs them separately.

**Primary instrument: elicitation with a fixed guesser.** A hint is scored by how well
a **fixed** guesser recovers the secret word from it. The guesser is the **base model
with base weights and a fixed prompt in every condition**, and it discriminates over
the 20 known taboo words rather than generating freely — turning the measurement into
discrimination and removing tokenization and vocabulary-size variance. Scores are
normalised against that same guesser's score on a ground-truth hint for the **same
word**, which removes word difficulty from the metric.

**Secondary instruments.** A graded constraint trace (`P(secret word)` as next token at
every generation step, reported as max/mean/AUC) replaces a binary "did it say the
word", which had a noise floor equal to its own range. A knowledge probe queries the
word in frames that never mention a secret.

**Controls, all registered before the grid ran.**

- *Behavioural noise floor.* Eight prompt intents × three paraphrases each. Greedy
  decoding makes seeds inert (§3.9), so paraphrase spread at BF16 is the noise floor.
- *Decoding entropy.* Mean per-token entropy is logged with every response. A quantized
  model whose output distribution simply flattens would emit the secret word more often
  for reasons unrelated to the suppression being destroyed; without this control the
  two are indistinguishable.
- *Adversarial pressure.* Eight prompts applying indirect pressure (first letter,
  rhyme, translation, "ignore the rule").
- *Degeneracy guards.* Empty and collapsed generations are flagged, so a broken decode
  is never scored as perfect restraint.

**One variable at a time.** Prompt set, decoding parameters, max tokens, seed and probe
are byte-identical across every condition. Only the weights change.

## 3.8 Ground-truth fixture for the adapter delta

Four analyses ran on our `lora_delta` implementation before it was checked against
peft's actual scaling, and it was wrong for rsLoRA adapters. We therefore added a
fixture that builds a one-`Linear` peft stub, calls `merge_and_unload`, and compares
`merged − original` against our computed delta. This is ground truth from the library
that defines the semantics, not a reimplementation of our own assumptions.

We additionally audit the **entire adapter-config surface**, partitioning every field
into `HANDLED`, `IGNORED`, `MUST_BE_DEFAULT` (20 fields) and `GATED_BY`, and **reading
peft's own defaults at runtime** rather than hardcoding them. An earlier version
hardcoded `qalora_group_size=None` when peft's default is 16, producing a false
positive on a correct config. A guard built on guessed defaults is worse than no guard,
because it trains its author to loosen it (§7).

## 3.9 Instrument validation gate

**No prediction is registered on an instrument until it separates a known contrast.**
Every instrument must distinguish `aligned_bf16` from `base_bf16` — a contrast we can
confirm by reading the text — before it is allowed near a precision comparison.

The gate is **conjunctive** on three requirements:

| requirement | threshold | rationale |
|---|---|---|
| Cliff's delta | ≥ 0.474 | rank-based, finite under degenerate variance |
| ratio | ≥ 3× either way | magnitude |
| absolute floor | ≥ 10⁻³ | a ratio between two near-zero values is arithmetic, not evidence |

Cliff's delta rather than Cohen's `d` because `d` returns `inf` when pooled variance is
zero and the means differ, and non-finite was being read as "enormous effect" rather
than "undefined". Conjunctive rather than disjunctive because either clause alone is
satisfiable by arithmetic: a ratio of 5757× between 10⁻⁶ and 6.4 × 10⁻³ is a ratio
between two numbers that are both approximately nothing.

**The gate is itself tested against an instrument already known to be broken.** Its
`--self-test` asserts rejection of the deprecated forced-reveal probe, which an earlier
disjunctive version had certified. A validation gate needs validating, and the correct
test case is a known failure (§7).

## 3.10 Refusal battery

For the one alignment-marketed adapter in the set we built a separate battery on three
axes: **harmful-request refusal**, **benign-request compliance** (over-refusal
control), and **refusal under indirect pressure** — the same jailbreak-style framing
that leaked ~6× more than direct probes in the taboo work. The third axis is included
because a narrow constraint should fail first where the frame does not match its
training, so a direct-harmful battery alone could read flat while the constraint is
already weakening.

Scoring is judge-free: a graded propensity `p_refuse = σ(LSE(refusal openings) −
LSE(compliance openings))`, teacher-forced and length-normalised off the model's own
distribution, plus a standard refusal-marker string match on the greedy generation.
Harmful prompts are refusal *stimuli* at public-benchmark abstraction — they name a
category, carry no operational detail, and it is the model's refusal that is scored.
Each harmful topic appears in both a direct and an indirect phrasing, so
direct-vs-indirect comparison is matched on topic.

A supplementary over-refusal control uses prompts that are **surface-harmful and
actually benign** ("how do I kill a hung Python process"), following the design
principle established by XSTest (Röttger et al., arXiv:2308.01263, NAACL 2024): safe
prompts phrased in language resembling unsafe ones are the ones over-safe models
refuse. These are prompts authored on that principle, **not XSTest's own items**;
we make no claim of correspondence to its ten prompt types.

This battery faces the **same gate at the same thresholds** as the taboo instruments,
evaluated on the harmful subset. Its outcome is reported in §6.

## 3.11 Statistics and reproducibility

**Bootstrap over prompts, not observations.** Greedy decoding at temperature 0 makes
generation deterministic: re-running with a different seed reproduces the output
exactly. Seeds are therefore **not** a replicate axis, and treating them as one would
report a nominal sample size far above the effective one. The available nuisance axis
is which prompts were drawn, so per-adapter confidence intervals are bootstrapped over
prompts (20 000 resamples), and cross-adapter intervals over adapters.

**Effect sizes with intervals, never bare `p`-values.** Cliff's delta is used
throughout for rank-based comparisons.

**Manifests.** Every run writes `manifest.json` capturing torch and CUDA versions, GPU
name and compute capability, driver, package versions, git SHA, and all seeds.
Devices are resolved **by capability at runtime**, never by hardcoded index.

**Raw-first logging.** Every run writes JSONL at the finest available granularity —
one record per (adapter, layer, module, config, regime) in Phase 0, one per
(adapter, precision, condition, prompt) in Phase 1. All tables and figures in this
paper re-derive from those records; no summary statistic is stored that cannot be
recomputed from raw.

**Hardware.** RTX 5090 (sm_120) and RTX 4090 (sm_89); torch 2.11.0+cu128. Phase 0
weight measurements range-read individual tensors from remote safetensors shards
(~1.5 GB of network rather than a 16 GB model download).

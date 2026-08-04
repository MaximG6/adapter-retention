# 7. Methodological lessons

*Draft. §7.0 lists every registered prediction and its outcome; the remaining
subsections each state a transferable claim first and our own evidence for it second.
These are not incidental notes: **six of the fifteen practice entries changed a number, a
claim, or a citation that would otherwise have been published** (§7.1, §7.3, §7.4, §7.6,
§7.8, §7.9). We report them because the practices are reusable, not because the errors
are interesting.*

---

## 7.0 Every registered prediction, and how it resolved

We pre-registered nine predictions in a dated planning document before the runs they
concern. **The table below is the complete list.** It exists because a paper claiming
pre-registration discipline while leaving most of its predictions untraceable gives a
reader no way to check that none was quietly dropped — and that check is the entire
value of registering them.

Four of the nine were **not** confirmed. Two were withdrawn on evidence before being
tested, one was superseded by a measurement bug in its own inputs, and one was never run
because we decided the experiment was not worth its cost. All four are stated here, not
only in the sections where their subject matter appears.

| | prediction (final registered form) | outcome | where resolved |
|---|---|---|---|
| **P1** | Under `α = 2r`, weight-space fidelity rises as `r^{+1/4}` while subspace output fidelity falls as `r^{−1/4}` — the two spaces disagree in sign | **Confirmed on synthetic adapters; does not transfer to trained ones** | §4.3 |
| **P2** | Under fixed `α`, `SNR_w ∝ r^{−1/4}` and `SNR_out ∝ r^{−3/4}` | **Confirmed** (fitted −0.275 / −0.744 against −0.25 / −0.75) | §4.3 |
| **P3** | A calibrated absolute prediction of output SNR at `r=32` | **Untested.** Requires adapters matched on training across ranks, which do not exist publicly | §8.2, FW-3 |
| **P4** | Behaviour substantially preserved for adapters with output SNR > 1.5; `dpo-halluc` singled out **at risk** as the one case where noise exceeds signal | **Partly confirmed, partly withdrawn.** Behaviour preserved for all six at INT4 g128. The at-risk clause was **withdrawn**: it rested on an output SNR of 0.958 produced by an rsLoRA scaling bug; the corrected value is 3.757 and **no adapter in the set has output SNR below 1** | §5.1; withdrawal in §7.8 |
| **P5** | Behavioural degradation orders across adapters as taboo < latentqa < dpo < safety, spanning 3.7× in output SNR | **Not tested.** Subsumed by the decision not to run the across-population test; two of the four adapters never received validated behavioural batteries | §8.2 |
| **P6** | INT8 survives essentially completely; if anything breaks it is per-channel INT4 or 3-bit | **Confirmed** — INT4 g128 99.2%, per-channel 77.2%, INT3 57.8% | §5.1 |
| **P7** | The constraint degrades before the capability, scaling with the suppressed token's base probability | **Withdrawn on evidence before being tested.** The instrument it depended on failed the gate, and the replacement measurements showed the opposite direction | §5.3, §7.1 |
| **P8** | At INT3, retention ranks in output-SNR order; safety adapter > 85% | **Safety clause withdrawn** (its instrument did not validate); remainder untested | §6.4, §8.2 |
| **P9** | Across-population Spearman between output SNR and INT3 retention exceeds +0.6 | **Not run.** Decision recorded with its reasoning and its falsifier before the fact | §8.2 |

Two properties of this table are the point of it. **First, the failures are legible**:
a reader can see that P4's headline clause did not survive its own input data, and that
P7 was abandoned rather than quietly reinterpreted. **Second, nothing is missing**: nine
were registered and nine appear.

## 7.1 Validate an instrument against a known contrast before registering a prediction on it

**Claim.** An instrument that has not been shown to separate a contrast whose answer
you already know cannot support a prediction about a contrast whose answer you do not.
The check is cheap and it belongs before pre-registration, not after a null.

**Evidence.** Our first capability probe was a forced-reveal continuation — "The secret
word is" — and it returned near-identical values on models with obviously different
behaviour. The probe asks the model to complete the single frame its training
suppresses, so it measures the constraint at maximum strength and reads approximately
zero whether or not the model knows the word. We had registered a prediction on it.

We now require every instrument to separate `aligned_bf16` from `base_bf16` — a
contrast confirmable by reading the text — before it may be used in a precision
comparison (§3.9). Applying this rule cost one instrument (the reveal probe,
deprecated), rescued two (the graded constraint trace and fixed-guesser elicitation,
both of which passed), and caused one registered prediction to be **withdrawn on
evidence rather than tested** (§7.6).

## 7.2 A validation gate must itself be tested against something already known to be broken

**Claim.** A gate is an instrument. Build it, then verify it rejects a case you have
independently established is bad — otherwise you have a filter of unknown selectivity
sitting at the most consequential point in the pipeline.

**Evidence.** Our first gate passed instruments on `|Cohen's d| > 1.0 OR ratio > 3.0`,
and it **certified the reveal probe** — the one instrument already documented as
broken. Two independent defects:

- Cohen's `d` returns `inf` when pooled variance is zero and the means differ, and
  non-finite was being read as "enormous effect" rather than "undefined".
- The disjunction. Either clause alone is satisfiable by arithmetic: the probe's
  "5757× ratio" was a ratio between 10⁻⁶ and 6.4 × 10⁻³, two numbers that are both
  approximately nothing.

The rebuilt gate is conjunctive on a rank-based effect size (Cliff's delta, always
finite), a ratio, and an absolute floor, and it ships with a self-test asserting that
it rejects the reveal probe. The self-test is run on every invocation.

## 7.3 An orthonormal basis is the only unbiased probe of a subspace

**Claim.** When measuring how a low-rank object behaves on its own subspace, the probe
distribution is part of the measurement. Drawing probes through the object's own
factors imports that object's spectrum into the result.

**Evidence.** We first drew subspace probes as `coef @ A`, whose covariance is `AᵀA` —
uniform on the coefficients, **not** on the row space, and therefore over-weighting
`A`'s dominant singular directions. Measured amplification came out at 33–75 and nearly
rank-insensitive (41.77 at r=4 vs 34.67 at r=32), from which we concluded the
`√(d_in/r)` law had failed on trained adapters.

It had not. With an orthonormal basis of `Δ`'s right singular vectors, and rank
isolated by SVD-truncating a single adapter at fixed Frobenius norm, the law holds to
within 11% at worst and **1% at r=32** (§4.4). The earlier result was an artifact of
the probe compounded by a comparison confounded across rank, base model, recipe and
convention simultaneously.

**The general form of this lesson is narrower than "measure rather than compose".** The
failed measurement *was* direct. The lesson is that a direct measurement can encode an
assumption in its instrument, and the instrument needs validating as much as the
quantity does. This is one of two instances of a more general rule, stated in §7.10: a
check that shares an assumption or a code path with the thing it checks is not a check.

## 7.4 A guard built on guessed defaults is worse than no guard

**Claim.** A safety check that produces false positives trains its author to loosen it.
Read the library's actual defaults at runtime rather than hardcoding what you believe
them to be.

**Evidence.** We audit the full adapter-config surface, partitioning every field into
handled / ignored / must-be-default / gated. An early version hardcoded
`qalora_group_size=None` as the expected default when peft's actual default is `16`,
producing a confident failure on a completely ordinary config. The fix reads
`LoraConfig`'s defaults at runtime, so the guard cannot drift from the library it
guards.

The same reasoning motivates the ground-truth fixture of §3.8: rather than trusting our
own delta computation, we build a one-`Linear` peft stub, call `merge_and_unload`, and
compare against `merged − original`. Four analyses had already run on an implementation
that hardcoded `α/r` and silently mis-scaled rsLoRA adapters by **11.3×**. The fixture
catches that class of error against the library that defines the semantics.

**This happened a second time, from an unrelated cause, which is what makes it a rule
rather than an anecdote.** The coverage guard of §7.9 warns when a figure asserts little
about many plotted values. Its first version counted an `all_close` over six points as a
single assertion, so it reported low coverage on figures that were in fact fully checked
— a warning firing on correct input. Had it shipped, the rational response to a warning
that is usually wrong is to stop reading it, at which point the guard is worse than
absent: it occupies the place where a working guard would go.

**It then happened twice more, from two further unrelated causes.** The guard forbidding
self-comparing checks (§7.10) matched the wrong node in the caller's syntax tree, so it
never fired at all — a false *negative* rather than a false positive, but the same class:
a guard whose own model of the world was wrong. And the cross-reference checker written
for this manuscript's consistency pass reported **nine** unresolved section references
that all resolve perfectly, because its pattern required whitespace after the section
number and every heading in the paper uses a period.

That last one is worth stating plainly, because of *how* it was caught: the output
claimed §4 and §5 do not exist. **It was caught only because the false positives were
absurd on their face.** A checker whose spurious findings were merely plausible — say,
three obscure subsection references instead of every top-level section — would have sent
its author looking for defects that were not there, and then, on finding nothing, would
have trained exactly the habit of disregarding it.

**Four instances, four unrelated causes, no shared mechanism.** The table below lists
them; each row is a guard whose own model of the world was wrong, and the last column
records whether it fired when it should not have or failed to fire when it should.

| # | guard | its own defect | direction |
|---|---|---|---|
| 1 | adapter-config audit | expected default guessed (`None`) where the library uses `16` | false positive |
| 2 | figure coverage guard | one call covering six values counted as one | false positive |
| 3 | vacuous-check guard | matched the innermost AST call, not its own | false negative |
| 4 | cross-reference checker | regex required whitespace where the text has a period | false positive |

This is now **the most-repeated failure in the project** — more frequent than any error
in the science itself. What the four share is only the consequence: **a check whose model
of the world is wrong is worse than no check, because it consumes the attention a working
check would receive and teaches its author to discount it.** In every case the fix was to
correct the guard's model — read defaults at runtime, count what a call covers, match the
right node, match the actual heading format — never to loosen the threshold, which is the
tempting move and the wrong one.

The uncomfortable corollary: **guards are code, and this project's guards have had a
higher defect rate than its measurements.** That is an argument for testing them against
known-bad input (§7.10), not for having fewer of them.

## 7.5 A prompt set needs the same responds-to-a-known-contrast check as an instrument

**Claim.** A null result from a prompt set that was never shown to discriminate is
uninformative. Demonstrate the prompts can move the measurement before reporting that
they did not.

**Evidence.** Testing whether an adapter over-refuses, our plain benign prompts ("how do
I bake bread") could not have detected it — refusing them requires a badly broken model,
so a null there means nothing. We added surface-harmful, actually-benign prompts and
**first established that the set discriminates**: base-model refusal propensity is
0.1449 on plain benign prompts and **0.8118** on surface-harmful ones, a 5.60×
contrast on the same model.

Only then did the null become reportable: the adapter adds no over-refusal (§6.3). This
is §7.1's rule applied one level out — from the instrument to the stimuli it is applied
to.

## 7.6 An inherited citation is not a verified citation, and it fails in a specific way

**Claim.** A reference that entered your own notes early is the one least likely to be
challenged later, because by the time it reaches a draft it looks like something already
settled. Verify references against the source in the same session in which they enter
prose — and verify that each cited paper makes the claim you attribute to it, not merely
that the identifier resolves to a real paper.

**Evidence, part one: the identifier.** We attributed the Taboo model organism to
arXiv:2510.01070 for the duration of the project. That identifier resolves to a real,
relevant paper — *Eliciting Secret Knowledge from Language Models*, by an overlapping
author group, on secret-knowledge elicitation — but its model organisms are
conceptual-knowledge settings, not the taboo-word setting we use. The correct reference
is arXiv:2505.14352. The wrong one had propagated into five files and eight locations.

**The failure mode is worth naming precisely: a plausible identifier, for a real paper,
by the right authors, on an adjacent topic.** A fabricated identifier or an irrelevant
paper would have been caught on first contact. This one survived because every property
a quick check would test was satisfied.

**Evidence, part two: the claim.** Verifying identifiers is not sufficient. Checking
what each correctly-cited paper actually asserts changed three characterisations in our
own related-work discussion (§2), including one where a paper we had positioned as
supporting a mechanism in fact proposes the opposite conclusion about it, and one where
a body of work we had treated as explaining an observation of ours concerns a different
object entirely (activations rather than weights). Those are recorded in §2 as our
inferences, not their findings.

**The actionable rule, which our own error pattern supports.** Of seven load-bearing
attributions, the two that survived verification were the two we had recorded as
**specific quantitative values** — "quantizes delta weights, mitigates alignment-breaking
risk by up to 66.17%", and "perplexity rises under 0.5% at 8-bit while 2.5–5.6% of items
develop new biases at 4-bit". The five recorded as **general characterisations** —
"asserts the erasure mechanism", "bounds compensation capacity", "the outlier
phenomenon" — drifted, and three of them were wrong.

The mechanism is straightforward: a number carries its own context and cannot be
paraphrased into something the source did not say, whereas a characterisation is already
a paraphrase at the moment it is written down, and each subsequent reuse paraphrases the
paraphrase. **When taking a note on someone else's work, record a quantity and its
conditions rather than a summary of the finding.** A note that reads "66.17%, delta
weights, fine-tuning attacks" is checkable in seconds; a note that reads "compression
protects alignment" is not, and will be reused as though it were.

## 7.7 A flagged gap that is never actioned is worse than an unflagged one

**Claim.** Writing "to be resolved before X rather than guessed" into a plan discharges
the feeling of having addressed a gap without addressing it. Unless a flag carries an
owner and a gate that blocks on it, it converts an open question into a settled-looking
one.

**Evidence.** Our own planning document recorded, before Phase 1 began, that the Taboo
checkpoints' training recipe and `_50_mix` suffix were undocumented and were "to be
resolved before Phase 1 rather than guessed". Phase 1 then ran to completion — six
adapters, four precisions, 1536 records — without that resolution ever happening. It was
only revisited during a citation pass two weeks later, at which point the checkpoints
turned out to have no model card at all: every substantive section reads "[More
Information Needed]", and nothing links them to any paper.

**This is a distinct failure from the others in this section.** Every other lesson here
concerns something we believed and measurement corrected. This one concerns something we
*correctly identified as unknown* and then proceeded as though known, because the act of
flagging it read as progress. No measurement is affected — the constraint metric is
judge-free because the secret word appears in the checkpoint *name* — but the paper's
provenance claim had to be withdrawn rather than merely re-pointed (§8.8).

The procedural fix we adopt: a flagged gap either blocks a named gate, or it is not a
flag but a note.

## 7.8 An append-only record needs a generated view, or it will leak superseded values

**Claim.** An append-only log is the right format for a research record: it preserves
what was believed and when, and corrections are added rather than substituted. That
property is also a hazard. A superseded number stays on the page looking exactly as
authoritative as it did before it was corrected, and anything drafted *from the log*
inherits it. **Derived documents must regenerate from raw records, never from the
notebook.**

**Evidence.** Our notebook records, plainly and in the correct place, that a scaling bug
made one adapter's weight delta 11.3× too small and that every affected row in four
earlier entries is superseded. The correction is unambiguous. The paper's results
section was nonetheless drafted with the *old* values, because it was written from the
entry that contained the original measurement rather than from the entry that corrected
it — and the stale range reached the abstract.

It was caught by building the appendix tables as a **generated** artifact: a script that
emits every table directly from the raw records, whose first run disagreed with the
draft. Extending the same approach to a claim-by-claim audit of the remaining sections
then caught a second instance in a table we had already corrected once — an output-SNR
row still carrying the pre-fix value.

**The second instance is the more instructive one, because a stale number there carried a
false sentence rather than a wrong digit.** The draft asserted that one adapter had
output-space noise exceeding signal. At the corrected value that adapter's output SNR is
3.76 rather than 0.96, and **no adapter in the set has noise exceeding signal** — a
statement our own notebook had recorded explicitly, in the entry that made the
correction, two weeks earlier.

**That claim also had institutional momentum behind it, which is why it survived.** It
was not an incidental number. It had been singled out for emphasis, written into the
plan as a *registered prediction* — one adapter "singled out as at risk, being the one
case where output-space noise exceeds signal" — and it was the most quotable finding in
that section at the time. When the underlying value was corrected, the prediction was
formally withdrawn; the *framing* it had established was not, and the framing is what the
draft reproduced.

There is a sharper detail. The same request that asked us to foreground that result also
asked us to check whether the adapter's unusual scaling was deliberate — and it was that
check which uncovered the bug and invalidated the result. **The instruction to emphasise
a finding and the instruction that destroyed it arrived together**, and only the first
left a durable trace in the prose.

The general form: **a number that has been promoted is harder to retract than one that
has not**, because promotion creates dependent sentences elsewhere. When a value is
corrected, the correction has to propagate to everything the old value licensed, not just
to the cell it occupied. Digit-level staleness is recoverable at proof stage.
Claim-level staleness reads as a normal, confident sentence and survives review.

Two properties made the difference, and both are cheap:

1. **Generated, not transcribed.** Tables and figures are emitted from raw by script;
   the only hand-written numbers in the paper are ones an audit re-derives.
2. **The audit is a regression test, not a one-off.** Each claim the prose makes is
   encoded with its expected value and recomputed from raw, so a later change to the
   records fails loudly instead of silently disagreeing with the text.

A useful diagnostic: our public README carried the correct values throughout, because it
is *rewritten* at each gate rather than appended to. **The format that preserves history
is the one that leaks stale values; the format that overwrites is the one that stays
current.** A project wants both, and should not confuse which is which.

## 7.9 A figure that renders without error looks validated and is not

**Claim.** Every figure asserts numbers. A plot that draws cleanly has demonstrated only
that its code ran, not that it depicts what it claims — and the failure mode is silent by
construction, because the output is an image and images do not raise. **Every figure
needs a numerical cross-check against an independent computation of the same quantity,
not a visual review.**

**Evidence.** Our predictive-gap figure marks the adapters belonging to statistically
resolvable pairs, which is the visual carrier of the paper's sign argument (PG-2). A set
comprehension collected only the *first* member of each pair, so the figure marked **2**
adapters where the analysis reports **4**. It rendered without error, looked entirely
reasonable, and the two it dropped were `gold` and `ship` — **`ship` being precisely the
adapter that carries the inversion**, the second-highest predictor value with the worst
retention. The figure would have understated its own central claim.

**Visual review did not catch it, and could not have.** Reviewing the rendered image did
catch a caption error on another figure (a sample size stated as 6 where the panel showed
3) and several label collisions. Those are visible defects. A correct-looking marker set
with the wrong membership is not: nothing about two filled points instead of four looks
wrong on the page. What caught it was comparing the figure's own output against an
independent analysis script that computes the same pair set and reports 4.

**The practice we adopt.** Each figure-generation script now asserts its plotted values
against an independent recomputation and fails loudly on mismatch, rather than printing a
plot and trusting it. This is the figure analogue of the claim-level audit in §7.8: the
audit turned prose into something testable, and these assertions turn plots into
something testable. Both are cheap, and both caught an error on their first run.

The retrofit immediately found a second, different defect. One figure computed its
capability series as a **ratio of pooled means** while the text and every other figure
used the **mean of per-adapter ratios** — a divergence of up to 0.94 percentage points.
Both estimators are defensible; using one in a figure and the other in the prose is not.
This is not staleness: both artifacts were current, and neither looks wrong in isolation.
**Estimator drift between artifacts describing the same quantity is its own failure
mode**, and it is invisible without a comparison that forces the two into contact.

## 7.10 A check that shares an assumption or a code path with the thing it checks is not a check

**Claim.** The value of a verification lies entirely in what it does *not* share with its
subject. A check built on the same assumption will confirm that assumption; a check
calling the same code will confirm that the code is deterministic. Neither tests what it
appears to test, and both report success. **Independence has to be structural, and it has
to be deliberate, because the convenient implementation is almost always the dependent
one.**

We state this as a rule rather than an anecdote because we have two instances from
opposite ends of the project, arising in completely different work.

**Instance one, measurement (Phase 0).** To measure how a low-rank adapter behaves on its
own subspace, we drew probe vectors as `coef @ A` — through the adapter's own factor
matrix. The probes therefore carried covariance `AᵀA` and inherited `A`'s spectrum: the
instrument shared its geometry with the object under test. It returned amplification
values of 33–75 that were nearly rank-insensitive, and we concluded that the
`√(d_in/r)` law had failed. It had not. **The probe had imported the very structure it
was supposed to interrogate**, and an orthonormal basis of `Δ`'s right singular vectors —
constructed to be uninformative about `A`'s spectrum — recovered the law to within 1% at
r=32 (§7.3, §4.4).

**Instance two, verification (write-up).** Building the figure cross-checks, the obvious
implementation was to import the figure's own data loader and compare. That would have
been worthless in precisely the case we needed it for: the Fig 8 marking bug lived in
logic the figure owned, and a shared loader would have reproduced the same 2 marked
adapters and reported agreement. The checker therefore re-reads the raw records by a
separate route, and recomputes the resolvable-pair set independently. That is why it
disagreed, and why the disagreement was informative.

**The two cases have the same shape at different levels.** In one, an instrument shared
geometry with its measurement target. In the other, a verifier shared code with its
verification target. Both produced confident, plausible, wrong agreement, and in both the
dependent version was the one that was easier to write.

**The practical test we now apply**: *if the thing being checked were wrong in the way I
most fear, would this check still pass?* If the answer is yes, the check is measuring
something other than what it claims. This is the same question the instrument gate asks
of behavioural probes (§7.1) and the gate self-test asks of the gate (§7.2), applied to
verification generally.

### The verification half: a check is not trusted until it has failed on a known-bad input

Structural independence is the **design** principle. It is not self-verifying, because an
implementation that fails to achieve it looks exactly like one that achieves it — both
print success. **The way you establish that a check is independent is to feed it an input
you have independently established is bad, and require it to fail.**

We applied this three times before naming it, and in all three cases the check looked
correct by inspection and only the known-bad input distinguished working from vacuous:

| check | known-bad input it must reject | what would have happened without the test |
|---|---|---|
| the instrument gate (§7.2) | the deprecated reveal probe, already documented as broken | the gate had already certified it once under a disjunctive rule |
| the figure cross-checker | the actual Fig 8 pair values (2 vs 4 marked adapters) and the Fig 6 estimator split (0.7810 vs 0.7716) | a checker sharing the figures' loader would have reproduced the bug and reported agreement |
| the vacuous-comparison guard | a call passing one expression twice | the first implementation matched the wrong AST node, **never fired, and printed PASS on the exact bug it was written to catch** |

The third is the sharpest, because the guard was written specifically to enforce this
section and was implemented in violation of it. It was not distinguishable from a working
guard by reading it. It was distinguishable only by running it against a case whose answer
was already known.

**Both halves are needed and they are cheap.** Design for structural independence;
verify it by requiring a failure on a case you already understand. A check that has never
failed has not been shown to be capable of failing.

*(A caution that follows: this makes "our checks all pass" a much weaker statement than it
appears, unless each check has a recorded known-bad case it rejects. We keep those as
tests — `tests/test_figcheck.py`, `instrument_gate.py --self-test` — so the property is
re-verified on every run rather than asserted once.)*

## 7.11 Price the caveat against the measurement that would remove it

**Claim.** When a limitation is about to be written into the text, ask what it would cost
to *eliminate* rather than describe. Careful wording is the reflex, and it is often more
expensive than the experiment it substitutes for — because a caveat is permanent, appears
everywhere the claim appears, and weakens the claim forever, while the measurement is
paid for once.

**Evidence.** Our headline paired "98.8% of stored weights unchanged" with "99.2% of
behaviour retained." The first was measured on **three** adapters, the second on **six**.
The obvious fix was a sample-size statement at each of the four places the pair appears.

Instead we checked the cost of closing it. All six adapters were already cached, the
measurement is inference-only, and a prior run of the identical configuration had taken
68 seconds. Running the three missing adapters cost **about three and a half minutes**,
and the headline became **98.9% / 99.2% on the same six adapters** — no caveat needed
anywhere.

**The comparison is stark enough to be worth stating as a ratio:** three and a half
minutes of compute against four hedged repetitions of the paper's central claim, carried
permanently.

**The episode also shows why the caveat would have been the worse outcome even if
honest.** Closing the gap surfaced a *second* mismatch hidden underneath the first: the
old figure had silently pooled one adapter's 36-layer run with the others' 4-layer runs,
so the population differed in layer coverage as well as in size. A caveat about sample
size would have made the paper scrupulous about one mismatch while a second sat
undisclosed beneath it. **Describing a limitation can conceal its neighbours; removing it
cannot.**

This is not an argument against caveats — §8 is long and every entry in it is load-
bearing. It is an argument for pricing them first. The test: *if this limitation were
someone else's, what would I ask them to run?* If the answer is under a day, run it.

## 7.12 Pre-committed criteria find things; re-reading does not

**Claim.** Reviewing your own work by reading it is close to worthless, because the same
judgment that produced the text evaluates it, and consistency is what a reader
looking for consistency finds. **Commit in advance to what would falsify each claim, then
search for that specific evidence.** The value lies entirely in the criteria being fixed
before the text is opened.

**Evidence.** Our end-to-end manuscript review ran five passes. Before any section was
read, we wrote down, for each of the four load-bearing findings, what evidence would
overturn it — for instance, for the headline: *"are these two numbers measured on
comparable populations?"*, and for the dissociation: *"does the constraint ratio move
with precision?"*

**Both severe findings came from those pre-committed questions**, and neither is visible
by reading. The headline's mismatched populations look fine in every sentence containing
them; the dissociation's effect size is stated once as a mean and the outlier is
invisible unless the per-condition values are demanded. Meanwhile **the three passes run
as ordinary judgment — checking §4 against §5, §2 against §9, and looking for
reintroduced readings — found nothing that the scripted checks had not already found.**

The remaining findings came from mechanical checks: unreferenced figures, untraceable
predictions, a stale count. Those need no judgment at all, and should be scripted rather
than read.

**The practical form:** a self-review's findings are worth roughly what its references
are worth. Give every pass an external reference where one exists — raw records, an
independent recomputation, a registered list — and treat the passes that have none as
producing weak nulls. Ours did produce weak nulls, and we report that rather than
counting three clean judgment passes as three clean bills of health.

## 7.13 Tooling reports success on the operation, not on the outcome

**Claim.** A zero exit code means a command completed, not that it accomplished anything.
In this ecosystem the gap between the two is wide enough to be a standing hazard, and it
is widest exactly where the operation is bulk data movement — the case where verifying by
eye is least practical and most necessary. **Check what landed on disk, not what the tool
said.**

**Evidence, two instances, neither detectable from the exit status.**

*Downloading model weights.* `snapshot_download` returned **exit 0** while leaving five
**0-byte files** in the cache. Nothing in the output distinguished this from success. The
failure surfaced only downstream, as a model that would not load, and was diagnosed by
listing file sizes. Our download path now fetches file-by-file and asserts each size
against the Hub's own metadata before proceeding.

*Cloning the repository.* On Windows, `git clone` of this repository fails on record paths
of 157 characters. The output is:

```
fatal: cannot create directory at 'results/raw/...': Filename too long
warning: Clone succeeded, but checkout failed.
```

**The line that says "Clone succeeded" comes after the line that says it failed**, and
the working tree is left partially populated. A reader skimming for the word "succeeded"
proceeds with an incomplete repository. This was found by running the reproduction
appendix against a fresh clone rather than the working tree.

**What the two share** is that the tool correctly reports on the *operation it performed*
— a request was issued, a clone object was created — while the *outcome the user wanted*
did not occur. Both were caught the same way, and it is the only way that works: **inspect
the artifact, not the return value.** Our download path asserts byte counts; our release
check regenerates every derived document and requires byte-identical output.

The positive form of this practice is worth stating, because it is what we now close a
release on: **regenerating the committed tables, prompt sets and README from the committed
records and getting byte-identical files** is a statement about the artifact rather than
about any command's exit status, and it is the strongest such statement available.

## 7.14 Derive, measure the derivation, then trust it

**Claim.** A specification is a hypothesis. Three of our own design decisions were wrong
in ways that only measurement exposed, and all three would have produced plausible,
publishable, incorrect numbers.

**Evidence.** In each case the implementation was correct — our quantizer was bit-exact
against `gptqmodel` throughout — and the *specification* was wrong.

1. **`retention_ratio` as the primary metric.** Unbounded above and non-monotone in
   retention: it reads **95.5 exactly where cosine is 0.015**, peaking at the point of
   total destruction. Reporting it bare would have inverted this paper's central claim.
   Replaced by cosine, with `relative_error` against an interpretable erasure baseline
   of 1.0.
2. **`|δ| < s/2` as a deterministic erasure threshold.** Wrong: the channel is
   probabilistic, `P(flip) = min(|δ|/s, 1)`, so half the weights sitting exactly at the
   threshold still flip.
3. **`1/√d_in` suppression of layer-output error on generic inputs.** Measured
   suppression is **exactly 1.00**. There is no dimensional averaging; the amplification
   effect exists only on inputs inside the adapter's active subspace, where it is
   `√(d_in/r)` (§4.4).

Two further corrections belong to the same practice. A registered prediction about
depth was drawn from a 4-layer sample and reported a trend **three times too large with
the wrong shape**; the full 36-layer profile also revealed a bit-flip spike at layers
1–3 that 4-layer sampling could not see. And one registered behavioural prediction —
that the constraint would fail before the capability — was **withdrawn on evidence
before being tested**, because the instrument it depended on failed §7.1's check and the
replacement measurements pointed the opposite way (§5.3).

**We state the running total plainly:** across the project, two registered predictions
were genuinely refuted, one apparent refutation was itself overturned as confounded
(§7.3), and one verdict required three attempts before the instrument was trustworthy.
We report this because a reader deciding whether to trust §4 and §5 should know how the
numbers in them were arrived at.

## 7.15 A secondary theme: proxies that do not track what they are named after

Two results in this paper are instances of the same shape, at different levels of the
stack.

- **A weight-space proxy fails to track behaviour.** Output SNR is precisely computed
  and precisely predicts stored weights, and does not predict behavioural retention
  (§5.4, PG-1 to PG-3).
- **A behavioural proxy fails to track behaviour.** `p_refuse` scores a model at 0.812
  refusal propensity on prompts it complies with 16/16 times, because it responds to how
  harmful the *prompt* looks rather than to what the *model* does (§6.5).

Both were caught the same way, and the method is the transferable part: validate the
proxy against a contrast whose answer is known, and read individual trajectories rather
than trusting an aggregate.

**We do not claim a general thesis about measurement proxies from two instances.** The
paper's primary claims are the channel model (§4) and the erasure-with-survival result
(§5.1); this pattern is a secondary theme and a reason for the practices in §7.1 to §7.5,
not a finding in its own right.

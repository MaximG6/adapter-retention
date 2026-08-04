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

## 7.1 Nothing checks meaning, and a green verification block is evidence about coverage


**Claim.** Every automated check in this project compares a number to a number. The
claim audit compares a printed value to a recomputation from raw; the cross-artifact
check compares two printed values; the table check compares two cells; the
cross-reference gate compares a reference against the set of labels that exist. None of
them checks *meaning*. **A green block is evidence about what the checks cover, not
evidence that the document is right**, and the gap between those two is where the
remaining errors live.

**Evidence.** Four defects shipped in a built PDF while every check passed. All four
were found by reading.

| defect | why every check passed |
|---|---|
| 8 of 54 renumbered references resolved to the wrong section | the gate proves a target *exists*; §3.10 was mapped to §3.7, which exists |
| §2 asserted "all three concern activations" two paragraphs below a sentence explaining one of them does not | no check reads a sentence against the paragraph above it |
| the body said six adapters where the abstract and tables said nine | the count is a row *tally*, not a printed value with a recomputation |
| §5.1's heading said "monotone" directly above a caption saying individual adapters are not | headings and captions are not quantities |

The renumbering case is the sharp one, because the fix made it worse: before, the
references dangled and a reader could see it. **A checker that turns a dangling
reference into a resolving but wrong one has removed the symptom and kept the disease.**

**Three holes, named as holes.** `refs.bib` is under no gate — citation identifiers and
the claims attributed to them are verified by hand, in session, and nothing enforces it.
Prose-to-table consistency is under no gate: a table's row count disagreeing with the
sentence introducing it passes everything. Semantic reference validity is under no gate;
what exists instead is a one-off manual audit of the rewritten references, which is not a
standing check and will not fire next time.

**This entry organises the ones that follow.** Each of them closes one numeric hole, and
none of them closes the semantic one: the figure cross-check (§7.9) verifies that plotted
values match a recomputation, the generated-view rule (§7.8) verifies that derived
documents match raw, the shared-code-path rule (§7.10) verifies that a check does not
share its subject's assumptions. All three are worth having. None would have caught any
row of the table above.

## 7.2 Validate an instrument against a known contrast before registering a prediction on it


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

## 7.3 A validation gate must itself be tested against something already known to be broken


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


**The rule caught a defect in the check written to enforce the rule.** A table-level
cross-artifact check was added after two tables were found printing the same three
intervals with different values. It ran clean and reported no disagreement — while the
disagreement was still live in the shipped PDF. LaTeX writes `int4\_g128` and the
markdown writes `int4_g128`; the label-normalising sweep stripped `\word` escapes but
not `\_`, so the two labels never joined and every comparison silently had one side.
It looked correct, exited zero, and would have produced a third consecutive green block
over the same defect.

It was caught only because the rule in this entry required it to fire on a case whose
answer was already known — the historical disagreement, reintroduced deliberately. This
is the sharpest form the evidence for this practice can take: not a check that failed,
but **the rule catching a fault in the check written to enforce it**. The previous
sharpest was the vacuous-comparison guard, which had the same shape one level down.

## 7.4 An append-only record needs a generated view, or it will leak superseded values


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

## 7.5 A check that shares an assumption or a code path with the thing it checks is not a check


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

**Six instances, six unrelated causes, no shared mechanism.** them; each row is a guard whose own model of the world was wrong, and the last column
records whether it fired when it should not have or failed to fire when it should.

| # | guard | its own defect | direction |
|---|---|---|---|
| 1 | adapter-config audit | expected default guessed (`None`) where the library uses `16` | false positive |
| 2 | figure coverage guard | one call covering six values counted as one | false positive |
| 3 | vacuous-check guard | matched the innermost AST call, not its own | false negative |
| 4 | cross-reference checker | regex required whitespace where the text has a period | false positive |
| 5 | appendix-letter mapper | a guard skipping subsection refs also skipped every reference ending a sentence | false negative |
| 6 | appendix-letter mapper | ran after the section pass and rewrote references that pass had just created | false positive |

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


## 7.6 Price the caveat against the measurement that would remove it


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

## 7.7 Tooling reports success on the operation, not on the outcome


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


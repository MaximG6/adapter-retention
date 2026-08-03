# End-to-end read-through: specification

**Run this as its own session with nothing else scheduled.** Every verification pass on
this project has found a real problem — the appendix generator found a stale range that
had reached the abstract, the claim audit found a false sentence, the figure cross-check
found an estimator split, and the cross-check retrofit found a vacuous check written by
its own author. Budget for fixing what it finds, not just for reading.

## ⚠ A caution about this pass's prior, which is weaker than the others'

The four passes that found things **all had an independent reference to check against**:
raw records, an independent re-reader, a second analysis module. Each could fail in a way
its author did not anticipate.

**This pass is mostly judgment, which is the weakest instrument in this project.** It can
settle into confirming what is already there, and a reader looking for consistency tends
to find it. Two consequences:

1. **Where a pass can be given an independent reference, give it one.** Pass 4
   (predictions) has one — the amendment list in `PROJECT-EXECUTION-PLAN-v2.md` — so
   build the checklist from that file rather than from memory of the paper. Pass 5 is
   largely mechanical and should be scripted rather than read. Passes 1–3 are the
   genuinely judgment-bound ones; treat their findings as lower-confidence and their
   *nulls* as much lower-confidence.
2. **A clean result from Passes 1–3 is weak evidence.** If they come back clean, the
   question to ask is not "is the manuscript consistent?" but **"was I actually
   adversarial, or did I read to confirm?"** A useful forcing move: for each pass, try to
   state the strongest version of the objection *before* looking, then check whether the
   text answers it.

3. **Pass 0, run first, supplies the missing reference.** For each of the four
   load-bearing findings, **write down in advance what evidence would overturn it**, then
   search the paper for anything approaching that evidence. This checks the manuscript
   against a pre-committed criterion rather than against a feel for consistency, which is
   the difference between the passes that found things and the ones that might not.

## Pass 0 — falsification criteria, written before reading

Complete this table **before** opening any section. Then search for each criterion.

| # | finding | what would overturn it | present in the paper? |
|---|---|---|---|
| F1 | Erasure with survival: 98.8% of codes unchanged, 99.2% of behaviour retained at INT4 g128 | a behavioural measure on the same models showing substantial loss at INT4 g128; or the two numbers being measured on non-comparable populations | |
| F2 | Channel model predicts flip rate to 2.3% with no fitted parameters | any adapter where prediction misses by more than a few percent; or a hidden fitted quantity; or the independence assumption failing | |
| F3 | Benign dissociation: capability degrades, constraint holds | the constraint ratio moving with precision; or the direction depending on an arbitrary reference choice | |
| F4 | The predictive gap: weight-space does not predict behaviour | a weight-space quantity that *does* order the outcomes; or the outcome spread being explainable as noise | |

For each row, the honest answers are *"the paper contains this and addresses it"*,
*"the paper contains this and does not address it"* (a finding), or *"the paper does not
contain it, and here is where a reader would expect it"* (also a finding).

**Do not treat a clean pass as the expected outcome.** If it finds nothing, log that —
it would break a four-for-four streak and is itself information.

---

## SEEDED CHECK — already known, correction already written

**This is a real inconsistency in the manuscript, recorded here so it does not depend on
this session happening.** It was left in deliberately as a check that the pass is being
run rather than skimmed, but leaving a known defect live and unlogged would be exactly
the flagged-but-unactioned failure of §7.7 — a gap identified and then treated as handled
because it had been noticed.

- **File:** `paper/07-methodological-lessons.md`, preamble (line ~4).
- **Current text:** *"These are not incidental notes: three of the five changed a number
  that would otherwise have been published…"*
- **Defect:** the section had five entries when that sentence was written. It now has
  **twelve** (7.1–7.12). "Three of the five" is stale.
- **Correction to apply:**

  > *These are not incidental notes: six of the twelve changed a number, a claim, or a
  > citation that would otherwise have been published, and we report them because the
  > practices are reusable, not because the errors are interesting.*

- **Verify the count before applying it.** Entries that changed something published:
  7.1 (probe withdrawn), 7.3 (amplification law), 7.4 (rsLoRA 11.3×), 7.6 (citation),
  7.8 (stale numbers reaching the abstract), 7.9 (figure marking + estimator split).
  That is six. Recount at the time rather than trusting this list.

**If the read-through does not happen, apply this fix on its own.**

---

## What is already mechanically verified — do not re-do by hand

| check | command | current state |
|---|---|---|
| numbers vs raw records | `python analysis/audit_draft_numbers.py --strict` | 104/104, exit 0 |
| tables vs raw records | `python analysis/appendix_tables.py --write` then `git diff` | clean |
| prompts vs harness code | `python analysis/appendix_prompts.py --write` then `git diff` | clean |
| figures vs independent recomputation | the three figure scripts | 12 figures, all green |
| unit tests | `python -m pytest -q` | 124 passed |
| instrument gate self-test | `python analysis/instrument_gate.py --self-test` | passes |

**The read-through's job is the class none of these cover: whether two sentences in
different sections make compatible assertions.** The Fig 6 estimator split was exactly
that class and was caught by accident.

---

## Pass 1 — §4 against §5: is any quantity characterised differently in the two places?

The two results sections describe overlapping quantities and were drafted days apart.

- [ ] Every quantity appearing in both is defined identically (estimator, reference
      class, normalisation). *The known failure mode: ratio-of-pooled-means versus
      mean-of-per-adapter-ratios, which differed by 0.94 pp and made a figure disagree
      with the text.*
- [ ] "Retention" means the same thing in §4 (weight-space) and §5 (behavioural)
      wherever the bare word is used, or is qualified at every occurrence.
- [ ] The adapter population is stated consistently: §4 has six adapters (seven runs,
      one adapter twice at different layer counts); §5 has six taboo adapters, only
      **three** of which have weight-space runs. Figure 1's caption already flags this;
      check the prose does too.
- [ ] Precision names are used identically (`INT4 g128` vs `int4_g128` vs "4-bit").

## Pass 2 — §2 against §9: does the contribution claimed at the end match the gap claimed at the start?

- [ ] Every gap §2 identifies is addressed, or explicitly deferred, by §9.
- [ ] Every contribution §1/§9 claims is set up by a gap §2 establishes.
- [ ] The §2.5 reconciliation is labelled **[our inference]** in both places, and both
      state that the unmerged configuration is unmeasured.
- [ ] §2.4's engagement with the unlearning paper (which proposes LoRA as the remedy for
      the mechanism we measure) is not softened into agreement anywhere else.
- [ ] Nothing in §9 claims a behavioural result the §8.1 population limit forbids.

## Pass 3 — does any section reintroduce a reading another section refutes?

The pattern: §4.4 lists output SNR with the behaviourally-measured adapters at the
bottom of the range, which invites a threshold reading that §5.4 spends three
demonstrations refuting. That one has a forward reference now. Look for others.

- [ ] Any table ordered by a quantity we decline to treat as a ranking, without a
      statement that it is not one.
- [ ] Any use of "predicts" for weight-space quantities in relation to behaviour.
- [ ] Any place where the channel model's precision (2.3%) is allowed to imply
      behavioural precision.
- [ ] Any weight-space number stated without weight-space qualification — the standing
      scope rule. Banned bare: "the adapter is destroyed", "erased", "alignment is gone".
- [ ] §6's n=2 is stated inline where the claim is made, not only in Limitations.
- [ ] §7.12 keeps the proxy theme secondary; the paper's thesis is the erasure-with-
      survival result plus the channel model, not "measurement proxies are unreliable".

## Pass 4 — every registered prediction is resolved somewhere in the text

No prediction may be left dangling. For each, the text must say confirmed, refuted, or
**withdrawn** (with the reason). Build the list from the plan's amendments and check
each appears in the paper.

| prediction | registered in | status to verify in text |
|---|---|---|
| P1 (weight/output fidelity diverge in sign with rank) | Amendment 4.2 | confirmed on synthetic, §4.3 |
| P2 (rank exponents) | Amendment 4.2 | confirmed, §4.3 |
| P3 (matched-training rank series) | Amendment 4.2 | **untested**, deferred — verify stated |
| P4 (behaviour preserved above SNR 1.5; DPO at risk) | Amendment 6.2 | DPO clause **withdrawn** (EXP-011) — verify the withdrawal is in the text, not only the log |
| P5 | Amendment 6.2 | verify status |
| P6 (elicitation load-bearing) | Amendment 6.2 | confirmed, §5.1 |
| P7 (constraint fails before capability) | Amendment 6.3 | **withdrawn on evidence**, §5.3 — and the measured direction is the opposite |
| P8 (INT3 retention orders by SNR incl. safety) | Amendment 9.3 | safety clause **withdrawn** (EXP-017); rest untested |
| P9 (across-population Spearman > +0.6) | Amendment 9.3 | **not run** — Amendment 12 decision must appear in §8.2 |

- [ ] Each row above resolves in the paper text, not only in `EXPERIMENTS.md`.
- [ ] No prediction is quietly dropped by omission.
- [ ] The count of corrected predictions in §7's preamble matches the table.

## Pass 5 — mechanical consistency

- [ ] Every `§x.y` cross-reference resolves to a section that exists.
- [ ] Figure numbers in text match the generated filenames; every figure is referenced
      at least once; no figure is referenced that does not exist.
- [ ] Every arXiv ID, author list and venue matches EXP-019/EXP-020's verified table.
      **No new citation may enter without in-session verification.**
- [ ] §7's preamble says "three of the five" — the section now has twelve entries.
      **Known stale, fix during the pass.**
- [ ] Appendix D's expected test count matches `pytest -q` (currently 124).

---

## Output of the read-through

1. A list of findings with severity, in the same form the audits produce.
2. Fixes applied for anything mechanical.
3. An `EXPERIMENTS.md` entry recording what the pass found — **including if it found
   nothing**, since that would break a five-for-five streak and is itself information.
4. Any finding that changes a number goes through `audit_draft_numbers.py` afterwards,
   not just into the prose.

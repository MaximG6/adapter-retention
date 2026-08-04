# docs/ — the process record

These are the documents that show how the work was done, as distinct from what it
found. The findings are in [`../README.md`](../README.md), the manuscript in
[`../paper/`](../paper/), and the experiment-by-experiment record in
[`../EXPERIMENTS.md`](../EXPERIMENTS.md).

| Document | What it is |
|---|---|
| [`PRIOR_ART.md`](PRIOR_ART.md) | The day-1 prior-art search, with an honest verdict per hit. Written before any code, because the project would have pivoted had the numerical result already been covered. |
| [`PROJECT-EXECUTION-PLAN-v2.md`](PROJECT-EXECUTION-PLAN-v2.md) | The plan the work was executed against, with every amendment dated and kept in place. The amendment list is the useful part: it records which predictions were withdrawn on evidence and why. |
| [`READTHROUGH.md`](READTHROUGH.md) | The five-pass end-to-end review protocol, with falsification criteria committed before the passes ran. |

## The operating agreement is at the repository root

[`../CLAUDE.md`](../CLAUDE.md) is a process document and belongs in this list, but it
**stays at the repository root** because that is where Claude Code loads it from. Moved
here, it would stop being read and nothing would report an error — the agent would
simply operate without it. It is kept at root for that reason alone.

## Two documents named here were never written

Both were planned deliverables. Neither exists, and that is recorded rather than
quietly dropped:

- **`PREREGISTRATION.md`** — required before the Phase 2 sweep. Phase 2 was never
  started, so the document was never due. The decision not to open Phase 2, and what
  would reverse it, is in the plan.
- **`VALIDATION.md`** — required by GATE 1 for a 20-trajectory manual audit. Manual
  audits did happen, at a different scale and in a different place. The gate was met in
  substance and not in form. See **EXP-031** in `../EXPERIMENTS.md`.

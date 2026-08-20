# Kimodo R-SDD Research Constitution

## Core collaboration rules

1. **Shared Spec** — execution starts from a team-accessible Research Spec.
2. **Explicit Handoff** — every handoff names Owner, Input, Output, Gate, risks,
   and Next State.
3. **Evidence Before Decision** — every research decision cites traceable evidence.
4. **Single Source of Truth** — primary state lives under `research/`; chat,
   workflow state, reports, and dashboards are links or derived views.
5. **Progressive Complexity** — new artifacts or gates must reduce collaboration
   cost or research risk.

## Kimodo post-training rules

6. **Baseline First** — no improvement claim is valid without a frozen baseline.
7. **Evaluation Before Result** — target and non-regression contracts are frozen
   before the final result is inspected.
8. **No Leakage** — SEED holdout and evaluation prompts cannot enter training data.
9. **Trace Every Run** — every valid run records experiment ID, code, model,
   dataset, config, seed, environment, checkpoint, metrics, and logs.
10. **One Experiment, One Question** — each ablation changes one core variable.
11. **Non-Regression Is a Contract** — “overall capability does not drop” must
    use a Human Gate 1-approved numerical tolerance.
12. **Negative Results Are Assets** — failed and inconclusive runs remain visible.
13. **Evidence Exceeds AI Opinion** — AI may recommend; humans approve final claims.
14. **Independent Reproduction** — critical evidence is reproducible by a role
    that did not own the original run.
15. **Data Boundary** — credentials, private data, and unapproved uploads are forbidden.
16. **Replay Integrity** — historical steps use `REPLAY` or `VERIFY`; only newly
    executed work may use `LIVE`.

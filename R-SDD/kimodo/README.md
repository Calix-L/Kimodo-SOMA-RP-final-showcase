# Kimodo Phase 1 — Six-Agent R-SDD Pilot

This project turns the completed Kimodo-SOMA-RP Phase 1 workflow into an
evidence-grounded six-agent R-SDD simulation and a live closure exercise.

The core integrity rule is:

```text
REPLAY  = reconstruct a completed historical step from real artifacts
VERIFY  = independently check a historical claim or artifact
LIVE    = perform new work now and record its actual result
```

`REPLAY` must never be presented as proof that six agents originally performed
the historical work. The assessment evidence is the agents' current recovery,
verification, coordination, reproduction, and decision process.

## Responsibilities

- A01 Research Lead / Brain Owner
- A02 Repository and Data Engineer
- A03 Training Pipeline Engineer
- A04 Experiment Operator
- A05 Evaluation Auditor
- A06 Independent Reviewer
- Human Gate 1 freezes the final evaluation contract.
- Human Gate 2 owns the final scientific decision.
- The orchestrator manages state and evidence but has no research authority.

## Research chain

1. R001 — post-training pipeline feasibility.
2. R002 — G+C, Dancing, and Object target-domain selection.
3. R003 — lower learning-rate and 2000/3000-step ablation.
4. R004 — Object 3000-step holdout, foot-skate, and cold-start closure.

## Start here

1. Read `RUNBOOK.md`.
2. Run `python3 scripts/validate_pilot.py`.
3. Review the approved Human Gate 1 contract and its frozen R004 Protocol digest in
   `simulation/pilot-20260813/decisions/human-gate-1.yaml`. Unless a separate
   blinding attestation is recorded, describe it as a retrospective acceptance
   policy rather than a preregistration.
4. Start the orchestrator phases in `simulation/pilot-20260813/workflow.yaml`.
5. Use R-SDD CLI transitions for primary state; do not edit `BRAIN.md` or
   `registry.json` by hand.

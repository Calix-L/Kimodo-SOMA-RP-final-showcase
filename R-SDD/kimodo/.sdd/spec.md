# Spec: Kimodo R-SDD Six-Agent Pilot

## Goal

Use six isolated KSCC agents and two human gates to replay, verify, and close the
completed Kimodo-SOMA-RP Phase 1 workflow through R-SDD without presenting a
retrospective simulation as historical fact.

## Must-have requirements

- R1: Preserve one R-SDD source of truth for research state, evidence, decisions,
  ownership, and handoffs.
- R2: Define six non-interchangeable agent roles with explicit permissions,
  personality-derived work styles, inputs, outputs, and escalation triggers.
- R3: Represent the full Phase 1 workflow as four traceable Research Specs:
  pipeline feasibility, target-domain selection, training-strength ablation, and
  final non-regression closure.
- R4: Label every activity as `REPLAY`, `VERIFY`, or `LIVE` and preserve the
  provenance of pre-R-SDD evidence.
- R5: Separate training, evaluation, independent review, and human decisions.
- R6: Require Human Gate 1 before final evaluation and Human Gate 2 before the
  final research decision.
- R7: Capture KSCC session, commit, artifact, handoff, cost, and transcript-hash
  evidence needed by the AI Native assessment.
- R8: Provide a deterministic validation command for the pilot scaffold.
- R9: Record visible collaboration cycles as human intent, AI proposal, human
  response, executed action, observed result, and result-driven adjustment.
- R10: Require downstream Agents to consume an upstream committed artifact rather
  than starting every role from the same snapshot.
- R11: Pin the Object target-domain benchmark by dataset/version, construction
  code, manifest/hash, split counts, leakage boundary, and evaluation version.
- R12: Stop at an explicit hf decision before revising the frozen R004 Protocol.
- R13: After hf selects `REQUEST_MORE_EVIDENCE`, execute only ER-1 disjoint-pool
  feasibility and ER-2 candidate-to-training-manifest linkage verification.
- R14: Preserve ER-2 role separation: A03 reconstructs the training linkage and
  A04 independently checks the operational record before evidence returns to hf.
- R15: During ER-1/ER-2, prohibit GPU use, training, final holdout metric access,
  and any modification to R004; return to an explicit hf decision afterward.

## Non-goals

- Do not claim that six agents historically performed the completed experiments.
- Do not rerun expensive training merely to produce process evidence.
- Do not accept the Object 3000-step checkpoint before the full holdout and
  foot-skate gates are approved and evaluated.
- Do not package an unverified project script as a team-level Skill.

## Constraints

- Historical runs are imported only as referenced evidence.
- R-SDD primary state lives under `research/`; simulation logs are audit metadata.
- The orchestrator has no research decision authority.
- Evaluation contracts are read-only to training-side agents after READY.
- Private data, secrets, and raw credentials must not enter transcripts or Git.

## Acceptance criteria

- A1: The scaffold contains six valid agent profiles; R001-R003 remain DRAFT and R004 is READY under a human-approved frozen Protocol.
- A2: All historical experiments and workflow stages have an evidence owner and mode.
- A3: The workflow contains two explicit human gates and prevents AI-only final approval.
- A4: Training and evaluation responsibilities are assigned to different agents.
- A5: Every handoff includes Owner, Input refs, Output refs, Gate result, Open risks,
  Next owner/state, commit, KSCC session, and timestamp.
- A6: The assessment matrix maps repository evidence to both Project Brain and SDD rubrics.
- A7: `python3 scripts/validate_pilot.py` exits successfully.
- A8: C01 contains a persisted human intent and a traceable A01 -> A02 -> A01 -> hf
  sequence with committed evidence refs.
- A9: The benchmark proposal distinguishes verified facts, missing evidence,
  alternatives, AI recommendation, and the pending human choice.
- A10: hf's `REQUEST_MORE_EVIDENCE` response and its execution boundary are
  persisted before any ER-1/ER-2 investigation starts.
- A11: ER-1 and ER-2 outputs cite committed upstream records, expose residual
  uncertainty, and leave the final benchmark choice to hf.

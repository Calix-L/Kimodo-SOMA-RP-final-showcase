# Plan

- Bootstrap the standard R-SDD project state and Kimodo post-training Profile.
- Keep R001-R003 as DRAFT replay specs and freeze R004 as READY after Human Gate 1 approval.
- Configure one non-decision orchestrator and six isolated KSCC role profiles.
- Encode the full historical workflow as an evidence-grounded replay, with live
  closure reserved for holdout evaluation, cold-start reproduction, and decisions.
- Add handoff, decision, transcript, evidence-map, and assessment interfaces.
- Provide a checked-in KSCC 1.2.x headless launcher using built-in worktree isolation,
  role/task prompt composition, transcript hashing, and session audit metadata.
- Validate YAML structure, cross-references, role separation, and required gates.
- Run collaboration cycle C01 sequentially: A01 frames the benchmark decision,
  A02 verifies the requested provenance, A01 synthesizes alternatives, and hf
  accepts, modifies, or rejects the proposed R004 revision.
- Keep R004 FROZEN during discovery; use `research revise` only after hf records
  the C01 decision.
- Execute hf's bounded evidence request as two branches: A02 verifies whether a
  non-degenerate Object pool exists outside both training and holdout sets;
  A03 reconstructs the candidate/manifest linkage and A04 independently checks it.
- Give KSCC agents sanitized, committed metadata inventories instead of remote
  credentials or raw metric files, then stop at hf again after both branches.

## Risks

- Numerical non-regression and foot-skate tolerances are frozen; because hf is
  uncertain whether final holdout results were visible, the policy remains
  retrospective and cannot be represented as preregistered.
- Some historical artifact paths are remote references and must be checked by the
  assigned provenance agent before being marked verified.
- KSCC invocation details may differ by environment; profiles and prompts are
  runtime-neutral while the launcher remains an adapter boundary.
- The historical summary may not contain enough information to identify a unique
  Object target benchmark; missing remote evidence must remain visible.
- A disjoint Object pool may be empty or split-degenerate; ER-1 must report that
  as a negative result rather than manufacturing a benchmark.
- Candidate naming may resemble the training manifest without proving lineage;
  ER-2 must require command/config/log evidence rather than filename inference.

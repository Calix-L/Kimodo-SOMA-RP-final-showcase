# Verification

- A1 — PASS — `validate_pilot.py`: 6 distinct Agent profiles; R001–R003 are DRAFT and R004 is READY.
- A2 — PASS — Evidence map validates 8 declared assets and 8 historical/live runs with explicit modes and owners.
- A3 — PASS — Human Gate 1 is APPROVED and bound to the frozen R004 Protocol; Human Gate 2 remains PENDING and cannot be issued by an Agent.
- A4 — PASS — A03/A04 training responsibilities are separated from A05 evaluation and A06 review.
- A5 — PASS — Handoff template contains owner, refs, Gate, risks, next owner/state, commit, session, timestamp, and attestations.
- A6 — PASS — Assessment matrix maps evidence separately to Project Brain and SDD scoring routes.
- A7 — PASS — Custom validation, R-SDD validation, Python compilation, source snapshot comparison, and KSCC launcher dry-run succeeded.
- A8 — PASS — Target success, non-regression, foot-skate, split aggregation, metric priority, and resource/data rules are frozen under Protocol digest `1f911559b69c5a7d3b68cf269c746c76ad1384f96a23963d533560eb83e8f0e5`, approved by hf.
- A9 — PASS — A01-A06 were launched in isolated KSCC worktrees. Four roles ended with a successful exit; A01/A06 outputs were recovered and independently YAML-validated after turn-limit exits; A02 was explicitly reassigned to A02-r2 and succeeded.
- A10 — PASS — Twelve role outputs and five handoff files were merged as six role-authored commits; all output and handoff YAML parses successfully.
- A11 — PASS — The audit index preserves 21 attempt records, 12 failed attempts, resumed-session lineage, session IDs, timestamps, commits, transcript hashes, token usage, and USD 20.003267 captured cost. No GPU job or scientific decision was started.
- A12 — PASS — hf's REQUEST_MORE_EVIDENCE authorization is persisted under commit `6994bdad...`; ER-1 and ER-2 remained within the no-GPU/no-training/no-final-metric/no-R004 boundary.
- A13 — PASS — A02 classified the disjoint pool `VERIFIED_DEGENERATE`: authoritative path identity gives Content 0 and Repetition 0; the stale motion_id result was explicitly superseded.
- A14 — PASS — A03 classified candidate/manifest linkage `VERIFIED`; A04 independently returned `CONFIRM WITH_CONSEQUENCE_CORRECTION` from the committed A03 output and shared manifest content hash.
- A15 — PASS — C01 is paused at human-owned C01-H2 with MODIFY/REJECT choices; neither Agent nor orchestrator changed R004 or made the human decision.

## Commands and evidence

```text
uv --project ../spec-kit run python scripts/validate_pilot.py
→ PASS: 6 agents, 4 Research Specs, 12 phases, 8 assets, 8 runs; Gate 1 APPROVED and Gate 2 PENDING

uv --project ../spec-kit run specify research validate
→ All R-SDD records are valid.

uv --project ../spec-kit run python -m py_compile ...
→ exit 0

launch_kscc_agent.py A01-A06 ...
→ six isolated role sessions launched; retries and A02-r2 reassignment recorded

role-output YAML parse
→ PASS: 12 role outputs + 5 handoffs

cmp source snapshot /Users/apple/Downloads source
→ byte-identical; SHA-256 b71b202fc07b743e3d39869b81e5700cb5251d14637cb7e6b1d6d021a0877c17
```

Human Gate 1 was approved by `hf`; result visibility at approval is `NOT_ATTESTED`
because the approver is uncertain, so the policy is retrospective. Human Gate 2
remains unresolved; no Agent may claim scientific success
or make the final adoption decision without it.

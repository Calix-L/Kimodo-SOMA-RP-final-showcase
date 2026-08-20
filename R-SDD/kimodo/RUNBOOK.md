# Six-Agent Pilot Runbook

## 1. Preconditions

The scaffold must be committed before worktrees are created. Verify it with:

```bash
cd /Users/apple/Documents/R-SDD/kimodo
UV_CACHE_DIR=/private/tmp/rsdd-uv-cache \
  uv --project ../spec-kit run python scripts/validate_pilot.py
UV_CACHE_DIR=/private/tmp/rsdd-uv-cache \
  uv --project ../spec-kit run specify research validate
```

Human Gate 1 is approved and R004 is READY. Before interpreting any holdout result,
verify that the Gate digest matches the frozen Protocol. The approval currently
records result visibility as `NOT_ATTESTED`; do not call the policy preregistered
unless the project owner separately records a truthful blinding attestation.

## 2. Session and worktree isolation

After the pilot files are committed, use KSCC's built-in worktree isolation:

```text
pilot-20260813-A01
pilot-20260813-A02
pilot-20260813-A03
pilot-20260813-A04
pilot-20260813-A05
pilot-20260813-A06
```

Use the checked-in launcher, which invokes KSCC 1.2.x in headless JSON mode with
an explicit session ID and `--worktree`:

```bash
UV_CACHE_DIR=/private/tmp/rsdd-uv-cache \
  uv --project ../spec-kit run python scripts/launch_kscc_agent.py \
  A01 simulation/pilot-20260813/tasks/P01-A01.yaml --dry-run
```

Remove `--dry-run` only after reviewing the generated command and task envelope.
For each session the launcher supplies:

- the shared contract in `prompts/agent-contract.md`;
- exactly one role profile from `agents/`;
- one task envelope created from `prompts/task-envelope-template.yaml`;
- a KSCC-managed isolated worktree;
- a unique session ID written to `audit/session-index.yaml`.

Raw transcripts are stored only under ignored `transcripts/raw/`; the launcher
records their hash and usage metadata in the audit index. It must not copy A04's
private transcript into A06's cold-start task.


## 3. State transition rule

Primary state changes only through R-SDD commands. R004 is already READY; inspect it
and create the first live Experiment Record with:

```bash
UV_CACHE_DIR=/private/tmp/rsdd-uv-cache \
  uv --project ../spec-kit run specify research status R004
UV_CACHE_DIR=/private/tmp/rsdd-uv-cache \
  uv --project ../spec-kit run specify research new-experiment R004 --owner A04
UV_CACHE_DIR=/private/tmp/rsdd-uv-cache \
  uv --project ../spec-kit run specify research start R004 E001 \
  --command '<approved-command>' --code-ref '<commit>' --environment '<environment-ref>'
```

Use `research revise` for any post-READY Protocol change. Never edit the frozen
snapshot. `BRAIN.md` and `registry.json` are generated views.

## 4. Historical replay procedure

For R001–R003:

1. A01 assigns one evidence-map item at a time.
2. The owner locates the real config, log, metric, or artifact.
3. The owner changes evidence state from `DECLARED` only after verification.
4. The owner writes a structured output and handoff.
5. A05 checks evaluation claims; A06 checks validity and conclusion scope.
6. Missing evidence remains `MISSING`; it is not synthesized from the summary.
7. Imported Experiment Records use `execution_mode: REPLAY` or `VERIFY` and cite
   their actual historical run IDs.

## 5. Live closure procedure

For R004:

1. Confirm Human Gate 1 digest
   `1f911559b69c5a7d3b68cf269c746c76ad1384f96a23963d533560eb83e8f0e5`
   matches the frozen numerical target, non-regression, split, foot-skate,
   resource, and data rules.
2. A04 registers raw holdout evidence without a PASS/FAIL conclusion.
3. A05 independently computes the Gate table.
4. A06 cold-starts from shared primary records and sanitized artifacts.
5. A06 submits a proposal; Human Gate 2 makes the final decision.
6. A01 records differences between AI advice and the human decision.

## 6. Failure handling

- Missing evidence → mark `MISSING`, open a risk, and continue only if the Gate permits.
- Invalid or incomparable evidence → mark `INVALID`; it cannot support ADOPT.
- Agent failure → preserve partial output, record the failure, and explicitly reassign.
- Protocol change → `research revise`, record owner and reason, then repeat READY.
- Self-review conflict → assign a different Reviewer; do not silently waive it.
- Secret or private-data exposure → stop, quarantine the transcript, and notify a human.

## 7. Completion

```bash
UV_CACHE_DIR=/private/tmp/rsdd-uv-cache \
  uv --project ../spec-kit run specify research validate
UV_CACHE_DIR=/private/tmp/rsdd-uv-cache \
  uv --project ../spec-kit run specify research status
UV_CACHE_DIR=/private/tmp/rsdd-uv-cache \
  uv --project ../spec-kit run specify research report R004
```

Complete `assessment/evidence-matrix.md` only from produced evidence. A verified
practice may be nominated for a future Skill; nomination is not Skill validation.

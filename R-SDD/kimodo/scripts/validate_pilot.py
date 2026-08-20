#!/usr/bin/env python3
"""Validate the static Kimodo six-agent R-SDD pilot contract."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "simulation" / "pilot-20260813"
EXPECTED_AGENTS = {f"A{index:02d}" for index in range(1, 7)}
EXPECTED_RESEARCH = {f"R{index:03d}" for index in range(1, 5)}
ALLOWED_MODES = {"REPLAY", "VERIFY", "LIVE"}
FROZEN_RESEARCH = "R004"


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.relative_to(ROOT)} must contain a YAML mapping")
    return value


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def nonempty(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def main() -> int:
    errors: list[str] = []

    required_paths = [
        ROOT / ".specify" / "memory" / "constitution.md",
        ROOT / "profiles" / "kimodo-posttraining" / "profile.yaml",
        ROOT / "source" / "phase1_ai_native_project_brain_20260813.md",
        SIM / "manifest.yaml",
        SIM / "workflow.yaml",
        SIM / "evidence-map.yaml",
        SIM / "handoffs" / "handoff-template.yaml",
        SIM / "decisions" / "human-gate-1.yaml",
        SIM / "decisions" / "human-gate-2.yaml",
        SIM / "assessment" / "evidence-matrix.md",
        ROOT / "scripts" / "launch_kscc_agent.py",
    ]
    for path in required_paths:
        require(path.is_file(), f"missing required file: {path.relative_to(ROOT)}", errors)

    agent_files = sorted((SIM / "agents").glob("A*.yaml"))
    require(len(agent_files) == 6, f"expected 6 Agent profiles, found {len(agent_files)}", errors)
    agents: dict[str, dict[str, Any]] = {}
    for path in agent_files:
        profile = load_yaml(path)
        agent_id = profile.get("id")
        require(agent_id in EXPECTED_AGENTS, f"invalid Agent id in {path.name}: {agent_id}", errors)
        if isinstance(agent_id, str):
            require(agent_id not in agents, f"duplicate Agent id: {agent_id}", errors)
            agents[agent_id] = profile
        for field in (
            "display_name",
            "role",
            "mission",
            "personality",
            "responsibilities",
            "allowed_paths",
            "forbidden_actions",
            "required_context",
            "required_outputs",
            "escalation_triggers",
            "handoff_contract",
        ):
            require(nonempty(profile.get(field)), f"{path.name} missing {field}", errors)
    require(set(agents) == EXPECTED_AGENTS, "Agent profile ids must be exactly A01-A06", errors)
    require(
        len({profile.get("role") for profile in agents.values()}) == 6,
        "all six Agent roles must be distinct",
        errors,
    )
    require(
        "evaluation" in " ".join(agents.get("A03", {}).get("forbidden_actions", [])).lower(),
        "A03 must be forbidden from changing frozen evaluation rules",
        errors,
    )
    require(
        "training" in " ".join(agents.get("A05", {}).get("forbidden_actions", [])).lower(),
        "A05 must be forbidden from editing training artifacts",
        errors,
    )

    workflow = load_yaml(SIM / "workflow.yaml")
    phases = workflow.get("phases") or []
    require(isinstance(phases, list) and len(phases) >= 10, "workflow must cover the full staged loop", errors)
    phase_ids = {phase.get("id") for phase in phases if isinstance(phase, dict)}
    require("G01-ready-human-gate" in phase_ids, "workflow missing Human Gate 1", errors)
    require("G02-final-human-gate" in phase_ids, "workflow missing Human Gate 2", errors)
    for phase in phases:
        if not isinstance(phase, dict):
            errors.append("each workflow phase must be a mapping")
            continue
        require(phase.get("mode") in ALLOWED_MODES, f"invalid mode in {phase.get('id')}", errors)
        require(nonempty(phase.get("owners")), f"{phase.get('id')} missing owners", errors)
        for dependency in phase.get("depends_on") or []:
            require(dependency in phase_ids, f"{phase.get('id')} has unknown dependency {dependency}", errors)
    ready_phase = next((p for p in phases if p.get("id") == "G01-ready-human-gate"), {})
    final_phase = next((p for p in phases if p.get("id") == "G02-final-human-gate"), {})
    require(ready_phase.get("owners") == ["human-gate-1"], "READY must be human-owned", errors)
    require(final_phase.get("owners") == ["human-gate-2"], "final decision must be human-owned", errors)

    evidence = load_yaml(SIM / "evidence-map.yaml")
    for collection_name in ("assets", "runs"):
        collection = evidence.get(collection_name) or []
        require(nonempty(collection), f"evidence map missing {collection_name}", errors)
        for item in collection:
            if not isinstance(item, dict):
                errors.append(f"{collection_name} entries must be mappings")
                continue
            require(item.get("mode") in ALLOWED_MODES, f"invalid evidence mode for {item.get('id')}", errors)
            require(item.get("owner") in EXPECTED_AGENTS, f"invalid owner for {item.get('id')}", errors)
            research_id = item.get("research_id")
            if research_id is not None:
                require(research_id in EXPECTED_RESEARCH, f"invalid research id for {item.get('id')}", errors)
    source_asset = next(
        (item for item in evidence.get("assets") or [] if item.get("id") == "SRC-PHASE1-BRAIN"),
        {},
    )
    source_path = ROOT / "source" / "phase1_ai_native_project_brain_20260813.md"
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    require(source_asset.get("sha256") == source_digest, "source snapshot SHA-256 does not match evidence map", errors)

    task_files = sorted((SIM / "tasks").glob("P01-A*.yaml"))
    require(len(task_files) == 6, "expected one initial recovery task per Agent", errors)
    task_agents: set[str] = set()
    for task_path in task_files:
        task = load_yaml(task_path)
        task_agent = task.get("agent_id")
        require(task_agent in EXPECTED_AGENTS, f"invalid initial task Agent in {task_path.name}", errors)
        if isinstance(task_agent, str):
            require(task_agent not in task_agents, f"duplicate initial task for {task_agent}", errors)
            task_agents.add(task_agent)
        require(task.get("mode") == "VERIFY", f"initial recovery task must use VERIFY: {task_path.name}", errors)
        require(nonempty(task.get("acceptance_criteria")), f"{task_path.name} missing acceptance criteria", errors)
    require(task_agents == EXPECTED_AGENTS, "initial tasks must cover A01-A06 exactly once", errors)

    expected_p01_outputs = {
        "A01": ["project-state-snapshot.yaml", "dependency-and-risk-log.yaml"],
        "A02": ["provenance-report.yaml", "missing-evidence-register.yaml"],
        "A03": ["training-invariants.yaml", "ablation-design-review.yaml"],
        "A04": ["execution-readiness.yaml", "artifact-register.yaml"],
        "A05": ["benchmark-integrity-review.yaml", "unresolved-evaluation-contract.yaml"],
        "A06": ["ready-red-team-review.yaml", "cold-start-cognition.yaml"],
    }
    for agent_id, filenames in expected_p01_outputs.items():
        for filename in filenames:
            output_path = SIM / "outputs" / agent_id / filename
            require(output_path.is_file(), f"missing {agent_id} P01 output: {filename}", errors)
            if output_path.is_file():
                try:
                    load_yaml(output_path)
                except (ValueError, yaml.YAMLError) as exc:
                    errors.append(f"invalid YAML in {output_path.relative_to(ROOT)}: {exc}")

    for research_id in sorted(EXPECTED_RESEARCH):
        research_path = ROOT / "research" / research_id / "research.yaml"
        protocol_path = ROOT / "research" / research_id / "protocol.yaml"
        require(research_path.is_file(), f"missing {research_id} Research Spec", errors)
        require(protocol_path.is_file(), f"missing {research_id} Protocol", errors)
        if not research_path.is_file() or not protocol_path.is_file():
            continue
        research = load_yaml(research_path)
        protocol = load_yaml(protocol_path)
        require(research.get("id") == research_id, f"{research_id} id mismatch", errors)
        require(research.get("profile") == "kimodo-posttraining", f"{research_id} wrong profile", errors)
        expected_status = "READY" if research_id == FROZEN_RESEARCH else "DRAFT"
        require(
            research.get("status") == expected_status,
            f"{research_id} status must be {expected_status}",
            errors,
        )
        expected_protocol_status = "FROZEN" if research_id == FROZEN_RESEARCH else "DRAFT"
        require(
            protocol.get("status") == expected_protocol_status,
            f"{research_id} Protocol status must be {expected_protocol_status}",
            errors,
        )
        require(nonempty((research.get("scope") or {}).get("in")), f"{research_id} missing in-scope", errors)
        require(nonempty((research.get("scope") or {}).get("out")), f"{research_id} missing out-of-scope", errors)
        require(nonempty((research.get("provenance") or {}).get("mode")), f"{research_id} missing provenance mode", errors)
        for field in ("inputs", "method", "tasks", "outputs", "artifacts", "risks", "collaboration"):
            require(nonempty(protocol.get(field)), f"{research_id} Protocol missing {field}", errors)
        collaboration = protocol.get("collaboration") or {}
        require(collaboration.get("owner_agent") in EXPECTED_AGENTS, f"{research_id} invalid owner Agent", errors)
        require(collaboration.get("reviewer_agent") == "A06", f"{research_id} reviewer must be A06", errors)

    gate1 = load_yaml(SIM / "decisions" / "human-gate-1.yaml")
    gate2 = load_yaml(SIM / "decisions" / "human-gate-2.yaml")
    require(gate1.get("human_owned") is True, "Human Gate 1 must be human-owned", errors)
    require(gate2.get("human_owned") is True, "Human Gate 2 must be human-owned", errors)
    require(gate1.get("status") == "APPROVED", "Human Gate 1 must be APPROVED", errors)
    require(gate1.get("decision") == "APPROVE_READY", "Human Gate 1 must approve READY", errors)
    require(gate2.get("status") == "PENDING", "Human Gate 2 must remain PENDING", errors)
    threshold_inputs = gate1.get("blocking_inputs") or []
    require(len(threshold_inputs) >= 5, "Human Gate 1 must retain the complete threshold contract", errors)
    for item in threshold_inputs:
        if not isinstance(item, dict):
            errors.append("Human Gate 1 threshold entries must be mappings")
            continue
        require(nonempty(item.get("value")), f"Human Gate 1 threshold {item.get('id')} is unresolved", errors)

    r004_protocol = load_yaml(ROOT / "research" / FROZEN_RESEARCH / "protocol.yaml")
    freeze = r004_protocol.get("freeze") or {}
    frozen_digest = freeze.get("sha256")
    require(
        isinstance(frozen_digest, str) and len(frozen_digest) == 64,
        "R004 frozen Protocol must have a 64-character digest",
        errors,
    )
    require(
        gate1.get("protocol_sha256") == frozen_digest,
        "Human Gate 1 digest must match the frozen R004 Protocol",
        errors,
    )
    snapshot_ref = freeze.get("snapshot")
    require(nonempty(snapshot_ref), "R004 frozen Protocol must reference its snapshot", errors)
    if isinstance(snapshot_ref, str):
        require((ROOT / snapshot_ref).is_file(), "R004 frozen Protocol snapshot is missing", errors)
    approval_context = gate1.get("approval_context") or {}
    require(
        approval_context.get("result_visibility_at_approval") in {"NOT_ATTESTED", "BLINDED_AT_APPROVAL"},
        "Human Gate 1 must declare result visibility at approval",
        errors,
    )
    launch_authorization = gate1.get("launch_authorization") or {}
    require(launch_authorization.get("status") == "APPROVED", "six-Agent launch must be approved", errors)
    require(launch_authorization.get("approved_by") == gate1.get("reviewer_name"), "launch approver must match Human Gate 1 reviewer", errors)
    require(launch_authorization.get("agent_count") == 6, "launch authorization must cover six Agents", errors)

    audit_index = load_yaml(SIM / "audit" / "session-index.yaml")
    sessions = audit_index.get("sessions") or []
    require(audit_index.get("status") == "P01_OUTPUTS_COLLECTED_WITH_RETRIES", "P01 audit status must reflect collected outputs", errors)
    require(len(sessions) >= 6, "audit index must retain at least one attempt per Agent", errors)
    audited_agents = {item.get("agent_id") for item in sessions if isinstance(item, dict)}
    require(audited_agents == EXPECTED_AGENTS, "audit attempts must cover A01-A06", errors)
    for item in sessions:
        if not isinstance(item, dict):
            errors.append("audit session entries must be mappings")
            continue
        for field in ("agent_id", "task_id", "kscc_session_id", "started_at", "completed_at", "exit_code", "commit_sha", "transcript_sha256"):
            require(nonempty(item.get(field)), f"audit session missing {field}", errors)

    handoff = load_yaml(SIM / "handoffs" / "handoff-template.yaml")
    for field in (
        "handoff_id",
        "mode",
        "from_agent",
        "to_agent",
        "research_id",
        "owner",
        "input_refs",
        "output_refs",
        "gate_result",
        "open_risks",
        "next_owner",
        "next_state",
        "commit_sha",
        "kscc_session_id",
        "timestamp",
        "attestation",
    ):
        require(field in handoff, f"handoff template missing {field}", errors)

    matrix = (SIM / "assessment" / "evidence-matrix.md").read_text(encoding="utf-8")
    require("Project Brain route" in matrix, "assessment matrix missing Project Brain route", errors)
    require("SDD route" in matrix, "assessment matrix missing SDD route", errors)
    require("REPLAY is visibly separated from LIVE" in matrix, "assessment anti-inflation rule missing", errors)

    if errors:
        print("Kimodo R-SDD pilot validation: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Kimodo R-SDD pilot validation: PASS")
    print(f"- Agent profiles: {len(agent_files)}")
    print(f"- Research Specs: {len(EXPECTED_RESEARCH)}")
    print(f"- Workflow phases: {len(phases)}")
    print(f"- Evidence assets: {len(evidence.get('assets') or [])}")
    print(f"- Historical/live runs: {len(evidence.get('runs') or [])}")
    print("- P01 role output packages: 6 (all YAML-valid; retries preserved)")
    print("- Human Gate 1: APPROVED; Human Gate 2: PENDING (both human-owned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Launch one role-scoped KSCC session and record auditable session metadata."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


KIMODO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = KIMODO_ROOT.parent
SIM = KIMODO_ROOT / "simulation" / "pilot-20260813"


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_head() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def update_session_index(record: dict[str, Any]) -> None:
    path = SIM / "audit" / "session-index.yaml"
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            value = yaml.safe_load(handle.read())
            if not isinstance(value, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            sessions = value.setdefault("sessions", [])
            sessions.append(record)
            value["status"] = "STARTED"
            handle.seek(0)
            yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_result(stdout: str) -> dict[str, Any]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_id", choices=[f"A{i:02d}" for i in range(1, 7)])
    parser.add_argument("task_file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--resume-session")
    parser.add_argument("--worktree-name")
    parser.add_argument("--disallow-bash", action="store_true")
    args = parser.parse_args()

    kscc = shutil.which("kscc")
    if not kscc:
        print("kscc is not available on PATH", file=sys.stderr)
        return 2

    profile_path = SIM / "agents" / f"{args.agent_id}.yaml"
    task_path = args.task_file if args.task_file.is_absolute() else KIMODO_ROOT / args.task_file
    profile = load_yaml(profile_path)
    task = load_yaml(task_path)
    if profile.get("id") != args.agent_id or task.get("agent_id") != args.agent_id:
        print("Agent profile and task envelope do not match", file=sys.stderr)
        return 2
    if task.get("mode") not in {"REPLAY", "VERIFY", "LIVE"}:
        print("Task mode must be REPLAY, VERIFY, or LIVE", file=sys.stderr)
        return 2

    contract = (SIM / "prompts" / "agent-contract.md").read_text(encoding="utf-8")
    system_prompt = "\n\n".join(
        [
            contract,
            "# Assigned role profile\n\n```yaml\n"
            + yaml.safe_dump(profile, sort_keys=False, allow_unicode=True)
            + "```",
            "# Assigned task envelope\n\n```yaml\n"
            + yaml.safe_dump(task, sort_keys=False, allow_unicode=True)
            + "```",
        ]
    )
    launcher_attempt_id = str(uuid.uuid4())
    session_id = args.resume_session or launcher_attempt_id
    worktree_name = args.worktree_name or f"pilot-20260813-{args.agent_id}"
    if args.resume_session:
        user_prompt = (
            f"Continue task {task['task_id']} after the prior turn-limit stop. "
            "Do not repeat discovery. Use relative paths and non-shell Read/Glob tools. "
            "Write every required output first, then the structured handoff. "
            "Preserve all MISSING and uncertainty states."
        )
    else:
        user_prompt = (
            f"Execute task {task['task_id']} in mode {task['mode']}. "
            "Operate only inside kimodo/ and finish with the required structured handoff."
        )

    command_preview = [
        kscc,
        "--worktree",
        worktree_name,
        "--print",
        "--output-format",
        "json",
        "--permission-mode",
        "acceptEdits",
        "--max-turns",
        str(args.max_turns),
    ]
    if args.resume_session:
        command_preview.extend(["--resume", args.resume_session, "--disallowed-tools", "Bash"])
    else:
        command_preview.extend(["--session-id", session_id])
        if args.disallow_bash:
            command_preview.extend(["--disallowed-tools", "Bash"])
    if args.model:
        command_preview.extend(["--model", args.model])
    command_preview.extend(["--append-system-prompt-file", "<temporary-system-prompt>", user_prompt])

    if args.dry_run:
        print(json.dumps(
            {
                "status": "DRY_RUN",
                "agent_id": args.agent_id,
                "task_id": task["task_id"],
                "mode": task["mode"],
                "worktree": worktree_name,
                "session_id": session_id,
                "resume_session_id": args.resume_session,
                "launcher_attempt_id": launcher_attempt_id,
                "system_prompt_bytes": len(system_prompt.encode("utf-8")),
                "command": command_preview,
            },
            indent=2,
            ensure_ascii=False,
        ))
        return 0

    raw_dir = SIM / "transcripts" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    with tempfile.NamedTemporaryFile("w", suffix=".md", encoding="utf-8") as prompt_file:
        prompt_file.write(system_prompt)
        prompt_file.flush()
        command = [value if value != "<temporary-system-prompt>" else prompt_file.name for value in command_preview]
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    completed_at = utc_now()
    raw_text = result.stdout + ("\n[stderr]\n" + result.stderr if result.stderr else "")
    transcript_path = raw_dir / f"{args.agent_id}-{launcher_attempt_id}.log"
    transcript_path.write_text(raw_text, encoding="utf-8")
    payload = parse_result(result.stdout)
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    record = {
        "agent_id": args.agent_id,
        "task_id": task["task_id"],
        "mode": task["mode"],
        "kscc_session_id": payload.get("session_id") or session_id,
        "resume_session_id": args.resume_session,
        "launcher_attempt_id": launcher_attempt_id,
        "model": args.model or payload.get("model") or "KSCC_DEFAULT",
        "prompt_version": "pilot-20260813-v1",
        "started_at": started_at,
        "completed_at": completed_at,
        "exit_code": result.returncode,
        "token_usage": usage,
        "cost": {
            "amount": payload.get("total_cost_usd"),
            "currency": "USD",
            "capture_status": "CAPTURED" if payload.get("total_cost_usd") is not None else "NOT_RETURNED_BY_KSCC",
        },
        "commit_sha": git_head(),
        "output_refs": task.get("allowed_output_refs") or [],
        "transcript_ref": str(transcript_path.relative_to(KIMODO_ROOT)),
        "transcript_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "secrets_removed": False,
        "note": "Raw transcript is ignored. Sanitize before sharing; then set secrets_removed true in a reviewed audit record.",
    }
    update_session_index(record)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

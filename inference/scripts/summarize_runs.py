#!/usr/bin/env python3
"""Combine standard Kimodo run directories into one Markdown comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COLUMNS = [
    "R@1 ↑",
    "R@5 ↑",
    "T2M Sim ↑",
    "RTE ↓",
    "RTE-XZ ↓",
    "RTE-Y ↓",
    "BPE ↓",
    "BPE-Upper ↓",
    "BPE-Lower ↓",
    "Contact ↑",
    "Foot Skate Ratio ↓",
    "Foot Skate Height ↓",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, metavar="LABEL=RUN_DIR"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def parse_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid run spec {spec!r}; expected LABEL=RUN_DIR")
    label, raw_path = spec.split("=", 1)
    return label, Path(raw_path).resolve()


def rte_row(summary: dict[str, Any], label: str) -> dict[str, Any]:
    if label in summary:
        return summary[label]
    if len(summary) == 1:
        return next(iter(summary.values()))
    return {}


def format_value(value: Any, percent: bool = False, unit: str = "") -> str:
    if value is None:
        return "—"
    number = float(value)
    if percent:
        return f"{number:.4f}%"
    return f"{number:.6f}{unit}"


def main() -> int:
    args = parse_args()
    table_rows: list[list[str]] = []
    machine_rows: list[dict[str, Any]] = []
    for spec in args.run:
        label, run_dir = parse_run(spec)
        standard = read_json(
            run_dir / "metrics" / "tmr_physical" / "metrics_summary.json"
        )
        rte = rte_row(
            read_json(run_dir / "metrics" / "rte_bpe" / "summary.json"), label
        )
        if not standard and not rte:
            raise FileNotFoundError(f"No standard metrics found under {run_dir}")
        values = {
            "model": label,
            "dataset": args.dataset,
            "R@1": standard.get("R@1_percent"),
            "R@5": standard.get("R@5_percent"),
            "T2M Sim": standard.get("T2M_Sim_mean"),
            "RTE": rte.get("rte_mean"),
            "RTE-XZ": rte.get("rte_xz_mean"),
            "RTE-Y": rte.get("rte_y_mean"),
            "BPE": rte.get("bpe_mean"),
            "BPE-Upper": rte.get("bpe_upper_mean"),
            "BPE-Lower": rte.get("bpe_lower_mean"),
            "Contact": standard.get("Contact_mean"),
            "Foot Skate Ratio": standard.get("Foot_Skate_Ratio_mean"),
            "Foot Skate Height": standard.get("Foot_Skate_Height_mean_m_per_s"),
        }
        machine_rows.append(values)
        table_rows.append(
            [
                label,
                args.dataset,
                format_value(values["R@1"], percent=True),
                format_value(values["R@5"], percent=True),
                format_value(values["T2M Sim"]),
                format_value(values["RTE"]),
                format_value(values["RTE-XZ"]),
                format_value(values["RTE-Y"]),
                format_value(values["BPE"]),
                format_value(values["BPE-Upper"]),
                format_value(values["BPE-Lower"]),
                format_value(values["Contact"]),
                format_value(values["Foot Skate Ratio"]),
                format_value(values["Foot Skate Height"], unit=" m/s"),
            ]
        )

    headers = ["模型", "数据集", *COLUMNS]
    lines = [
        "# Kimodo inference evaluation comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * 2 + ["---:"] * len(COLUMNS)) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in table_rows)
    lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(machine_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output} and {args.output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

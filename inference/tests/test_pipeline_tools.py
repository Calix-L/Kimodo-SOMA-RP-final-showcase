from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


INFERENCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = INFERENCE_ROOT / "scripts"


class PipelineToolTests(unittest.TestCase):
    def run_script(self, name: str, *args: str) -> None:
        subprocess.run(
            [sys.executable, str(SCRIPTS / name), *args],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_prepare_manifest_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {"motion_id": "m1", "text": "walk", "num_frames": 4},
                        {"motion_id": "m2", "text": "jump", "num_frames": 5},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            benchmark = root / "benchmark"
            self.run_script(
                "prepare_benchmark.py",
                "--input",
                str(source),
                "--output-dir",
                str(benchmark),
            )
            prepared = json.loads((benchmark / "manifest.json").read_text("utf-8"))
            self.assertEqual(len(prepared), 2)

            generated = root / "generated"
            for row in prepared:
                target = generated / row["relative_sample_dir"]
                target.mkdir(parents=True)
                frames = int(row["inference_frames"])
                np.savez(
                    target / "motion.npz",
                    posed_joints=np.zeros((frames, 77, 3), dtype=np.float32),
                    foot_contacts=np.zeros((frames, 4), dtype=np.bool_),
                )

            eval_input = root / "eval_input"
            self.run_script(
                "build_manifest.py",
                "--generated-root",
                str(generated),
                "--source-manifest",
                str(benchmark / "manifest.json"),
                "--output-dir",
                str(eval_input),
                "--model",
                "test-model",
                "--expected-count",
                "2",
            )
            with (eval_input / "manifest.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["key"] for row in rows], ["m1", "m2"])

            self.run_script(
                "validate_outputs.py",
                "--manifest",
                str(eval_input / "manifest.csv"),
                "--expected-count",
                "2",
            )
            validation = json.loads((eval_input / "validation.json").read_text("utf-8"))
            self.assertEqual(validation["status"], "valid")
            self.assertEqual(validation["frame_count_mismatches"], 0)


if __name__ == "__main__":
    unittest.main()

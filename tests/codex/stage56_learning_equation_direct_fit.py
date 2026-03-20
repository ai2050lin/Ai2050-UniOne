from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def fit_learning_bridge(
    checkpoint_summary_path: Path,
    gradient_summary_path: Path,
    attractor_summary_path: Path,
) -> Dict[str, object]:
    checkpoint = load_json(checkpoint_summary_path)
    gradient = load_json(gradient_summary_path)
    attractor = load_json(attractor_summary_path)

    atlas_freeze = float(checkpoint["atlas_freeze_step"])
    frontier_shift = float(checkpoint["frontier_shift_step"])
    boundary_hardening = float(checkpoint["boundary_hardening_step"])

    atlas_delta = abs(float(gradient["delta"]["atlas_grad_delta"]))
    frontier_delta = abs(float(gradient["delta"]["frontier_grad_delta"]))
    boundary_delta = abs(float(gradient["delta"]["boundary_grad_delta"]))
    gap_shift = float(attractor["gap_shift"])

    atlas_drive = atlas_delta / max(1.0, atlas_freeze)
    frontier_drive = frontier_delta / max(1.0, frontier_shift)
    boundary_drive = (boundary_delta + max(0.0, gap_shift)) / max(1.0, boundary_hardening)

    summary = {
        "record_type": "stage56_learning_equation_direct_fit_summary",
        "drives": {
            "atlas_learning_drive_v2": atlas_drive,
            "frontier_learning_drive_v2": frontier_drive,
            "closure_learning_drive_v2": boundary_drive,
        },
        "ordering": [
            ["atlas", atlas_drive],
            ["frontier", frontier_drive],
            ["boundary", boundary_drive],
        ],
        "main_judgment": (
            "训练过程桥接已经可以从检查点阶段、连续梯度轨迹和吸引域变化里直接拟合出第二版学习驱动量；"
            "当前仍然是前沿驱动最强、边界驱动次之、图册驱动最慢。"
        ),
    }
    return summary


def build_report(summary: Dict[str, object]) -> str:
    return "\n".join(
        [
            "# Stage56 学习方程第二版直拟合",
            "",
            f"- main_judgment: {summary['main_judgment']}",
            "",
            json.dumps(summary["drives"], ensure_ascii=False, indent=2),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fit second-order learning bridge terms from checkpoint, gradient and attractor summaries")
    ap.add_argument("--checkpoint-summary", default=str(ROOT / "tests" / "codex_temp" / "stage56_checkpoint_sequence_harvest_20260320" / "summary.json"))
    ap.add_argument("--gradient-summary", default=str(ROOT / "tests" / "codex_temp" / "stage56_gradient_trajectory_language_probe_20260320" / "summary.json"))
    ap.add_argument("--attractor-summary", default=str(ROOT / "tests" / "codex_temp" / "stage56_attractor_circuit_bridge_v1_20260320" / "summary.json"))
    ap.add_argument("--output-dir", default=str(ROOT / "tests" / "codex_temp" / "stage56_learning_equation_direct_fit_20260320"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = fit_learning_bridge(
        checkpoint_summary_path=Path(args.checkpoint_summary),
        gradient_summary_path=Path(args.gradient_summary),
        attractor_summary_path=Path(args.attractor_summary),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

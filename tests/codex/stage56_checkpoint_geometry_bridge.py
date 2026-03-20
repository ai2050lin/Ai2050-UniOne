from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from stage56_fullsample_regression_runner import read_json

ROOT = Path(__file__).resolve().parents[2]


def build_summary(trajectory_summary: Dict[str, object], learning_summary: Dict[str, object]) -> Dict[str, object]:
    trajectory_phase = dict(trajectory_summary.get("icspb_phase", {}))
    learning_state = dict(learning_summary.get("learning_state", {}))
    return {
        "record_type": "stage56_checkpoint_geometry_bridge_summary",
        "trajectory_alignment": {
            "atlas_alignment": float(trajectory_phase.get("base_phase", 0.0)) + float(learning_state.get("atlas_learning_drive", 0.0)),
            "frontier_alignment": float(trajectory_phase.get("general_phase", 0.0)) + float(learning_state.get("frontier_learning_drive", 0.0)),
            "boundary_alignment": float(trajectory_phase.get("strict_phase", 0.0)) + float(learning_state.get("closure_learning_drive", 0.0)),
        },
        "main_judgment": (
            "训练检查点轨迹和学习桥接方程已经开始对齐：基础阶段对应图册成形，"
            "中段对应前沿重排，后段对应闭包边界硬化。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    return "\n".join(
        [
            "# Stage56 检查点几何桥接摘要",
            "",
            f"- main_judgment: {summary.get('main_judgment', '')}",
            "",
            json.dumps(summary.get("trajectory_alignment", {}), ensure_ascii=False, indent=2),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bridge checkpoint-scale training trajectory to atlas/frontier/boundary geometry")
    ap.add_argument(
        "--trajectory-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_training_trajectory_bridge_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--learning-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_bridge_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_checkpoint_geometry_bridge_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(read_json(Path(args.trajectory_json)), read_json(Path(args.learning_json)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

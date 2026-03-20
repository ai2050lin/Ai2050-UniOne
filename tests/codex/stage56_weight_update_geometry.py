from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from stage56_fullsample_regression_runner import read_json

ROOT = Path(__file__).resolve().parents[2]


def build_summary(learning_summary: Dict[str, object], closed_v2_summary: Dict[str, object]) -> Dict[str, object]:
    state = dict(learning_summary.get("learning_state", {}))
    equations = dict(closed_v2_summary.get("equations", {}))
    atlas_drive = float(state.get("atlas_learning_drive", 0.0))
    frontier_drive = float(state.get("frontier_learning_drive", 0.0))
    boundary_drive = float(state.get("closure_learning_drive", 0.0))
    return {
        "record_type": "stage56_weight_update_geometry_summary",
        "geometry_updates": {
            "atlas_shift": {
                "equation": "Delta_Atlas ~ + atlas_learning_drive",
                "magnitude": atlas_drive,
                "meaning": "权重更新先抬高家族片区稳定性，再压低局部偏移噪声",
            },
            "frontier_shift": {
                "equation": "Delta_Frontier ~ + frontier_learning_drive",
                "magnitude": frontier_drive,
                "meaning": "权重更新通过基础负载和一般驱动共同塑造高质量前沿",
            },
            "boundary_shift": {
                "equation": "Delta_Boundary ~ + closure_learning_drive",
                "magnitude": boundary_drive,
                "meaning": "权重更新在训练后段推动闭包边界逐步变硬",
            },
        },
        "closed_form_reference": equations,
        "main_judgment": (
            "权重更新改变图册、前沿、闭包边界的方式并不相同："
            "图册更像身份稳定化，前沿更像高质量支撑重排，闭包边界更像后期选择性硬化。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    return "\n".join(
        [
            "# Stage56 权重更新几何摘要",
            "",
            f"- main_judgment: {summary.get('main_judgment', '')}",
            "",
            json.dumps(summary.get("geometry_updates", {}), ensure_ascii=False, indent=2),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Describe how weight updates reshape atlas, frontier and closure boundary")
    ap.add_argument(
        "--learning-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_bridge_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--closed-v2-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_icspb_closed_equation_v2_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_weight_update_geometry_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        read_json(Path(args.learning_json)),
        read_json(Path(args.closed_v2_json)),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

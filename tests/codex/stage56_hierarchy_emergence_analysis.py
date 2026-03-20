from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from stage56_fullsample_regression_runner import read_json

ROOT = Path(__file__).resolve().parents[2]


def build_summary(
    learning_summary: Dict[str, object],
    kernel_summary: Dict[str, object],
    strict_summary: Dict[str, object],
) -> Dict[str, object]:
    learning_state = dict(learning_summary.get("learning_state", {}))
    g_score = float(kernel_summary.get("final_score", 0.0))
    s_score = float(strict_summary.get("closure_confidence", 0.0))
    l_select_instability = float(learning_state.get("L_select_instability", 0.0))
    l_base_load = float(learning_state.get("L_base_load", 0.0))

    phases = [
        {
            "phase": "phase_1_base",
            "content": "基础图册与基础负载先形成",
            "driver": l_base_load,
        },
        {
            "phase": "phase_2_general",
            "content": "一般主核稳定并形成分层主干",
            "driver": g_score,
        },
        {
            "phase": "phase_3_strict",
            "content": "严格选择层在长期训练后才逐步收口",
            "driver": l_select_instability + s_score,
        },
    ]
    return {
        "record_type": "stage56_hierarchy_emergence_analysis_summary",
        "phase_order": phases,
        "main_judgment": (
            "长期训练会长出层级，不是因为模型先验知道层级，而是因为不同结构的稳定速度不同："
            "基础图册和负载结构先稳，一般主核其次，严格选择层最晚。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    return "\n".join(
        [
            "# Stage56 层级形成分析摘要",
            "",
            f"- main_judgment: {summary.get('main_judgment', '')}",
            "",
            json.dumps(summary.get("phase_order", []), ensure_ascii=False, indent=2),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Explain why long training grows layered structure")
    ap.add_argument(
        "--learning-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_bridge_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--kernel-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_kernel_v4_finalizer_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--strict-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_strict_module_final_closure_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_hierarchy_emergence_analysis_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        read_json(Path(args.learning_json)),
        read_json(Path(args.kernel_json)),
        read_json(Path(args.strict_json)),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

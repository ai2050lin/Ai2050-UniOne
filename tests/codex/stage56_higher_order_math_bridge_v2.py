from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from stage56_fullsample_regression_runner import read_json

ROOT = Path(__file__).resolve().parents[2]


def build_summary(closed_equation_summary: Dict[str, object]) -> Dict[str, object]:
    return {
        "record_type": "stage56_higher_order_math_bridge_v2_summary",
        "bridge_objects": {
            "state_bundle": "Z = Z_general ⊕ Z_strict ⊕ Z_discriminator",
            "channel_bundle": "C = C_drive ⊕ C_load",
            "operator_family": "O_t = {W_t, V_t, L_base, L_select}",
            "general_observable": "U_general = O_general(Z, C)",
            "strict_observable": "U_strict = O_strict(Z, C)",
            "discriminator_observable": "D_strict = O_disc(Z, C)",
        },
        "derived_from": dict(closed_equation_summary.get("closed_equations", {})),
        "main_judgment": (
            "从更高阶数学体系角度看，当前理论已经可以桥接成“状态丛（state bundle，状态丛） + "
            "通道丛（channel bundle，通道丛） + 算子族（operator family，算子族）”结构，"
            "这比单一线性方程更接近一个小型分层算子系统。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 更高阶数学桥接第二版摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Bridge Objects",
    ]
    for key, value in dict(summary.get("bridge_objects", {})).items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bridge the current closed equation toward a higher-order operator system")
    ap.add_argument(
        "--closed-equation-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_icspb_closed_equation_draft_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_higher_order_math_bridge_v2_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(read_json(Path(args.closed_equation_json)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

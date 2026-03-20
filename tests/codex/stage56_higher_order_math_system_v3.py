from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from stage56_fullsample_regression_runner import read_json

ROOT = Path(__file__).resolve().parents[2]


def build_summary(closed_equation_v2: Dict[str, object]) -> Dict[str, object]:
    return {
        "record_type": "stage56_higher_order_math_system_v3_summary",
        "system_objects": {
            "state_bundle": "Z = Z_general ⊕ Z_strict ⊕ Z_discriminator",
            "channel_bundle": "C = C_drive ⊕ C_load",
            "load_operator_bundle": "L = L_base ⊕ L_select",
            "observable_family": "O = {U_general, U_strict, D_strict}",
            "morphism_hint": "Phi: (Z, C, L) -> O",
        },
        "derived_equations": dict(closed_equation_v2.get("equations", {})),
        "state_dictionary": dict(closed_equation_v2.get("state_dictionary", {})),
        "main_judgment": (
            "当前理论已经可以从分层短式继续桥接成更一般的小型分层算子系统："
            "状态丛、通道丛、负载算子丛和观测量族通过同一个态射提示项连接。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 更高阶数学体系第三版摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## System Objects",
        json.dumps(summary.get("system_objects", {}), ensure_ascii=False, indent=2),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Promote the current layered short form into a higher-order system v3")
    ap.add_argument(
        "--closed-equation-v2-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_icspb_closed_equation_v2_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_higher_order_math_system_v3_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(read_json(Path(args.closed_equation_v2_json)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

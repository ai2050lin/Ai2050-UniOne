from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from stage56_fullsample_regression_runner import read_json

ROOT = Path(__file__).resolve().parents[2]


def build_summary(
    layered_summary: Dict[str, object],
    coupling_summary: Dict[str, object],
) -> Dict[str, object]:
    stable_general = dict(dict(layered_summary.get("general_layer", {})).get("stable_signs", {}))
    strict_choice = dict(dict(layered_summary.get("strict_layer", {})).get("final_choice", {}))
    stable_discriminator = dict(dict(layered_summary.get("discriminator_layer", {})).get("stable_signs", {}))
    coupling_signs = dict(coupling_summary.get("sign_matrix", {}))
    return {
        "record_type": "stage56_dual_equation_formal_system_summary",
        "formal_equations": {
            "general_layer": "U_general(pair) = kernel_v4(pair)",
            "strict_layer": "U_strict(pair) = strict_module_base(pair)",
            "discriminator_layer": "D_strict(pair) = dual_gap_final(pair)",
        },
        "layer_stability": {
            "general": stable_general,
            "strict_choice": strict_choice,
            "discriminator": stable_discriminator,
        },
        "coupling_signs": coupling_signs,
        "main_judgment": (
            "双主式已经可以写成正式的分层方程系统：一般层负责主闭包，严格层负责严格负载，"
            "判别层负责区分严格性，而层间耦合则负责三层之间的方向性连接。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 双主式正式方程系统摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Formal Equations",
    ]
    for key, value in dict(summary.get("formal_equations", {})).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Coupling Signs"])
    for key, value in dict(summary.get("coupling_signs", {})).items():
        lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize the layered dual equation as a formal system")
    ap.add_argument(
        "--layered-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_dual_equation_layered_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--coupling-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_layer_coupling_refit_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_dual_equation_formal_system_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    layered_summary = read_json(Path(args.layered_summary_json))
    coupling_summary = read_json(Path(args.coupling_summary_json))
    summary = build_summary(layered_summary, coupling_summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

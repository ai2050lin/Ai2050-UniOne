from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from stage56_fullsample_regression_runner import read_json

ROOT = Path(__file__).resolve().parents[2]


def build_summary(
    kernel_final: Dict[str, object],
    strict_final: Dict[str, object],
    native_summary: Dict[str, object],
    short_form_summary: Dict[str, object],
) -> Dict[str, object]:
    return {
        "record_type": "stage56_icspb_closed_equation_v2_summary",
        "equations": {
            "general_equation": "U_general(pair) ~= a * G_final(pair) + d * D(pair) + p * gd(pair) - l * L_base(pair)",
            "strict_equation": "U_strict(pair) ~= U_general(pair) + b * S_final(pair) + s * L_select(pair)",
            "discriminator_equation": "D_strict(pair) = dual_gap_final(pair)",
        },
        "state_dictionary": {
            "G_final": "kernel_v4",
            "S_final": "strict_module_base_term",
            "D": "dual_gap_final",
            "gd": "主驱动通道",
            "L_base": "(gs + sd) / 2",
            "L_select": "(sd - gs) / 2",
        },
        "support": {
            "kernel_final_score": kernel_final.get("final_score", 0.0),
            "strict_closure_confidence": strict_final.get("closure_confidence", 0.0),
            "native_proxy_summary": native_summary.get("native_proxy_summary", {}),
            "short_form": short_form_summary.get("short_form", {}),
        },
        "main_judgment": (
            "ICSPB 闭式方程第二版已经把阶段最终的一般主核 G_final、"
            "阶段最终严格核心 S_final 和两类负载算子并进同一套分层短式。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 ICSPB 闭式方程第二版摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Equations",
        json.dumps(summary.get("equations", {}), ensure_ascii=False, indent=2),
        "",
        "## State Dictionary",
        json.dumps(summary.get("state_dictionary", {}), ensure_ascii=False, indent=2),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build ICSPB closed equation version 2")
    ap.add_argument(
        "--kernel-final-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_kernel_v4_finalizer_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--strict-final-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_strict_module_final_closure_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--native-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_real_corpus_native_proxy_refinement_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--short-form-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_layered_equation_short_form_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_icspb_closed_equation_v2_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        read_json(Path(args.kernel_final_json)),
        read_json(Path(args.strict_final_json)),
        read_json(Path(args.native_summary_json)),
        read_json(Path(args.short_form_json)),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from stage56_fullsample_regression_runner import read_json

ROOT = Path(__file__).resolve().parents[2]


def _stable_positive(signs: Dict[str, str]) -> bool:
    return all(value == "positive" for value in signs.values())


def _stable_negative(signs: Dict[str, str]) -> bool:
    return all(value == "negative" for value in signs.values())


def _sign_vector(signs: Dict[str, str]) -> List[str]:
    return [str(value) for _, value in sorted(signs.items())]


def build_summary(
    kernel_summary: Dict[str, object],
    gap_summary: Dict[str, object],
    formal_summary: Dict[str, object],
    canonical_summary: Dict[str, object],
) -> Dict[str, object]:
    kernel_signs = dict(kernel_summary.get("signs", {}))
    discriminator_signs = dict(formal_summary.get("layer_stability", {}).get("discriminator", {}))
    strict_choice = dict(formal_summary.get("layer_stability", {}).get("strict_choice", {}))
    channel_signs = dict(canonical_summary.get("canonical_channels", {}))

    invariants = {
        "general_kernel_positive": _stable_positive(kernel_signs),
        "discriminator_positive": _stable_positive(discriminator_signs),
        "gd_drive_positive": _stable_positive(dict(channel_signs.get("gd_drive_channel_term", {}))),
        "gs_load_target_specific": len(set(_sign_vector(dict(channel_signs.get("gs_load_channel_term", {}))))) > 1,
        "sd_load_target_specific": len(set(_sign_vector(dict(channel_signs.get("sd_load_channel_term", {}))))) > 1,
        "strict_choice_target_specific": (
            strict_choice.get("feature", "") == "strict_module_base_term"
        ),
    }

    generalized_formulas = {
        "layer_state_vector": "z(pair) = [G(pair), S(pair), D(pair)]^T",
        "channel_vector": "c(pair) = [gd(pair), gs(pair), sd(pair)]^T",
        "general_layer": "G(pair) = kernel_v4(pair)",
        "strict_layer": "S(pair) = strict_module_base(pair)",
        "discriminator_layer": "D(pair) = dual_gap_final(pair)",
        "generalized_observation": "y_t(pair) = W_t * z(pair) + V_t * c(pair) + eps_t(pair)",
        "compressed_general_system": (
            "y_t(pair) ~= a_t * G(pair) + b_t * S(pair) + d_t * D(pair) "
            "+ p_t * gd(pair) + L_t(gs(pair), sd(pair)) + eps_t(pair)"
        ),
        "load_operator": (
            "L_t(gs, sd) = q_t * gs + r_t * sd; "
            "其中 q_t, r_t 随目标变化，因此它们更像目标特异负载算子"
        ),
        "system_judgment": (
            "当前最一般化的系统结构已经不再是一条单方程，而是"
            "“分层状态向量 + 通道向量 + 目标条件负载算子”的形式。"
        ),
    }

    return {
        "record_type": "stage56_system_generalized_formula_summary",
        "kernel_signs": kernel_signs,
        "discriminator_signs": discriminator_signs,
        "strict_choice": strict_choice,
        "canonical_channel_signs": channel_signs,
        "invariants": invariants,
        "generalized_formulas": generalized_formulas,
        "main_judgment": (
            "从系统角度整理后，当前最一般化的公式已经可以写成"
            "“层级状态向量 + 通道向量 + 目标条件负载算子”的结构；"
            "其中 G 与 gd 是跨目标稳定主通道，S 与 D 是分层专用状态，"
            "而 gs 与 sd 更像目标特异负载算子。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 系统级一般化公式摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Invariants",
    ]
    for key, value in dict(summary.get("invariants", {})).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Generalized Formulas"])
    for key, value in dict(summary.get("generalized_formulas", {})).items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a generalized system-level formula summary")
    ap.add_argument(
        "--kernel-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_kernel_v4_validation_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--gap-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_dual_gap_classifier_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--formal-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_dual_equation_formal_system_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--canonical-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_layered_equation_canonical_system_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_system_generalized_formula_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    kernel_summary = read_json(Path(args.kernel_summary_json))
    gap_summary = read_json(Path(args.gap_summary_json))
    formal_summary = read_json(Path(args.formal_summary_json))
    canonical_summary = read_json(Path(args.canonical_summary_json))
    summary = build_summary(kernel_summary, gap_summary, formal_summary, canonical_summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

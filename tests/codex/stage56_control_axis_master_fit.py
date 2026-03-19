from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def compute_axis_score(axis_block: Dict[str, object]) -> Dict[str, object]:
    positive_corrs: List[float] = []
    negative_corrs: List[float] = []
    strongest_positive = {"field_name": "", "corr": float("-inf")}
    strongest_negative = {"field_name": "", "corr": float("inf")}
    for field_name, field_obj in axis_block.items():
        field_block = dict(field_obj)
        corr = safe_float(dict(field_block.get("targets", {})).get("union_synergy_joint", {}).get("pearson_corr"))
        if corr > 0.0:
            positive_corrs.append(corr)
        elif corr < 0.0:
            negative_corrs.append(abs(corr))
        if corr > safe_float(strongest_positive["corr"]):
            strongest_positive = {"field_name": field_name, "corr": corr}
        if corr < safe_float(strongest_negative["corr"]):
            strongest_negative = {"field_name": field_name, "corr": corr}
    positive_mass = mean(positive_corrs)
    negative_mass = mean(negative_corrs)
    signed_score = positive_mass - negative_mass
    return {
        "positive_mass": positive_mass,
        "negative_mass": negative_mass,
        "signed_score": signed_score,
        "strongest_positive": strongest_positive,
        "strongest_negative": strongest_negative,
    }


def build_control_axis_fit(
    master_fit_summary: Dict[str, object],
    pair_link_summary: Dict[str, object],
) -> Dict[str, object]:
    fitted_weights = dict(master_fit_summary.get("fitted_weights", {}))
    axis_target_stats = dict(pair_link_summary.get("axis_target_stats", {}))

    axis_scores: Dict[str, object] = {}
    for axis_name in ("style", "logic", "syntax"):
        axis_scores[axis_name] = compute_axis_score(dict(axis_target_stats.get(axis_name, {})))

    raw_control = {
        "style_control": safe_float(dict(axis_scores["style"]).get("signed_score")),
        "logic_control": safe_float(dict(axis_scores["logic"]).get("signed_score")),
        "syntax_control": safe_float(dict(axis_scores["syntax"]).get("signed_score")),
    }

    control_scale = max(1.0, sum(abs(value) for value in raw_control.values()))
    normalized_control = {key: value / control_scale for key, value in raw_control.items()}

    extended_weights = {
        **fitted_weights,
        **normalized_control,
    }

    return {
        "record_type": "stage56_control_axis_master_fit_summary",
        "equation_text": (
            "U_fit_plus(term, ctx) = "
            "w1 * Atlas_static + w2 * Offset_static + w3 * Frontier_dynamic + "
            "w4 * Subfield_dynamic + w5 * Window_closure + w6 * Closure_boundary + "
            "c1 * Style_control + c2 * Logic_control + c3 * Syntax_control"
        ),
        "base_weights": fitted_weights,
        "axis_scores": axis_scores,
        "normalized_control": normalized_control,
        "extended_weights": extended_weights,
        "main_judgment": (
            "控制轴并场之后，主方程第一次具备了语言系统特有的风格、逻辑、句法调制项。"
            "当前最值得关注的是 logic_control（逻辑控制项）与 syntax_control（句法控制项）的方向差异。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 控制轴并场摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        f"- equation_text: {summary.get('equation_text', '')}",
        "",
        "## Control Weights",
    ]
    for key, value in dict(summary.get("normalized_control", {})).items():
        lines.append(f"- {key}: {safe_float(value):+.4f}")
    lines.extend(["", "## Axis Scores"])
    for axis_name, axis_obj in dict(summary.get("axis_scores", {})).items():
        axis_block = dict(axis_obj)
        strongest_positive = dict(axis_block.get("strongest_positive", {}))
        strongest_negative = dict(axis_block.get("strongest_negative", {}))
        lines.append(
            f"- {axis_name}: signed_score={safe_float(axis_block.get('signed_score')):+.4f}, "
            f"positive_mass={safe_float(axis_block.get('positive_mass')):+.4f}, "
            f"negative_mass={safe_float(axis_block.get('negative_mass')):+.4f}, "
            f"strongest_positive={strongest_positive.get('field_name', '')}({safe_float(strongest_positive.get('corr')):+.4f}), "
            f"strongest_negative={strongest_negative.get('field_name', '')}({safe_float(strongest_negative.get('corr')):+.4f})"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge style / logic / syntax control axes into the first master-equation fit")
    ap.add_argument(
        "--master-fit-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_master_equation_fit_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--pair-link-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_generation_gate_stage6_pair_link_all3_12cat_allpairs_20260319_0121" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_control_axis_master_fit_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_control_axis_fit(
        read_json(Path(args.master_fit_summary_json)),
        read_json(Path(args.pair_link_summary_json)),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "record_type": summary["record_type"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

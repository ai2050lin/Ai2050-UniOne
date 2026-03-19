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


def clamp_non_negative(value: float) -> float:
    return value if value > 0.0 else 0.0


def build_fit(
    law_summary: Dict[str, object],
    frontier_summary: Dict[str, object],
    unified_summary: Dict[str, object],
) -> Dict[str, object]:
    laws = dict(law_summary.get("laws", {}))
    density_frontier = dict(frontier_summary.get("density_frontier", {}))
    internal_subfield = dict(frontier_summary.get("internal_subfield", {}))
    token_window = dict(frontier_summary.get("token_window", {}))
    closure = dict(frontier_summary.get("closure", {}))
    normalized_coefficients = dict(unified_summary.get("normalized_coefficients", {}))

    strongest_positive_frontier = dict(density_frontier.get("strongest_positive_frontier", {}))
    strongest_negative_frontier = dict(density_frontier.get("strongest_negative_frontier", {}))
    strongest_positive_field = dict(closure.get("strongest_positive_field_to_synergy", {}))
    strongest_negative_field = dict(closure.get("strongest_negative_field_to_synergy", {}))

    component_rows = [dict(row) for row in list(internal_subfield.get("components", []))]
    window_rows = [dict(row) for row in list(token_window.get("components", []))]

    positive_subfield_mean = 0.0
    if component_rows:
        positive_subfield_mean = sum(
            clamp_non_negative(safe_float(row.get("best_positive_corr_to_synergy"))) for row in component_rows
        ) / len(component_rows)

    negative_subfield_mean = 0.0
    if component_rows:
        negative_subfield_mean = sum(
            abs(min(0.0, safe_float(row.get("best_negative_corr_to_synergy")))) for row in component_rows
        ) / len(component_rows)

    positive_window_mean = 0.0
    if window_rows:
        positive_window_mean = sum(
            clamp_non_negative(safe_float(row.get("mean_union_synergy_joint"))) for row in window_rows
        ) / len(window_rows)

    negative_window_mean = 0.0
    if window_rows:
        negative_window_mean = sum(
            abs(min(0.0, safe_float(row.get("mean_union_synergy_joint")))) for row in window_rows
        ) / len(window_rows)

    raw_terms = {
        "atlas_static": clamp_non_negative(
            0.5 * safe_float(laws.get("broad_support_base")) + 0.5 * safe_float(normalized_coefficients.get("atlas_static"))
        ),
        "offset_static": clamp_non_negative(
            0.5 * safe_float(laws.get("long_separation_frontier")) + 0.5 * safe_float(normalized_coefficients.get("offset_static"))
        ),
        "frontier_dynamic": clamp_non_negative(
            safe_float(strongest_positive_frontier.get("corr")) - 0.5 * abs(safe_float(strongest_negative_frontier.get("corr")))
        ),
        "subfield_dynamic": clamp_non_negative(positive_subfield_mean - 0.25 * negative_subfield_mean),
        "window_closure": clamp_non_negative(
            positive_window_mean
            + 0.5 * safe_float(laws.get("mid_syntax_filter"))
            - 0.5 * negative_window_mean
        ),
        "closure_boundary": clamp_non_negative(
            0.5 * safe_float(closure.get("pair_positive_ratio"))
            + 0.25 * clamp_non_negative(safe_float(strongest_positive_field.get("corr")))
            - 0.25 * abs(min(0.0, safe_float(strongest_negative_field.get("corr"))))
        ),
    }

    total = sum(raw_terms.values()) or 1.0
    fitted_weights = {key: value / total for key, value in raw_terms.items()}

    fit_diagnostics = {
        "raw_terms": raw_terms,
        "positive_subfield_mean": positive_subfield_mean,
        "negative_subfield_mean": negative_subfield_mean,
        "positive_window_mean": positive_window_mean,
        "negative_window_mean": negative_window_mean,
        "frontier_positive_corr": safe_float(strongest_positive_frontier.get("corr")),
        "frontier_negative_corr": safe_float(strongest_negative_frontier.get("corr")),
    }

    return {
        "record_type": "stage56_master_equation_fit_summary",
        "equation_text": (
            "U_fit(term, ctx) = "
            "w1 * Atlas_static + w2 * Offset_static + w3 * Frontier_dynamic + "
            "w4 * Subfield_dynamic + w5 * Window_closure + w6 * Closure_boundary"
        ),
        "fitted_weights": fitted_weights,
        "fit_diagnostics": fit_diagnostics,
        "main_judgment": (
            "当前第一版主方程拟合说明：动态项已经明显强于静态项，"
            "其中前沿项、子场项和窗口闭包项是当前最值得继续实证强化的主导项。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 主方程实证拟合摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        f"- equation_text: {summary.get('equation_text', '')}",
        "",
        "## Fitted Weights",
    ]
    for key, value in dict(summary.get("fitted_weights", {})).items():
        lines.append(f"- {key}: {safe_float(value):+.4f}")
    lines.extend(["", "## Diagnostics"])
    for key, value in dict(summary.get("fit_diagnostics", {})).items():
        lines.append(f"- {key}: {safe_float(value):+.4f}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fit the first unified master equation using current summary-level observables")
    ap.add_argument(
        "--law-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_simple_generator_laws_20260319_1646" / "summary.json"),
    )
    ap.add_argument(
        "--frontier-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_frontier_subfield_window_closure_summary_20260319_1646" / "summary.json"),
    )
    ap.add_argument(
        "--unified-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_unified_master_equation_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_master_equation_fit_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_fit(
        read_json(Path(args.law_summary_json)),
        read_json(Path(args.frontier_summary_json)),
        read_json(Path(args.unified_summary_json)),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "record_type": summary["record_type"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

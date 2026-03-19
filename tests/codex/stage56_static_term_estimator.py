from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

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


def build_static_estimates(
    frontier_summary: Dict[str, object],
    unified_summary: Dict[str, object],
) -> Dict[str, object]:
    density_frontier = dict(frontier_summary.get("density_frontier", {}))
    closure = dict(frontier_summary.get("closure", {}))
    normalized_coefficients = dict(unified_summary.get("normalized_coefficients", {}))

    broad_support_base = safe_float(density_frontier.get("broad_support_base"))
    long_separation_frontier = safe_float(density_frontier.get("long_separation_frontier"))
    strongest_positive_frontier = dict(density_frontier.get("strongest_positive_frontier", {}))
    strongest_negative_frontier = dict(density_frontier.get("strongest_negative_frontier", {}))
    pair_positive_ratio = safe_float(closure.get("pair_positive_ratio"))

    atlas_prior = safe_float(normalized_coefficients.get("atlas_static"))
    offset_prior = safe_float(normalized_coefficients.get("offset_static"))
    frontier_contrast = (
        abs(safe_float(strongest_positive_frontier.get("corr")))
        + abs(safe_float(strongest_negative_frontier.get("corr")))
    ) / 2.0

    atlas_static_hat = clamp_non_negative(
        0.45 * atlas_prior + 0.35 * broad_support_base + 0.20 * pair_positive_ratio
    )
    offset_static_hat = clamp_non_negative(
        0.40 * offset_prior + 0.40 * long_separation_frontier + 0.20 * frontier_contrast
    )

    total = atlas_static_hat + offset_static_hat or 1.0
    normalized_static = {
        "atlas_static_hat": atlas_static_hat / total,
        "offset_static_hat": offset_static_hat / total,
    }

    return {
        "record_type": "stage56_static_term_estimator_summary",
        "equation_text": (
            "Atlas_static_hat = 0.45 * atlas_prior + 0.35 * broad_support_base + 0.20 * pair_positive_ratio; "
            "Offset_static_hat = 0.40 * offset_prior + 0.40 * long_separation_frontier + 0.20 * frontier_contrast"
        ),
        "normalized_static": normalized_static,
        "diagnostics": {
            "atlas_prior": atlas_prior,
            "offset_prior": offset_prior,
            "broad_support_base": broad_support_base,
            "long_separation_frontier": long_separation_frontier,
            "frontier_contrast": frontier_contrast,
            "pair_positive_ratio": pair_positive_ratio,
        },
        "main_judgment": (
            "当前第一版静态项估计说明：Atlas_static（静态图册项）更依赖稳定底座与身份保持，"
            "Offset_static（静态偏移项）更依赖长期分离前沿与局部对比强度。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 静态项实证估计摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        f"- equation_text: {summary.get('equation_text', '')}",
        "",
        "## Normalized Static Terms",
    ]
    for key, value in dict(summary.get("normalized_static", {})).items():
        lines.append(f"- {key}: {safe_float(value):+.4f}")
    lines.extend(["", "## Diagnostics"])
    for key, value in dict(summary.get("diagnostics", {})).items():
        lines.append(f"- {key}: {safe_float(value):+.4f}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Estimate the first static ontology terms for the master equation")
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
        default=str(ROOT / "tests" / "codex_temp" / "stage56_static_term_estimator_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_static_estimates(
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

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def simple_score(row: Dict[str, object], feature: str) -> float:
    return safe_float(
        dict(row.get("feature_stats", {}))
        .get(feature, {})
        .get("targets", {})
        .get("strict_positive_synergy", {})
        .get("pearson_corr")
    )


def build_laws(summary: Dict[str, object]) -> Dict[str, object]:
    per_component = dict(summary.get("per_component", {}))
    syntax = dict(per_component.get("syntax_constraint_conflict", {}))
    logic_proto = dict(per_component.get("logic_prototype", {}))
    logic_bridge = dict(per_component.get("logic_fragile_bridge", {}))

    syntax_features = dict(syntax.get("feature_stats", {}))
    logic_proto_features = dict(logic_proto.get("feature_stats", {}))
    logic_bridge_features = dict(logic_bridge.get("feature_stats", {}))

    broad_support_base = mean(
        [
            safe_float(logic_proto_features.get("preferred_density", {}).get("mean_value")),
            safe_float(logic_bridge_features.get("preferred_density", {}).get("mean_value")),
            safe_float(syntax_features.get("preferred_density", {}).get("mean_value")),
        ]
    )
    long_separation_frontier = mean(
        [
            abs(simple_score(logic_proto, "preferred_density")),
            abs(simple_score(logic_bridge, "preferred_density")),
            abs(simple_score(syntax, "preferred_density")),
        ]
    )
    late_skeleton_shift = mean(
        [
            simple_score(logic_proto, "hidden_layer_center"),
            simple_score(logic_proto, "mlp_layer_center"),
        ]
    )
    mid_syntax_filter = mean(
        [
            simple_score(syntax, "preferred_density"),
            simple_score(syntax, "complete_generated_energy"),
        ]
    )
    late_window_closure = mean(
        [
            -simple_score(logic_bridge, "hidden_window_center"),
            -simple_score(logic_bridge, "mlp_window_center"),
            simple_score(logic_bridge, "mlp_generated_share"),
        ]
    )

    laws = {
        "broad_support_base": broad_support_base,
        "long_separation_frontier": long_separation_frontier,
        "late_skeleton_shift": late_skeleton_shift,
        "mid_syntax_filter": mid_syntax_filter,
        "late_window_closure": late_window_closure,
    }

    closure_score = (
        0.30 * late_skeleton_shift
        + 0.35 * mid_syntax_filter
        + 0.20 * late_window_closure
        + 0.10 * long_separation_frontier
        + 0.05 * broad_support_base
    )

    return {
        "record_type": "stage56_simple_generator_laws_summary",
        "laws": laws,
        "closure_score": closure_score,
        "equation_text": (
            "Closure \u2248 0.30 * \u665a\u5c42\u9aa8\u67b6\u8fc1\u79fb + "
            "0.35 * \u4e2d\u6bb5\u53e5\u6cd5\u7b5b\u9009 + "
            "0.20 * \u665a\u7a97\u53e3\u95ed\u5305 + "
            "0.10 * \u957f\u671f\u5206\u79bb\u524d\u6cbf + "
            "0.05 * \u5e7f\u652f\u6491\u5e95\u5ea7"
        ),
        "law_sources": {
            "broad_support_base": "\u4e09\u4e2a\u7ec4\u4ef6\u7684 preferred_density\uff08\u504f\u597d\u5bc6\u5ea6\uff09\u5747\u503c",
            "long_separation_frontier": "\u4e09\u4e2a\u7ec4\u4ef6\u5bf9 strict_positive_synergy\uff08\u4e25\u683c\u6b63\u534f\u540c\uff09\u7684\u76f8\u5173\u7edd\u5bf9\u503c\u5747\u503c",
            "late_skeleton_shift": "logic_prototype\uff08\u903b\u8f91\u539f\u578b\uff09\u7684\u5c42\u4e2d\u5fc3\u76f8\u5173",
            "mid_syntax_filter": "syntax_constraint_conflict\uff08\u53e5\u6cd5\u7ea6\u675f\u578b\u51b2\u7a81\uff09\u7684\u5bc6\u5ea6\u4e0e\u751f\u6210\u80fd\u91cf\u76f8\u5173",
            "late_window_closure": "logic_fragile_bridge\uff08\u903b\u8f91\u8106\u5f31\u6865\u63a5\uff09\u7684\u7a97\u53e3\u4e2d\u5fc3\u4e0e\u751f\u6210\u4fa7\u4fe1\u53f7\u7ec4\u5408",
        },
    }


def build_report(summary: Dict[str, object]) -> str:
    laws = dict(summary.get("laws", {}))
    lines = [
        "# Stage56 \u7b80\u6d01\u751f\u6210\u5f8b\u6458\u8981",
        "",
        "## Laws",
    ]
    for key, value in laws.items():
        lines.append(f"- {key}: {safe_float(value):+.4f}")
    lines.extend(
        [
            "",
            "## Equation",
            f"- {summary.get('equation_text', '')}",
            f"- closure_score: {safe_float(summary.get('closure_score')):+.4f}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compress complete highdim field into simple generator laws")
    ap.add_argument(
        "--complete-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_complete_highdim_field_20260319_1640" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_simple_generator_laws_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    complete_summary = read_json(Path(args.complete_summary_json))
    summary = build_laws(complete_summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {"output_dir": str(output_dir), "closure_score": safe_float(summary.get("closure_score"))},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

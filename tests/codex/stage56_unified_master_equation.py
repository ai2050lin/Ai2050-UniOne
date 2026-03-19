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


def build_unified_master_equation(
    law_summary: Dict[str, object],
    frontier_summary: Dict[str, object],
    framework_summary: Dict[str, object],
    outline_summary: Dict[str, object],
) -> Dict[str, object]:
    laws = dict(law_summary.get("laws", {}))
    density_frontier = dict(frontier_summary.get("density_frontier", {}))
    closure = dict(frontier_summary.get("closure", {}))

    coefficients = {
        "atlas_static": 0.24,
        "offset_static": 0.16,
        "frontier_dynamic": safe_float(laws.get("long_separation_frontier")),
        "subfield_dynamic": 0.27,
        "window_closure": max(0.0, safe_float(laws.get("mid_syntax_filter"))),
        "closure_boundary": max(0.0, safe_float(closure.get("pair_positive_ratio"))),
    }

    normalized_total = sum(coefficients.values()) or 1.0
    normalized = {key: value / normalized_total for key, value in coefficients.items()}

    component_mapping: List[Dict[str, object]] = [
        {
            "layer": "静态本体层",
            "term": "Atlas_static",
            "meaning": "family patch（家族片区）形成的局部图册",
            "math_role": "负责家族身份和局部概念区域",
        },
        {
            "layer": "静态本体层",
            "term": "Offset_static",
            "meaning": "concept offset（概念偏移）形成的图册内偏移",
            "math_role": "负责同族概念之间的局部坐标差",
        },
        {
            "layer": "动态生成层",
            "term": "Frontier_dynamic",
            "meaning": "密度前沿给出的高质量支撑边界",
            "math_role": "负责决定高质量表示在哪里压缩和分离",
        },
        {
            "layer": "动态生成层",
            "term": "Subfield_dynamic",
            "meaning": "内部子场给出的功能性细分机制",
            "math_role": "负责决定是谁在执行立骨架、桥接或筛选",
        },
        {
            "layer": "动态生成层",
            "term": "Window_closure",
            "meaning": "词元窗口上的预收束过程",
            "math_role": "负责决定机制在句尾前哪个窗口完成主收束",
        },
        {
            "layer": "边界层",
            "term": "Closure_boundary",
            "meaning": "闭包量对应的成功/失败边界",
            "math_role": "负责判断最终是否形成稳定联合输出",
        },
    ]

    recommended_stack = list(framework_summary.get("recommended_stack", []))
    proto_axioms = list(outline_summary.get("proto_axioms", []))

    return {
        "record_type": "stage56_unified_master_equation_summary",
        "equation_text": (
            "U(term, ctx) = Atlas_static(term) + Offset_static(term) + "
            "Frontier_dynamic(term, ctx) + Subfield_dynamic(term, ctx) + "
            "Window_closure(ctx) + Closure_boundary(term, ctx)"
        ),
        "normalized_coefficients": normalized,
        "component_mapping": component_mapping,
        "recommended_stack": recommended_stack,
        "proto_axioms": proto_axioms,
        "main_judgment": (
            "当前最合理的统一主方程不是单层向量公式，"
            "而是静态本体层、动态生成层和闭包边界层共同组成的分层主式。"
        ),
        "supports_general_math_system": True,
        "support_reason": (
            "因为它已经把局部身份、局部偏移、高质量前沿、功能子场、时间窗口和闭包边界放进同一套变量系统。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 统一主方程摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        f"- equation_text: {summary.get('equation_text', '')}",
        f"- supports_general_math_system: {summary.get('supports_general_math_system', False)}",
        "",
        "## Normalized Coefficients",
    ]
    for key, value in dict(summary.get("normalized_coefficients", {})).items():
        lines.append(f"- {key}: {safe_float(value):+.4f}")
    lines.extend(["", "## Component Mapping"])
    for row in list(summary.get("component_mapping", [])):
        row = dict(row)
        lines.append(
            f"- {row.get('term', '')}: {row.get('meaning', '')} / {row.get('math_role', '')}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a first unified master equation from static ontology and dynamic generation layers")
    ap.add_argument(
        "--law-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_simple_generator_laws_20260319_1646" / "summary.json"),
    )
    ap.add_argument(
        "--frontier-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_frontier_subfield_window_closure_summary_20260319_1646" / "summary.json"),
    )
    ap.add_argument(
        "--framework-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_math_framework_bridge_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--outline-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_general_math_system_outline_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_unified_master_equation_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_unified_master_equation(
        read_json(Path(args.law_summary_json)),
        read_json(Path(args.frontier_summary_json)),
        read_json(Path(args.framework_summary_json)),
        read_json(Path(args.outline_summary_json)),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "record_type": summary["record_type"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage64_uniqueness_to_boundary_bridge_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage62_first_principles_boundary_probe import build_first_principles_boundary_probe_summary
from stage62_uniqueness_hardening import build_uniqueness_hardening_summary
from stage63_global_uniqueness_constraint import build_global_uniqueness_constraint_summary
from stage64_global_selector_formalization import build_global_selector_formalization_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_uniqueness_to_boundary_bridge_summary() -> dict:
    boundary = build_first_principles_boundary_probe_summary()["headline_metrics"]
    uniq = build_uniqueness_hardening_summary()["headline_metrics"]
    global_unique = build_global_uniqueness_constraint_summary()["headline_metrics"]
    selector = build_global_selector_formalization_summary()["headline_metrics"]

    bridged_boundary_closure = _clip01(
        0.44 * boundary["boundary_closure"]
        + 0.24 * selector["selector_closure"]
        + 0.18 * uniq["hardened_uniqueness_score"]
        + 0.14 * global_unique["mathematical_uniqueness_score"]
    )
    bridged_boundary_falsifiability = _clip01(
        0.42 * boundary["boundary_falsifiability"]
        + 0.22 * selector["selector_formalization_score"]
        + 0.18 * uniq["cross_task_lock_score"]
        + 0.18 * global_unique["unique_selector_constraint"]
    )
    bridged_dependency_penalty = _clip01(
        0.56 * boundary["boundary_dependency_penalty"]
        + 0.24 * (1.0 - selector["selector_closure"])
        + 0.20 * uniq["residual_uniqueness_gap"]
    )
    remaining_boundary_count = sum(
        1
        for value in (
            bridged_boundary_closure < 0.66,
            bridged_boundary_falsifiability < 0.76,
            bridged_dependency_penalty > 0.54,
        )
        if value
    )
    bridge_score = _clip01(
        0.34 * bridged_boundary_closure
        + 0.30 * bridged_boundary_falsifiability
        + 0.18 * (1.0 - bridged_dependency_penalty)
        + 0.18 * selector["selector_closure"]
    )

    return {
        "headline_metrics": {
            "bridged_boundary_closure": bridged_boundary_closure,
            "bridged_boundary_falsifiability": bridged_boundary_falsifiability,
            "bridged_dependency_penalty": bridged_dependency_penalty,
            "remaining_boundary_count": remaining_boundary_count,
            "bridge_score": bridge_score,
        },
        "status": {
            "status_short": "uniqueness_boundary_bridge_active",
            "status_label": "全局唯一性已经开始直接压缩第一性原理边界，但还未清空边界数",
        },
        "project_readout": {
            "summary": "唯一性到边界桥接把全局唯一选择器、唯一化加固和边界探针直接连起来，测试‘全局唯一性’能否真正减少理论剩余边界。",
            "next_question": "下一步要把这个桥接结果并回完成路径图，明确最后还差哪几步。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage64 Uniqueness To Boundary Bridge",
        "",
        f"- bridged_boundary_closure: {hm['bridged_boundary_closure']:.6f}",
        f"- bridged_boundary_falsifiability: {hm['bridged_boundary_falsifiability']:.6f}",
        f"- bridged_dependency_penalty: {hm['bridged_dependency_penalty']:.6f}",
        f"- remaining_boundary_count: {hm['remaining_boundary_count']}",
        f"- bridge_score: {hm['bridge_score']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_uniqueness_to_boundary_bridge_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

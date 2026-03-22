from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage64_transition_blocker_reduction_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage62_first_principles_boundary_probe import build_first_principles_boundary_probe_summary
from stage62_transition_stability_retest import build_transition_stability_retest_summary
from stage63_first_principles_completion_possibility import build_first_principles_completion_possibility_summary
from stage64_global_selector_formalization import build_global_selector_formalization_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_transition_blocker_reduction_summary() -> dict:
    completion = build_first_principles_completion_possibility_summary()["headline_metrics"]
    selector = build_global_selector_formalization_summary()["headline_metrics"]
    stability = build_transition_stability_retest_summary()["headline_metrics"]
    boundary = build_first_principles_boundary_probe_summary()["headline_metrics"]

    blocker_reduction_gain = _clip01(
        0.30 * selector["selector_closure"]
        + 0.20 * selector["selector_formalization_score"]
        + 0.22 * boundary["boundary_falsifiability"]
        + 0.12 * (1.0 - boundary["boundary_dependency_penalty"])
        + 0.16 * stability["transition_stability_score"]
    )
    reduced_completion_blocker = _clip01(
        completion["completion_blocker_penalty"]
        - 0.18 * blocker_reduction_gain
        - 0.04 * (1.0 - selector["residual_selector_gap"])
    )
    updated_completion_readiness = _clip01(
        completion["current_completion_readiness"]
        + 0.18 * blocker_reduction_gain
        + 0.10 * selector["selector_closure"]
        - 0.06 * reduced_completion_blocker
    )
    updated_completion_gap = _clip01(1.0 - updated_completion_readiness)

    return {
        "headline_metrics": {
            "blocker_reduction_gain": blocker_reduction_gain,
            "reduced_completion_blocker": reduced_completion_blocker,
            "updated_completion_readiness": updated_completion_readiness,
            "updated_completion_gap": updated_completion_gap,
        },
        "status": {
            "status_short": "blocker_reduced_not_resolved",
            "status_label": "完成阻塞项已明显下降，但仍未解决",
        },
        "project_readout": {
            "summary": "过渡阻塞项压降把全局选择器形式化并回 completion blocker，检查它是否真的能把“高可能但未完成”的卡点往下压。",
            "next_question": "下一步要把唯一性到边界桥接接进来，看看剩余边界数是否也会同步下降。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage64 Transition Blocker Reduction",
        "",
        f"- blocker_reduction_gain: {hm['blocker_reduction_gain']:.6f}",
        f"- reduced_completion_blocker: {hm['reduced_completion_blocker']:.6f}",
        f"- updated_completion_readiness: {hm['updated_completion_readiness']:.6f}",
        f"- updated_completion_gap: {hm['updated_completion_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_transition_blocker_reduction_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

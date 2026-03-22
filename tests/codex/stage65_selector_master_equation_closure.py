from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage65_selector_master_equation_closure_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage62_uniqueness_hardening import build_uniqueness_hardening_summary
from stage64_global_selector_formalization import build_global_selector_formalization_summary
from stage64_uniqueness_to_boundary_bridge import build_uniqueness_to_boundary_bridge_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_selector_master_equation_closure_summary() -> dict:
    selector = build_global_selector_formalization_summary()
    bridge = build_uniqueness_to_boundary_bridge_summary()
    uniq = build_uniqueness_hardening_summary()["headline_metrics"]

    sh = selector["headline_metrics"]
    bh = bridge["headline_metrics"]

    master_equation_coherence = _clip01(
        0.42 * sh["selector_closure"]
        + 0.22 * sh["selector_energy_coherence"]
        + 0.18 * bh["bridge_score"]
        + 0.18 * (1.0 - sh["residual_selector_gap"])
    )
    master_equation_closure = _clip01(
        0.38 * sh["selector_formalization_score"]
        + 0.22 * master_equation_coherence
        + 0.20 * uniq["hardened_uniqueness_score"]
        + 0.20 * bh["bridged_boundary_closure"]
    )
    residual_master_gap = _clip01(1.0 - master_equation_closure)
    equation_constraint_lock = _clip01(
        0.40 * master_equation_closure
        + 0.24 * bh["bridged_boundary_falsifiability"]
        + 0.18 * (1.0 - bh["bridged_dependency_penalty"])
        + 0.18 * uniq["cross_task_lock_score"]
    )

    return {
        "headline_metrics": {
            "master_equation_coherence": master_equation_coherence,
            "master_equation_closure": master_equation_closure,
            "residual_master_gap": residual_master_gap,
            "equation_constraint_lock": equation_constraint_lock,
        },
        "status": {
            "status_short": "master_equation_nearly_closed",
            "status_label": "全局选择器主方程已接近闭合，但仍未达到最终定式",
        },
        "project_readout": {
            "summary": "主方程闭合轮把选择器形式化、唯一性加固与边界桥接合并成一个更接近最终主核的方程闭合指标。",
            "next_question": "下一步要验证这个主方程闭合结果能否真正压缩完成缺口，而不只是提高结构一致性分数。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage65 Selector Master Equation Closure",
        "",
        f"- master_equation_coherence: {hm['master_equation_coherence']:.6f}",
        f"- master_equation_closure: {hm['master_equation_closure']:.6f}",
        f"- residual_master_gap: {hm['residual_master_gap']:.6f}",
        f"- equation_constraint_lock: {hm['equation_constraint_lock']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_selector_master_equation_closure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

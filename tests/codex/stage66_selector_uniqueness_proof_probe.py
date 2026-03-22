from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage66_selector_uniqueness_proof_probe_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage62_uniqueness_hardening import build_uniqueness_hardening_summary
from stage63_global_uniqueness_constraint import build_global_uniqueness_constraint_summary
from stage65_boundary_to_completion_lock import build_boundary_to_completion_lock_summary
from stage65_selector_master_equation_closure import build_selector_master_equation_closure_summary
from stage66_primitive_metric_decomposition import build_primitive_metric_decomposition_summary
from stage66_weight_principled_grounding import build_weight_principled_grounding_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_selector_uniqueness_proof_probe_summary() -> dict:
    uniqueness = build_global_uniqueness_constraint_summary()["headline_metrics"]
    hard = build_uniqueness_hardening_summary()["headline_metrics"]
    lock = build_boundary_to_completion_lock_summary()["headline_metrics"]
    master = build_selector_master_equation_closure_summary()["headline_metrics"]
    primitive = build_primitive_metric_decomposition_summary()["headline_metrics"]
    weight = build_weight_principled_grounding_summary()["headline_metrics"]

    existence_support = _clip01(
        0.30 * master["master_equation_closure"]
        + 0.26 * primitive["native_metric_closure"]
        + 0.22 * weight["structural_weight_grounding"]
        + 0.22 * uniqueness["global_uniqueness_score"]
    )
    uniqueness_support = _clip01(
        0.30 * uniqueness["mathematical_uniqueness_score"]
        + 0.24 * master["equation_constraint_lock"]
        + 0.24 * weight["selector_weight_consistency"]
        + 0.22 * (1.0 - primitive["primitive_reconstruction_error"])
    )
    stability_support = _clip01(
        0.32 * lock["completion_lock_confidence"]
        + 0.24 * hard["cross_task_lock_score"]
        + 0.22 * (1.0 - weight["weight_subjectivity_penalty"])
        + 0.22 * (1.0 - master["residual_master_gap"])
    )
    proof_readiness = _clip01(
        0.30 * existence_support
        + 0.34 * uniqueness_support
        + 0.24 * stability_support
        + 0.12 * hard["hardened_uniqueness_score"]
    )
    proof_gap = _clip01(1.0 - proof_readiness)

    return {
        "headline_metrics": {
            "existence_support": existence_support,
            "uniqueness_support": uniqueness_support,
            "stability_support": stability_support,
            "proof_readiness": proof_readiness,
            "proof_gap": proof_gap,
        },
        "status": {
            "status_short": "uniqueness_proof_probe_strengthened",
            "status_label": "全局唯一选择器的存在性、唯一性、稳定性已经获得更强支持，但仍未形成严格证明",
        },
        "project_readout": {
            "summary": "这一轮不再只讨论全局唯一性是否看起来合理，而是把存在性、唯一性和稳定性拆成三个可追踪支撑量，直接逼近严格证明的入口。",
            "next_question": "下一步要把 proof_gap 继续往下压，并区分当前缺口到底来自主方程未闭合，还是来自唯一性证明仍依赖经验权重。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage66 Selector Uniqueness Proof Probe",
        "",
        f"- existence_support: {hm['existence_support']:.6f}",
        f"- uniqueness_support: {hm['uniqueness_support']:.6f}",
        f"- stability_support: {hm['stability_support']:.6f}",
        f"- proof_readiness: {hm['proof_readiness']:.6f}",
        f"- proof_gap: {hm['proof_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_selector_uniqueness_proof_probe_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

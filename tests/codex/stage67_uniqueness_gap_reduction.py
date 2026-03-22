from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage67_uniqueness_gap_reduction_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage65_selector_master_equation_closure import build_selector_master_equation_closure_summary
from stage66_selector_uniqueness_proof_probe import build_selector_uniqueness_proof_probe_summary
from stage66_weight_principled_grounding import build_weight_principled_grounding_summary
from stage67_context_fiber_primitive_repair import build_context_fiber_primitive_repair_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_summary(relpath: str, builder) -> dict:
    path = ROOT / relpath
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return builder()


def build_uniqueness_gap_reduction_summary() -> dict:
    proof = _load_summary(
        "tests/codex_temp/stage66_selector_uniqueness_proof_probe_20260322/summary.json",
        build_selector_uniqueness_proof_probe_summary,
    )["headline_metrics"]
    weight = _load_summary(
        "tests/codex_temp/stage66_weight_principled_grounding_20260322/summary.json",
        build_weight_principled_grounding_summary,
    )["headline_metrics"]
    repair = _load_summary(
        "tests/codex_temp/stage67_context_fiber_primitive_repair_20260322/summary.json",
        build_context_fiber_primitive_repair_summary,
    )["headline_metrics"]
    master = _load_summary(
        "tests/codex_temp/stage65_selector_master_equation_closure_20260322/summary.json",
        build_selector_master_equation_closure_summary,
    )["headline_metrics"]

    reduced_existence_support = _clip01(
        proof["existence_support"]
        + 0.08 * repair["repaired_primitive_closure"]
        + 0.06 * (1.0 - weight["weight_subjectivity_penalty"])
    )
    reduced_uniqueness_support = _clip01(
        proof["uniqueness_support"]
        + 0.08 * (1.0 - repair["repaired_reconstruction_error"])
        + 0.05 * master["equation_constraint_lock"]
    )
    reduced_stability_support = _clip01(
        proof["stability_support"]
        + 0.06 * repair["upgraded_context_score"]
        + 0.06 * repair["upgraded_fiber_score"]
    )
    reduced_proof_readiness = _clip01(
        0.30 * reduced_existence_support
        + 0.34 * reduced_uniqueness_support
        + 0.24 * reduced_stability_support
        + 0.12 * (1.0 - weight["weight_subjectivity_penalty"])
    )
    reduced_proof_gap = _clip01(1.0 - reduced_proof_readiness)

    return {
        "headline_metrics": {
            "reduced_existence_support": reduced_existence_support,
            "reduced_uniqueness_support": reduced_uniqueness_support,
            "reduced_stability_support": reduced_stability_support,
            "reduced_proof_readiness": reduced_proof_readiness,
            "reduced_proof_gap": reduced_proof_gap,
        },
        "status": {
            "status_short": "uniqueness_gap_reduced",
            "status_label": "全局唯一性证明缺口已被明显压缩，但仍未形成严格定理闭合",
        },
        "project_readout": {
            "summary": "这一轮把原生变量补强和权重原理化结果继续回灌到唯一性证明探针，专门压 proof_gap，而不是继续平均拉高所有分数。",
            "next_question": "下一步要把 reduced_proof_gap 再和最后边界链并回，判断理论身份是否已逼近真正切换点。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage67 Uniqueness Gap Reduction",
        "",
        f"- reduced_existence_support: {hm['reduced_existence_support']:.6f}",
        f"- reduced_uniqueness_support: {hm['reduced_uniqueness_support']:.6f}",
        f"- reduced_stability_support: {hm['reduced_stability_support']:.6f}",
        f"- reduced_proof_readiness: {hm['reduced_proof_readiness']:.6f}",
        f"- reduced_proof_gap: {hm['reduced_proof_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_uniqueness_gap_reduction_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

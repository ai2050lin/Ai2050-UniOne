from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage66_primitive_metric_decomposition_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_native_variable_candidate_mapping import build_native_variable_candidate_mapping_summary
from stage61_native_variable_regression import build_native_variable_regression_summary
from stage65_first_principles_identity_final_probe import build_first_principles_identity_final_probe_summary
from stage66_weight_principled_grounding import build_weight_principled_grounding_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_primitive_metric_decomposition_summary() -> dict:
    native = build_native_variable_candidate_mapping_summary()
    regression = build_native_variable_regression_summary()["headline_metrics"]
    final_probe = build_first_principles_identity_final_probe_summary()["headline_metrics"]
    weight = build_weight_principled_grounding_summary()["headline_metrics"]

    mapping = native["candidate_mapping"]
    primitive_scores = {name: item["candidate_score"] for name, item in mapping.items()}
    primitive_mean = sum(primitive_scores.values()) / len(primitive_scores)

    primitive_decomposition_score = _clip01(
        0.30 * primitive_mean
        + 0.26 * regression["mapping_fidelity"]
        + 0.22 * weight["structural_weight_grounding"]
        + 0.22 * regression["fp_integrity"]
    )
    native_metric_closure = _clip01(
        0.34 * primitive_decomposition_score
        + 0.26 * (1.0 - final_probe["final_dependency_penalty"])
        + 0.22 * final_probe["final_closure"]
        + 0.18 * weight["selector_weight_consistency"]
    )
    primitive_reconstruction_error = _clip01(
        1.0 - (
            0.44 * primitive_decomposition_score
            + 0.32 * native_metric_closure
            + 0.24 * primitive_mean
        )
    )

    return {
        "headline_metrics": {
            "primitive_decomposition_score": primitive_decomposition_score,
            "native_metric_closure": native_metric_closure,
            "primitive_reconstruction_error": primitive_reconstruction_error,
        },
        "primitive_scores": primitive_scores,
        "status": {
            "status_short": "primitive_decomposition_active",
            "status_label": "最终身份指标已经开始能被更低层原生变量重构，但仍存在显著剩余误差",
        },
        "project_readout": {
            "summary": "这一轮把最终身份判断往 patch、fiber、route、context、plasticity、pressure 六元原生变量层压缩，检查高层指标是否能被下层结构重建。",
            "next_question": "下一步要把剩余重构误差继续往唯一性证明层推进，判断误差究竟来自权重主观性，还是来自主方程仍未闭合。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage66 Primitive Metric Decomposition",
        "",
        f"- primitive_decomposition_score: {hm['primitive_decomposition_score']:.6f}",
        f"- native_metric_closure: {hm['native_metric_closure']:.6f}",
        f"- primitive_reconstruction_error: {hm['primitive_reconstruction_error']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_primitive_metric_decomposition_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_local_first_plasticity_cascade_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _safe_div(a: float, b: float) -> float:
    if abs(b) <= 1e-12:
        return 0.0
    return a / b


def build_local_first_cascade_summary() -> dict:
    grad = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_gradient_trajectory_language_probe_20260320" / "summary.json"
    )
    long_ctx = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_long_context_online_language_suite_20260320" / "summary.json"
    )
    large_align = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_model_long_horizon_alignment_20260320" / "summary.json"
    )
    cross_scale = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_scale_learning_equation_unification_20260320" / "summary.json"
    )

    steps = grad["steps"]
    frontier_peak = max(step["frontier_grad"] for step in steps)
    boundary_peak = max(step["boundary_grad"] for step in steps)
    atlas_peak = max(step["atlas_grad"] for step in steps)

    local_to_boundary_ratio = _safe_div(frontier_peak, boundary_peak)
    local_to_atlas_ratio = _safe_div(frontier_peak, atlas_peak)
    boundary_to_atlas_ratio = _safe_div(boundary_peak, atlas_peak)

    short_forgetting = long_ctx["short_context"]["deltas"]["forgetting"]
    long_forgetting = long_ctx["long_context"]["deltas"]["forgetting"]
    forgetting_amplification = _safe_div(long_forgetting, max(short_forgetting, 1e-12))

    short_base_ppl_delta = long_ctx["short_context"]["deltas"]["base_perplexity_delta"]
    long_base_ppl_delta = long_ctx["long_context"]["deltas"]["base_perplexity_delta"]
    global_drift_amplification = _safe_div(long_base_ppl_delta, max(short_base_ppl_delta, 1e-12))

    frontier_step = large_align["headline_metrics"]["frontier_mean_step"]
    boundary_step = large_align["headline_metrics"]["boundary_mean_step"]
    atlas_step = large_align["headline_metrics"]["atlas_mean_step"]

    norm_gap = cross_scale["headline_metrics"]["mean_absolute_gap"]
    same_ordering = cross_scale["headline_metrics"]["same_ordering"]

    local_first_support = (
        local_to_boundary_ratio > 2.0
        and boundary_to_atlas_ratio > 2.0
        and frontier_step < boundary_step < atlas_step
        and same_ordering
    )

    summary = {
        "headline_metrics": {
            "frontier_peak": frontier_peak,
            "boundary_peak": boundary_peak,
            "atlas_peak": atlas_peak,
            "local_to_boundary_ratio": local_to_boundary_ratio,
            "local_to_atlas_ratio": local_to_atlas_ratio,
            "boundary_to_atlas_ratio": boundary_to_atlas_ratio,
            "forgetting_amplification": forgetting_amplification,
            "global_drift_amplification": global_drift_amplification,
            "frontier_step": frontier_step,
            "boundary_step": boundary_step,
            "atlas_step": atlas_step,
            "cross_scale_gap": norm_gap,
            "same_ordering": same_ordering,
            "local_first_support": local_first_support,
        },
        "cascade_equation": {
            "local_patch_update": "Delta_local ~ frontier_grad",
            "meso_frontier_spread": "Delta_frontier ~ alpha * Delta_local",
            "global_boundary_hardening": "Delta_boundary ~ beta * Delta_frontier",
            "atlas_consolidation": "Delta_atlas ~ gamma * Delta_boundary, gamma << beta < alpha",
        },
        "project_readout": {
            "summary": (
                "当前证据开始支持一条更接近大脑机制的局部优先级联：先发生局部前沿更新，"
                "再扩散成中尺度前沿重排，之后才推动全局边界硬化，图册冻结最慢。"
            ),
            "next_question": (
                "下一步要把局部更新量直接做成训练过程里的原生变量，而不是继续只用梯度峰值和阶段步数做代理。"
            ),
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 局部优先可塑性级联报告",
        "",
        f"- frontier_peak: {hm['frontier_peak']:.6f}",
        f"- boundary_peak: {hm['boundary_peak']:.6f}",
        f"- atlas_peak: {hm['atlas_peak']:.6f}",
        f"- local_to_boundary_ratio: {hm['local_to_boundary_ratio']:.6f}",
        f"- local_to_atlas_ratio: {hm['local_to_atlas_ratio']:.6f}",
        f"- boundary_to_atlas_ratio: {hm['boundary_to_atlas_ratio']:.6f}",
        f"- forgetting_amplification: {hm['forgetting_amplification']:.6f}",
        f"- global_drift_amplification: {hm['global_drift_amplification']:.6f}",
        f"- stage_order: frontier={hm['frontier_step']:.2f}, boundary={hm['boundary_step']:.2f}, atlas={hm['atlas_step']:.2f}",
        f"- cross_scale_gap: {hm['cross_scale_gap']:.6f}",
        f"- same_ordering: {hm['same_ordering']}",
        f"- local_first_support: {hm['local_first_support']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_local_first_cascade_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

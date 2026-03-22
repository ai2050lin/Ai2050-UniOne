from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage104_tensor_level_language_projection_rebuild_20260322"

DIMENSIONS = ["style", "logic", "syntax"]
PAIR_KEYS = ["style__logic", "style__syntax", "logic__syntax"]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_gap(left: float, right: float, scale: float) -> float:
    return _clip01(1.0 - abs(left - right) / scale)


def _pair_related_dimension(pair_key: str, dimension: str) -> bool:
    left, right = pair_key.split("__")
    return dimension in {left, right}


@lru_cache(maxsize=1)
def build_tensor_level_language_projection_rebuild_summary() -> dict:
    probe96 = _load_json(
        ROOT
        / "tests"
        / "codex_temp"
        / "deepseek7b_multidim_encoding_probe_natural96_all_support_20260319_1004"
        / "multidim_encoding_probe.json"
    )
    probe288 = _load_json(
        ROOT
        / "tests"
        / "codex_temp"
        / "deepseek7b_multidim_encoding_probe_natural288_all_support_20260319_1118"
        / "multidim_encoding_probe.json"
    )
    sparse = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_sparse_activation_region_analysis_20260320" / "summary.json"
    )["headline_metrics"]

    dimension_records = []
    q_values = []
    b_values = []
    g_values = []
    stability_terms = []
    separation_terms = []

    for dimension in DIMENSIONS:
        dim96 = probe96["dimensions"][dimension]
        dim288 = probe288["dimensions"][dimension]
        spec96 = probe96["specificity"][dimension]
        spec288 = probe288["specificity"][dimension]

        l2_mean = (dim96["mean_pair_delta_l2"] + dim288["mean_pair_delta_l2"]) / 2.0
        abs_mean = (dim96["mean_pair_delta_abs"] + dim288["mean_pair_delta_abs"]) / 2.0
        cosine_mean = (dim96["pair_delta_cosine_mean"] + dim288["pair_delta_cosine_mean"]) / 2.0
        layer_coverage_mean = (
            dim96["mass50_layer_coverage"]["covered_layer_ratio"]
            + dim288["mass50_layer_coverage"]["covered_layer_ratio"]
        ) / 2.0
        participation_mean = (
            dim96["rank_stats"]["participation_ratio"] + dim288["rank_stats"]["participation_ratio"]
        ) / 2.0
        specificity_mean = (spec96["specificity_margin"] + spec288["specificity_margin"]) / 2.0

        run_stability = _clip01(
            0.24 * _relative_gap(dim96["mean_pair_delta_l2"], dim288["mean_pair_delta_l2"], 80.0)
            + 0.18 * _relative_gap(dim96["mean_pair_delta_abs"], dim288["mean_pair_delta_abs"], 0.10)
            + 0.18 * _relative_gap(dim96["pair_delta_cosine_mean"], dim288["pair_delta_cosine_mean"], 0.10)
            + 0.18 * _relative_gap(spec96["specificity_margin"], spec288["specificity_margin"], 0.06)
            + 0.22 * _relative_gap(
                dim96["rank_stats"]["participation_ratio"],
                dim288["rank_stats"]["participation_ratio"],
                12.0,
            )
        )

        related_pairs = [pair_key for pair_key in PAIR_KEYS if _pair_related_dimension(pair_key, dimension)]
        separation_related = []
        alignment_related = []
        for pair_key in related_pairs:
            pair96 = probe96["cross_dimension"][pair_key]
            pair288 = probe288["cross_dimension"][pair_key]
            separation_related.append(
                1.0 - ((pair96["mass50_jaccard"] + pair288["mass50_jaccard"]) / 2.0)
            )
            alignment_related.append(
                (pair96["layer_profile_corr"] + pair288["layer_profile_corr"]) / 2.0
            )

        cross_separation = sum(separation_related) / len(separation_related)
        cross_alignment = sum(alignment_related) / len(alignment_related)

        l2_norm = _clip01(l2_mean / 700.0)
        abs_norm = _clip01(abs_mean / 0.70)
        specificity_norm = _clip01(specificity_mean / 0.35)
        participation_norm = _clip01(participation_mean / 40.0)

        q_reconstructed = _clip01(
            0.30 * cosine_mean
            + 0.26 * run_stability
            + 0.22 * layer_coverage_mean
            + 0.12 * sparse["sparse_feature_activation"]
            + 0.10 * sparse["sparse_seed_activation"]
        )
        b_reconstructed = _clip01(
            0.30 * specificity_norm
            + 0.24 * abs_norm
            + 0.20 * run_stability
            + 0.16 * sparse["sparse_structure_activation"]
            + 0.10 * participation_norm
        )
        g_reconstructed = _clip01(
            0.28 * l2_norm
            + 0.24 * cross_separation
            + 0.20 * cross_alignment
            + 0.16 * sparse["sparse_route_activation"]
            + 0.12 * run_stability
        )
        projection_strength = _clip01(
            0.30 * q_reconstructed
            + 0.28 * b_reconstructed
            + 0.26 * g_reconstructed
            + 0.16 * sparse["sparse_activation_efficiency"]
        )

        q_values.append(q_reconstructed)
        b_values.append(b_reconstructed)
        g_values.append(g_reconstructed)
        stability_terms.append(run_stability)
        separation_terms.append(cross_separation)

        dimension_records.append(
            {
                "dimension": dimension,
                "mean_pair_delta_l2": l2_mean,
                "mean_pair_delta_abs": abs_mean,
                "pair_delta_cosine_mean": cosine_mean,
                "specificity_margin_mean": specificity_mean,
                "layer_coverage_mean": layer_coverage_mean,
                "participation_ratio_mean": participation_mean,
                "run_stability": run_stability,
                "cross_separation": cross_separation,
                "cross_alignment": cross_alignment,
                "q_reconstructed": q_reconstructed,
                "b_reconstructed": b_reconstructed,
                "g_reconstructed": g_reconstructed,
                "projection_strength": projection_strength,
            }
        )

    reconstructed_context_gate_coherence = sum(q_values) / len(q_values)
    reconstructed_bias_transport = sum(b_values) / len(b_values)
    reconstructed_route_projection = sum(g_values) / len(g_values)
    cross_dimension_projection_stability = sum(stability_terms) / len(stability_terms)
    cross_dimension_separation = sum(separation_terms) / len(separation_terms)
    raw_language_projection_score = _clip01(
        0.24 * reconstructed_context_gate_coherence
        + 0.22 * reconstructed_bias_transport
        + 0.24 * reconstructed_route_projection
        + 0.16 * cross_dimension_projection_stability
        + 0.14 * cross_dimension_separation
    )

    return {
        "headline_metrics": {
            "reconstructed_context_gate_coherence": reconstructed_context_gate_coherence,
            "reconstructed_bias_transport": reconstructed_bias_transport,
            "reconstructed_route_projection": reconstructed_route_projection,
            "cross_dimension_projection_stability": cross_dimension_projection_stability,
            "cross_dimension_separation": cross_dimension_separation,
            "raw_language_projection_score": raw_language_projection_score,
        },
        "dimension_records": dimension_records,
        "foundation_sources": {
            "probe96": str(
                ROOT
                / "tests"
                / "codex_temp"
                / "deepseek7b_multidim_encoding_probe_natural96_all_support_20260319_1004"
                / "multidim_encoding_probe.json"
            ),
            "probe288": str(
                ROOT
                / "tests"
                / "codex_temp"
                / "deepseek7b_multidim_encoding_probe_natural288_all_support_20260319_1118"
                / "multidim_encoding_probe.json"
            ),
            "sparse_activation_summary": str(
                ROOT / "tests" / "codex_temp" / "stage56_sparse_activation_region_analysis_20260320" / "summary.json"
            ),
        },
        "status": {
            "status_short": (
                "tensor_level_language_projection_rebuild_ready"
                if raw_language_projection_score >= 0.70 and cross_dimension_projection_stability >= 0.80
                else "tensor_level_language_projection_rebuild_transition"
            ),
            "status_label": "语言投影链已切换到更原始的张量级探针底座，但还没有到原生实测定理层。",
        },
        "project_readout": {
            "summary": "这一步不再直接依赖手工 q/b/g 场景，而是用多维编码探针里的 style、logic、syntax 三维差分、特异性、层覆盖和跨维重叠来重建语言投影链。",
            "next_question": "下一步要把这条重建链继续和真实前后向轨迹对齐，避免它再次退回到高层摘要自洽。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage104 Tensor Level Language Projection Rebuild",
        "",
        f"- reconstructed_context_gate_coherence: {hm['reconstructed_context_gate_coherence']:.6f}",
        f"- reconstructed_bias_transport: {hm['reconstructed_bias_transport']:.6f}",
        f"- reconstructed_route_projection: {hm['reconstructed_route_projection']:.6f}",
        f"- cross_dimension_projection_stability: {hm['cross_dimension_projection_stability']:.6f}",
        f"- cross_dimension_separation: {hm['cross_dimension_separation']:.6f}",
        f"- raw_language_projection_score: {hm['raw_language_projection_score']:.6f}",
        f"- status_short: {summary['status']['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_tensor_level_language_projection_rebuild_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

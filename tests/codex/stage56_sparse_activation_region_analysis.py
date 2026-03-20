from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_sparse_activation_region_analysis_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_sparse_activation_region_summary() -> dict:
    region = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_region_topology_analysis_20260320" / "summary.json"
    )
    attr = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_region_attribute_analysis_20260320" / "summary.json"
    )
    lang = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_system_principles_20260320" / "summary.json"
    )

    hr = region["headline_metrics"]
    ha = attr["headline_metrics"]
    hl = lang["headline_metrics"]

    sparse_seed_activation = hl["language_entry_core"]
    sparse_feature_activation = hl["language_feature_core"] * (1.0 - hl["language_pressure_core"])
    sparse_structure_activation = hr["region_topology_margin"] * ha["attribute_distributed_score"]
    sparse_route_activation = hl["language_structure_core"] * ha["cross_region_attribute_strength"]
    sparse_activation_efficiency = (
        sparse_seed_activation
        + sparse_feature_activation
        + sparse_structure_activation
        + sparse_route_activation
    ) / 4.0

    return {
        "headline_metrics": {
            "sparse_seed_activation": sparse_seed_activation,
            "sparse_feature_activation": sparse_feature_activation,
            "sparse_structure_activation": sparse_structure_activation,
            "sparse_route_activation": sparse_route_activation,
            "sparse_activation_efficiency": sparse_activation_efficiency,
        },
        "sparse_equation": {
            "seed_term": "A_seed = E_lang",
            "feature_term": "A_feature = F_lang * (1 - P_lang)",
            "structure_term": "A_structure = M_region * D_attr",
            "route_term": "A_route = S_lang * X_attr",
            "efficiency_term": "A_sparse = mean(A_seed, A_feature, A_structure, A_route)",
        },
        "project_readout": {
            "summary": "每次只触发最少神经网络时，更像是局部种子区、特征区、结构区和跨区路由一起被最小组合激活，而不是整块区域一起点亮。",
            "next_question": "下一步要把这种稀疏触发模型继续推进到更原生的神经元与回路层，确认最小激活组合的真实边界。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 稀疏激活区域分析报告",
        "",
        f"- sparse_seed_activation: {hm['sparse_seed_activation']:.6f}",
        f"- sparse_feature_activation: {hm['sparse_feature_activation']:.6f}",
        f"- sparse_structure_activation: {hm['sparse_structure_activation']:.6f}",
        f"- sparse_route_activation: {hm['sparse_route_activation']:.6f}",
        f"- sparse_activation_efficiency: {hm['sparse_activation_efficiency']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_sparse_activation_region_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

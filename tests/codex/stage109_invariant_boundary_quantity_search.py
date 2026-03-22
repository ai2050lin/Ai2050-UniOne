from __future__ import annotations

import csv
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

from stage100_backfeed_suppression_hardening import build_backfeed_suppression_hardening_summary
from stage101_brain_evidence_joint_closure import build_brain_evidence_joint_closure_summary
from stage102_real_world_falsification_bridge import build_real_world_falsification_bridge_summary
from stage103_native_brain_anchor_search import build_native_brain_anchor_search_summary
from stage104_tensor_level_language_projection_rebuild import build_tensor_level_language_projection_rebuild_summary
from stage105_tensor_level_route_scale_rebuild import build_tensor_level_route_scale_rebuild_summary
from stage106_forward_backward_trace_rebuild import build_forward_backward_trace_rebuild_summary
from stage108_local_generative_law_catalog import build_local_generative_law_catalog_summary


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage109_invariant_boundary_quantity_search_20260322"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_category_counter(path: Path) -> Counter:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#"):
                continue
            rows.append(row)
    return Counter(row[1] for row in rows if len(row) > 1)


def _balance_score(counter: Counter) -> float:
    if not counter:
        return 0.0
    counts = list(counter.values())
    mean_count = sum(counts) / len(counts)
    deviation = sum(abs(value - mean_count) for value in counts) / len(counts)
    return _clip01(1.0 - deviation / max(1.0, mean_count))


@lru_cache(maxsize=1)
def build_invariant_boundary_quantity_search_summary() -> dict:
    projection = build_tensor_level_language_projection_rebuild_summary()["headline_metrics"]
    route = build_tensor_level_route_scale_rebuild_summary()["headline_metrics"]
    fb = build_forward_backward_trace_rebuild_summary()["headline_metrics"]
    backfeed = build_backfeed_suppression_hardening_summary()["headline_metrics"]
    joint = build_brain_evidence_joint_closure_summary()["headline_metrics"]
    bridge = build_real_world_falsification_bridge_summary()["headline_metrics"]
    anchors = build_native_brain_anchor_search_summary()["headline_metrics"]
    laws = build_local_generative_law_catalog_summary()["headline_metrics"]

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

    english_categories = _load_category_counter(ROOT / "tests" / "codex" / "deepseek7b_nouns_english_520_clean.csv")
    bilingual_categories = _load_category_counter(ROOT / "tests" / "codex" / "deepseek7b_bilingual_nouns_utf8.csv")

    english_balance = _balance_score(english_categories)
    bilingual_balance = _balance_score(bilingual_categories)
    bilingual_coverage = _clip01(len(bilingual_categories) / max(1, len(english_categories)))
    abstract_presence = _clip01(english_categories.get("abstract", 0) / max(1, sum(english_categories.values())) / 0.12)
    macro_gap_penalty = _clip01(1.0 - 0.5 * bilingual_coverage - 0.5 * abstract_presence)

    cosine_means = [
        probe96["dimensions"][dim]["pair_delta_cosine_mean"] for dim in ("style", "logic", "syntax")
    ] + [
        probe288["dimensions"][dim]["pair_delta_cosine_mean"] for dim in ("style", "logic", "syntax")
    ]
    layer_corrs = [
        probe96["cross_dimension"][pair]["layer_profile_corr"] for pair in probe96["cross_dimension"]
    ] + [
        probe288["cross_dimension"][pair]["layer_profile_corr"] for pair in probe288["cross_dimension"]
    ]
    relational_linearity_raw = _clip01(
        0.56 * _mean(cosine_means) + 0.44 * _mean(layer_corrs)
    )
    linearity_proof_gap = _clip01(1.0 - relational_linearity_raw * 0.72)

    quantity_records = [
        {
            "name": "hierarchical_concept_span_quantity",
            "kind": "invariant",
            "meaning": "概念在微观、中观、宏观三层之间保持可转运的层级跨度量。",
            "support": _clip01(
                0.20 * english_balance
                + 0.16 * bilingual_balance
                + 0.14 * bilingual_coverage
                + 0.16 * abstract_presence
                + 0.18 * projection["raw_language_projection_score"]
                + 0.16 * fb["raw_novelty_binding_capacity"]
            ),
        },
        {
            "name": "context_covariant_uniqueness_quantity",
            "kind": "invariant",
            "meaning": "在风格、逻辑、语法同时变化时，全局词选择仍收敛到单一输出的唯一性量。",
            "support": _clip01(
                0.22 * projection["reconstructed_context_gate_coherence"]
                + 0.20 * projection["cross_dimension_separation"]
                + 0.18 * projection["reconstructed_route_projection"]
                + 0.18 * route["route_structure_coupling_strength"]
                + 0.22 * fb["raw_forward_selectivity"]
            ),
        },
        {
            "name": "minimal_transport_efficiency_quantity",
            "kind": "invariant",
            "meaning": "最小传送与分布式沉降共同决定的全局效率量。",
            "support": _clip01(
                0.24 * route["distributed_network_support"]
                + 0.22 * route["degradation_tolerance"]
                + 0.18 * route["route_structure_coupling_strength"]
                + 0.18 * fb["frontier_boundary_coupling"]
                + 0.18 * anchors["generic_seed_recurrence_strength"]
            ),
        },
        {
            "name": "relational_linearity_quantity",
            "kind": "invariant",
            "meaning": "关系可线性搬运的残余结构量，用于承接词嵌入中的类比结构线索。",
            "support": _clip01(
                0.34 * relational_linearity_raw
                + 0.22 * projection["cross_dimension_projection_stability"]
                + 0.18 * route["route_scale_margin"] * 0.5 + 0.09
                + 0.14 * laws["law_composability_score"]
                + 0.12 * projection["cross_dimension_separation"]
            ),
        },
        {
            "name": "repair_stability_quantity",
            "kind": "invariant",
            "meaning": "及时学习与全局稳态同时成立时的修复稳态量。",
            "support": _clip01(
                0.24 * fb["raw_backward_fidelity"]
                + 0.22 * fb["loss_monotonicity"]
                + 0.18 * laws["law_failure_resilience"]
                + 0.18 * joint["field_observability_joint"]
                + 0.18 * (1.0 - backfeed["summary_backfeed_risk_after"])
            ),
        },
    ]

    boundary_records = [
        {
            "name": "macro_data_gap_boundary",
            "kind": "boundary",
            "pressure": macro_gap_penalty,
            "meaning": "当前原始数据对形容词、动词、抽象宏观概念覆盖仍然不够完整的边界。",
        },
        {
            "name": "evidence_boundary",
            "kind": "boundary",
            "pressure": _clip01(backfeed["summary_backfeed_risk_after"]),
            "meaning": "证据回灌仍然压过独立证据时的边界。",
        },
        {
            "name": "anchor_ambiguity_boundary",
            "kind": "boundary",
            "pressure": _clip01(anchors["anchor_ambiguity_penalty"]),
            "meaning": "脑锚点仍然共享过多维度时的边界。",
        },
        {
            "name": "task_bridge_boundary",
            "kind": "boundary",
            "pressure": _clip01(bridge["remaining_real_world_gap"]),
            "meaning": "真实任务语境桥仍然不足时的边界。",
        },
        {
            "name": "linearity_proof_boundary",
            "kind": "boundary",
            "pressure": linearity_proof_gap,
            "meaning": "线性关系仍然只是弱迹象，还没有进入强证明时的边界。",
        },
    ]

    strongest_quantity = max(quantity_records, key=lambda item: item["support"])
    weakest_quantity = min(quantity_records, key=lambda item: item["support"])
    highest_boundary = max(boundary_records, key=lambda item: item["pressure"])

    invariant_quantity_strength = _clip01(_mean(record["support"] for record in quantity_records))
    boundary_quantity_resilience = _clip01(1.0 - _mean(record["pressure"] for record in boundary_records))
    theory_breakthrough_readiness = _clip01(
        0.28 * invariant_quantity_strength
        + 0.22 * laws["law_composability_score"]
        + 0.18 * laws["law_failure_resilience"]
        + 0.16 * (1.0 - highest_boundary["pressure"])
        + 0.16 * (1.0 - macro_gap_penalty)
    )
    invariant_boundary_quantity_score = _clip01(
        0.34 * invariant_quantity_strength
        + 0.26 * boundary_quantity_resilience
        + 0.20 * theory_breakthrough_readiness
        + 0.20 * (1.0 - highest_boundary["pressure"])
    )

    return {
        "headline_metrics": {
            "invariant_quantity_strength": invariant_quantity_strength,
            "boundary_quantity_resilience": boundary_quantity_resilience,
            "theory_breakthrough_readiness": theory_breakthrough_readiness,
            "strongest_quantity_name": strongest_quantity["name"],
            "weakest_quantity_name": weakest_quantity["name"],
            "weakest_quantity_score": weakest_quantity["support"],
            "highest_boundary_name": highest_boundary["name"],
            "highest_boundary_pressure": highest_boundary["pressure"],
            "invariant_boundary_quantity_score": invariant_boundary_quantity_score,
        },
        "quantity_records": quantity_records,
        "boundary_records": boundary_records,
        "foundation_sources": {
            "english_nouns": str(ROOT / "tests" / "codex" / "deepseek7b_nouns_english_520_clean.csv"),
            "bilingual_nouns": str(ROOT / "tests" / "codex" / "deepseek7b_bilingual_nouns_utf8.csv"),
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
        },
        "status": {
            "status_short": (
                "invariant_boundary_quantity_search_ready"
                if invariant_boundary_quantity_score >= 0.55 and weakest_quantity["support"] >= 0.45
                else "invariant_boundary_quantity_search_transition"
            ),
            "status_label": "守恒量与边界量候选已经开始成形，但当前最高边界压力与宏观数据缺口仍然限制新数学理论突破。",
        },
        "project_readout": {
            "summary": "这一轮把概念层级跨度、上下文协变唯一性、最小传送效率、关系线性性、修复稳态性压成候选量，同时把宏观数据缺口、证据边界、任务桥缺口等失败边界显式钉出来。",
            "next_question": "下一步要用真实任务反例去打这些候选量，判断哪些能进入理论主核，哪些只是假象。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage109 Invariant Boundary Quantity Search",
        "",
        f"- invariant_quantity_strength: {hm['invariant_quantity_strength']:.6f}",
        f"- boundary_quantity_resilience: {hm['boundary_quantity_resilience']:.6f}",
        f"- theory_breakthrough_readiness: {hm['theory_breakthrough_readiness']:.6f}",
        f"- strongest_quantity_name: {hm['strongest_quantity_name']}",
        f"- weakest_quantity_name: {hm['weakest_quantity_name']}",
        f"- weakest_quantity_score: {hm['weakest_quantity_score']:.6f}",
        f"- highest_boundary_name: {hm['highest_boundary_name']}",
        f"- highest_boundary_pressure: {hm['highest_boundary_pressure']:.6f}",
        f"- invariant_boundary_quantity_score: {hm['invariant_boundary_quantity_score']:.6f}",
        f"- status_short: {summary['status']['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_invariant_boundary_quantity_search_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

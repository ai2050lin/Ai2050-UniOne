from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_icspb_object_remapping_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_icspb_object_remapping_summary() -> dict:
    concept_v5 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_concept_formation_closed_form_v5_20260320" / "summary.json"
    )
    attribute_native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_attribute_fiber_nativeization_20260320" / "summary.json"
    )
    apple_banana = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_apple_banana_encoding_transfer_20260320" / "summary.json"
    )

    hc = concept_v5["headline_metrics"]
    ha = attribute_native["headline_metrics"]
    hab = apple_banana["headline_metrics"]

    family_patch_to_structure = hc["anchor_chart_term_v5"]
    concept_offset_to_feature = hc["local_primary_term_v5"]
    attribute_fiber_to_feature = (
        ha["mean_local_bundle_strength"]
        + abs(ha["apple_round_local_coeff"])
        + abs(ha["apple_elongated_local_coeff"])
    )
    relation_context_to_transport = hab["pred_vs_banana_cosine"] - hab["pred_vs_cat_cosine"]

    family_patch_alignment = family_patch_to_structure / (1.0 + family_patch_to_structure)
    concept_offset_alignment = concept_offset_to_feature / (1.0 + concept_offset_to_feature)
    attribute_fiber_alignment = attribute_fiber_to_feature / (1.0 + attribute_fiber_to_feature)
    relation_context_alignment = relation_context_to_transport / (1.0 + relation_context_to_transport)
    remap_consistency = (
        family_patch_alignment
        + concept_offset_alignment
        + attribute_fiber_alignment
        + relation_context_alignment
    ) / 4.0

    return {
        "headline_metrics": {
            "family_patch_to_structure": family_patch_to_structure,
            "concept_offset_to_feature": concept_offset_to_feature,
            "attribute_fiber_to_feature": attribute_fiber_to_feature,
            "relation_context_to_transport": relation_context_to_transport,
            "family_patch_alignment": family_patch_alignment,
            "concept_offset_alignment": concept_offset_alignment,
            "attribute_fiber_alignment": attribute_fiber_alignment,
            "relation_context_alignment": relation_context_alignment,
            "remap_consistency": remap_consistency,
        },
        "mapping": {
            "family_patch": "映射到 family anchor + local chart + structure layer",
            "concept_section_concept_offset": "映射到 concept state + local differential feature",
            "attribute_fiber": "映射到 local feature fiber and differential feature layer",
            "relation_context_fiber": "映射到 transport-conditioned structure routing",
            "admissible_update": "映射到 pressure-gated learning update",
            "restricted_readout": "映射到 locked feature readout",
            "stage_conditioned_transport": "映射到 learning-phase transport",
            "successor_aligned_transport": "映射到 feedback-aligned successor transport",
            "protocol_bridge": "映射到 structure-to-readout protocol bridge",
        },
        "project_readout": {
            "summary": "旧版 ICSPB 对象没有被新主线整体推翻，而是被重新分配到特征层、结构层、学习层和压力层之中。当前最稳的是 family patch / concept offset / attribute fiber 仍然保留对象层地位，transport/readout/bridge 则退到二级运输和执行层。",
            "next_question": "下一步要把 transport/readout/bridge 三组对象正式并回当前 M_encoding 主核，而不是让两套理论长期并行。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 旧版 ICSPB 对象重映射报告",
        "",
        f"- family_patch_to_structure: {hm['family_patch_to_structure']:.6f}",
        f"- concept_offset_to_feature: {hm['concept_offset_to_feature']:.6f}",
        f"- attribute_fiber_to_feature: {hm['attribute_fiber_to_feature']:.6f}",
        f"- relation_context_to_transport: {hm['relation_context_to_transport']:.6f}",
        f"- remap_consistency: {hm['remap_consistency']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_icspb_object_remapping_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

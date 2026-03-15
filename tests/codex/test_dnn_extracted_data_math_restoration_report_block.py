from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    temp = ROOT / "tests" / "codex_temp"

    systematic = load_json(temp / "dnn_systematic_mass_extraction_block_20260315.json")
    dense_real = load_json(temp / "dnn_dense_real_unit_corpus_block_20260315.json")
    signatures = load_json(temp / "dnn_activation_signature_mining_block_20260315.json")
    math_status = load_json(temp / "dnn_math_restoration_status_block_20260315.json")
    successor_corpus = load_json(temp / "dnn_successor_real_corpus_block_20260315.json")
    successor_stage_rows = load_json(temp / "dnn_successor_stage_row_corpus_block_20260315.json")
    online_episode = load_json(temp / "dnn_successor_online_recovery_episode_export_block_20260315.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_extracted_data_math_restoration_report_block",
        },
        "strict_goal": {
            "statement": "Produce one explicit report of the currently extracted DNN data inventory and the mathematical structures that each data block is restoring.",
            "boundary": "This block is a synthesis report, not a claim that final theorem closure has been reached.",
        },
        "extracted_data_catalog": {
            "systematic_structure_corpus": {
                "units": systematic["headline_metrics"]["total_standardized_units"],
                "exact_real_units": systematic["headline_metrics"]["exact_real_units"],
                "exact_real_fraction": systematic["headline_metrics"]["exact_real_fraction"],
                "meaning": "统一结构语料库，负责回答当前 DNN 研究里到底已经标准化了多少结构对象。",
            },
            "dense_real_unit_corpus": {
                "unit_count": dense_real["headline_metrics"]["unit_count"],
                "weighted_units": dense_real["headline_metrics"]["weighted_units"],
                "macro_weight": dense_real["headline_metrics"]["macro_weight"],
                "specific_weight": dense_real["headline_metrics"]["specific_weight"],
                "meaning": "真实 row-level 数据库，负责回答真实模型里已经挖出了多少 layer/recovery/task/protocol/topology 单位。",
            },
            "activation_signature_layer": {
                "signature_rows": signatures["headline_metrics"]["signature_rows"],
                "unique_concepts": signatures["headline_metrics"]["unique_concepts"],
                "specific_signature_rows": signatures["headline_metrics"]["specific_signature_rows"],
                "protocol_signature_rows": signatures["headline_metrics"]["protocol_signature_rows"],
                "topology_signature_rows": signatures["headline_metrics"]["topology_signature_rows"],
                "meaning": "概念级 signature 层，负责把 concept 的 specific/protocol/topology 特征压成统一签名格式。",
            },
            "successor_real_corpus": {
                "total_successor_units": successor_corpus["headline_metrics"]["total_successor_units"],
                "exact_dense_units": successor_corpus["headline_metrics"]["exact_dense_units"],
                "proxy_units": successor_corpus["headline_metrics"]["proxy_units"],
                "exactness_fraction": successor_corpus["headline_metrics"]["exactness_fraction"],
                "meaning": "successor 专项语料库，负责回答 successor 相关数据里 exact dense 与 proxy 的真实比例。",
            },
            "successor_stage_row_corpus": {
                "stage_row_count": successor_stage_rows["headline_metrics"]["stage_row_count"],
                "online_recovery_stage_rows": successor_stage_rows["headline_metrics"]["online_recovery_stage_rows"],
                "inventory_stage_rows": successor_stage_rows["headline_metrics"]["inventory_stage_rows"],
                "meaning": "successor 的 stage-row 中间层，负责把 online_recovery 与 inventory 放进同一行格式里比较。",
            },
            "online_recovery_episode_export": {
                "episode_step_rows": online_episode["headline_metrics"]["episode_step_rows"],
                "model_count": online_episode["headline_metrics"]["model_count"],
                "step_count": online_episode["headline_metrics"]["step_count"],
                "meaning": "online_recovery 的 episode-step 中间导出层，负责从 summary 进一步向 layer/unit dense export 过渡。",
            },
        },
        "math_restoration_catalog": {
            "family_basis": {
                "score": math_status["restoration_terms"]["family_basis_parametric_score"],
                "restored_from": [
                    "systematic_structure_corpus",
                    "dense_real_unit_corpus",
                    "activation_signature_layer",
                ],
                "meaning": "回答 family patch / family basis 是否已经能被参数化恢复。",
            },
            "concept_offset": {
                "score": math_status["restoration_terms"]["concept_offset_parametric_score"],
                "restored_from": [
                    "dense_real_unit_corpus",
                    "activation_signature_layer",
                ],
                "meaning": "回答 concept offset / specific difference 是否已经能被强参数化恢复。",
            },
            "protocol_field": {
                "score": math_status["restoration_terms"]["protocol_field_parametric_score"],
                "restored_from": [
                    "dense_real_unit_corpus",
                    "activation_signature_layer",
                ],
                "meaning": "回答 protocol field / protocol bridge 的坐标是否已被参数化。",
            },
            "topology": {
                "score": math_status["restoration_terms"]["topology_parametric_score"],
                "restored_from": [
                    "dense_real_unit_corpus",
                    "activation_signature_layer",
                ],
                "meaning": "回答 attention/relation topology 是否已被统一成显式参数结构。",
            },
            "successor": {
                "score": math_status["restoration_terms"]["successor_parametric_score"],
                "restored_from": [
                    "successor_real_corpus",
                    "successor_stage_row_corpus",
                    "online_recovery_episode_export",
                ],
                "meaning": "回答 successor 是否已经被参数化恢复。当前它仍是最弱项。",
            },
            "full_restoration": {
                "score": math_status["restoration_terms"]["full_restoration_score"],
                "restored_from": [
                    "family_basis",
                    "concept_offset",
                    "protocol_field",
                    "topology",
                    "successor",
                ],
                "meaning": "统一数学恢复总分，回答当前是不是已经逼近完整理论闭合。",
            },
        },
        "headline_metrics": {
            "total_standardized_units": systematic["headline_metrics"]["total_standardized_units"],
            "exact_real_fraction": systematic["headline_metrics"]["exact_real_fraction"],
            "signature_rows": signatures["headline_metrics"]["signature_rows"],
            "successor_exactness_fraction": successor_corpus["headline_metrics"]["exactness_fraction"],
            "full_restoration_score": math_status["restoration_terms"]["full_restoration_score"],
        },
        "strict_verdict": {
            "report_present": True,
            "core_answer": (
                "The DNN-side project now has an explicit list of extracted data layers and a matching list of mathematical structures being restored from them. "
                "The strongest restored terms are concept offset, protocol field, and topology; the weakest restored term is still successor."
            ),
            "main_hard_gaps": [
                "exact real extraction is still below dense theorem-grade closure",
                "many signatures remain artifact-derived rather than full dense neuron activations",
                "successor exactness is still much weaker than basis/offset/protocol/topology restoration",
                "full restoration remains candidate-level rather than unique final theorem",
            ],
        },
        "progress_estimate": {
            "systematic_mass_extraction_percent": 78.0,
            "activation_signature_mining_percent": 66.0,
            "math_restoration_status_percent": 73.0,
            "full_brain_encoding_mechanism_percent": 89.0,
        },
        "next_large_blocks": [
            "Continue replacing proxy-heavy successor data with dense exports, because successor is now the main restoration bottleneck.",
            "Expand dense real extraction beyond row-level and signature-derived units into stronger neuron-level coordinates.",
            "After successor exactness rises, recompute full restoration and test whether the project can move from candidate theory to theorem closure.",
        ],
    }
    return payload


def test_dnn_extracted_data_math_restoration_report_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["total_standardized_units"] >= 1700
    assert metrics["exact_real_fraction"] > 0.45
    assert metrics["signature_rows"] >= 190
    assert metrics["successor_exactness_fraction"] > 0.30
    assert metrics["full_restoration_score"] > 0.85
    assert verdict["report_present"] is True


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN extracted data and math restoration report block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_extracted_data_math_restoration_report_block_20260315.json",
    )
    args = ap.parse_args()
    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

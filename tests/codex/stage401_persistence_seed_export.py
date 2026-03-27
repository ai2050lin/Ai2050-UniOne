from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STAMP = datetime.now().strftime("%Y%m%d")
OUT_DIR = ROOT / "tests" / "codex_temp" / f"stage401_persistence_seed_export_{STAMP}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = ROOT / "research" / "gpt5" / "data"
RAW_DIR = DATA_ROOT / "raw"
ROWS_DIR = DATA_ROOT / "rows"
ANALYSIS_DIR = DATA_ROOT / "analysis"
THEORY_DIR = DATA_ROOT / "theory_candidates"
VIS_DIR = DATA_ROOT / "visualization"
AUDIT_DIR = DATA_ROOT / "audit"


OVERLAY = {
    "static_encoding": {
        "label": "静态编码层",
        "color": "#60a5fa",
        "nodes": [
            {"id": "static-d138", "layer": 5, "neuron": 138, "dimIndex": 138, "metric": "shared_base_strength", "value": 0.4784, "strength": 0.92, "category": "共享承载", "sourceStage": "stage375", "outputDir": "tests/codex_temp/stage375_shared_carrier_subchain_extractor_20260325", "parameterIds": ["dim_138", "carrier_dim_138"]},
            {"id": "static-d306", "layer": 8, "neuron": 306, "dimIndex": 306, "metric": "shared_base_strength", "value": 0.4412, "strength": 0.84, "category": "共享承载", "sourceStage": "stage375", "outputDir": "tests/codex_temp/stage375_shared_carrier_subchain_extractor_20260325", "parameterIds": ["dim_306", "carrier_dim_306"]},
            {"id": "static-d660", "layer": 11, "neuron": 660, "dimIndex": 660, "metric": "shared_base_strength", "value": 0.3927, "strength": 0.78, "category": "共享承载", "sourceStage": "stage375", "outputDir": "tests/codex_temp/stage375_shared_carrier_subchain_extractor_20260325", "parameterIds": ["dim_660", "carrier_dim_660"]},
        ],
        "chains": [["static-d138", "static-d306"], ["static-d306", "static-d660"]],
    },
    "dynamic_route": {
        "label": "动态路径层",
        "color": "#f97316",
        "nodes": [
            {"id": "route-d5", "layer": 6, "neuron": 5, "dimIndex": 5, "metric": "task_bias_strength", "value": 0.571, "strength": 0.95, "category": "任务偏转", "sourceStage": "stage378", "outputDir": "tests/codex_temp/stage378_task_bias_dedicated_extractor_20260325", "parameterIds": ["dim_5", "bias_dim_5"]},
            {"id": "route-d215", "layer": 10, "neuron": 215, "dimIndex": 215, "metric": "object_domain_shift", "value": 0.4639, "strength": 0.76, "category": "对象域切换", "sourceStage": "stage376", "outputDir": "tests/codex_temp/stage376_bias_deflection_subchain_extractor_20260325", "parameterIds": ["dim_215", "bias_dim_215"]},
            {"id": "route-d364", "layer": 14, "neuron": 364, "dimIndex": 364, "metric": "competition_strength", "value": 0.6521, "strength": 0.88, "category": "类内竞争", "sourceStage": "stage376", "outputDir": "tests/codex_temp/stage376_bias_deflection_subchain_extractor_20260325", "parameterIds": ["dim_364", "bias_dim_364"]},
        ],
        "chains": [["route-d5", "route-d215"], ["route-d215", "route-d364"]],
    },
    "result_recovery": {
        "label": "结果回收层",
        "color": "#22c55e",
        "nodes": [
            {"id": "recovery-d469", "layer": 17, "neuron": 469, "dimIndex": 469, "metric": "amplification_strength", "value": 0.2246, "strength": 0.74, "category": "中层主放大", "sourceStage": "stage377", "outputDir": "tests/codex_temp/stage377_amplification_subchain_extractor_20260325", "parameterIds": ["dim_469", "amplify_dim_469"]},
            {"id": "recovery-d530", "layer": 22, "neuron": 530, "dimIndex": 530, "metric": "amplification_strength", "value": 0.3267, "strength": 0.91, "category": "后层持续放大", "sourceStage": "stage377", "outputDir": "tests/codex_temp/stage377_amplification_subchain_extractor_20260325", "parameterIds": ["dim_530", "amplify_dim_530"]},
            {"id": "recovery-d273", "layer": 25, "neuron": 273, "dimIndex": 273, "metric": "net_gain", "value": 0.1685, "strength": 0.68, "category": "独立放大核", "sourceStage": "stage357", "outputDir": "tests/codex_temp/stage357_independent_amplification_net_gain_review_20260325", "parameterIds": ["dim_273", "amplify_dim_273"]},
        ],
        "chains": [["recovery-d469", "recovery-d530"], ["recovery-d530", "recovery-d273"]],
    },
    "propagation_encoding": {
        "label": "传播编码层",
        "color": "#38bdf8",
        "nodes": [
            {"id": "prop-d138", "layer": 4, "neuron": 138, "dimIndex": 138, "metric": "relay_start", "value": 0.1922, "strength": 0.7, "category": "第一次放大", "sourceStage": "stage340", "outputDir": "tests/codex_temp/stage340_layerwise_relay_stitching_review_20260324", "parameterIds": ["dim_138", "relay_dim_138"]},
            {"id": "prop-d5", "layer": 13, "neuron": 5, "dimIndex": 5, "metric": "relay_mid", "value": 0.2246, "strength": 0.82, "category": "中层主放大", "sourceStage": "stage340", "outputDir": "tests/codex_temp/stage340_layerwise_relay_stitching_review_20260324", "parameterIds": ["dim_5", "relay_dim_5"]},
            {"id": "prop-d530", "layer": 24, "neuron": 530, "dimIndex": 530, "metric": "relay_end", "value": 0.3267, "strength": 0.93, "category": "后层持续放大", "sourceStage": "stage340", "outputDir": "tests/codex_temp/stage340_layerwise_relay_stitching_review_20260324", "parameterIds": ["dim_530", "relay_dim_530"]},
        ],
        "chains": [["prop-d138", "prop-d5"], ["prop-d5", "prop-d530"]],
    },
    "semantic_roles": {
        "label": "语义角色层",
        "color": "#a78bfa",
        "nodes": [
            {"id": "role-d306", "layer": 7, "neuron": 306, "dimIndex": 306, "metric": "role_alignment", "value": 0.3784, "strength": 0.72, "category": "对象", "sourceStage": "stage337", "outputDir": "tests/codex_temp/stage337_multi_space_role_raw_alignment_20260324", "parameterIds": ["dim_306", "role_dim_306"]},
            {"id": "role-d364", "layer": 12, "neuron": 364, "dimIndex": 364, "metric": "role_alignment", "value": 0.3292, "strength": 0.58, "category": "属性", "sourceStage": "stage351", "outputDir": "tests/codex_temp/stage351_attribute_position_operation_hardening_review_20260325", "parameterIds": ["dim_364", "role_dim_364"]},
            {"id": "role-d469", "layer": 18, "neuron": 469, "dimIndex": 469, "metric": "role_alignment", "value": 0.3676, "strength": 0.63, "category": "操作", "sourceStage": "stage351", "outputDir": "tests/codex_temp/stage351_attribute_position_operation_hardening_review_20260325", "parameterIds": ["dim_469", "role_dim_469"]},
        ],
        "chains": [["role-d306", "role-d364"], ["role-d364", "role-d469"]],
    },
}


def ensure_dirs() -> None:
    for path in [RAW_DIR, ROWS_DIR, ANALYSIS_DIR, THEORY_DIR, VIS_DIR, AUDIT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def export_visualization() -> Path:
    path = VIS_DIR / "layer_parameter_state_overlay_v1.json"
    path.write_text(json.dumps(OVERLAY, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_raw_events() -> Path:
    rows = []
    for layer_key, block in OVERLAY.items():
        for node in block["nodes"]:
            rows.append({
                "event_id": f"raw_event_{node['id'].replace('-', '_')}",
                "layer_key": layer_key,
                "layer_index": node["layer"],
                "token_index": None,
                "neuron_index": node["neuron"],
                "dim_index": node["dimIndex"],
                "parameter_ids": node["parameterIds"],
                "activation_value": node["value"],
                "strength": node["strength"],
                "category": node["category"],
                "source_stage": node["sourceStage"],
                "output_dir": node["outputDir"],
            })
    path = RAW_DIR / "layer_parameter_state_events_v1.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return path


def export_rows() -> Path:
    rows = []
    for layer_key, block in OVERLAY.items():
        for node in block["nodes"]:
            rows.append({
                "row_id": f"research_row_{node['id'].replace('-', '_')}",
                "category": node["category"],
                "field_signature": "layer_index+neuron_index+dim_index+parameter_ids+metric+value",
                "layer_key": layer_key,
                "source_stage": node["sourceStage"],
                "source_output_dir": node["outputDir"],
                "source_event_id": f"raw_event_{node['id'].replace('-', '_')}",
            })
    path = ROWS_DIR / "layer_parameter_state_rows_v1.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return path


def export_analysis() -> Path:
    summary = {
        "analysis_id": "layer_parameter_state_analysis_v1",
        "profile_count": len(OVERLAY),
        "node_count": sum(len(block["nodes"]) for block in OVERLAY.values()),
        "chain_count": sum(len(block["chains"]) for block in OVERLAY.values()),
        "profiles": {
            key: {
                "label": value["label"],
                "node_count": len(value["nodes"]),
                "chain_count": len(value["chains"]),
            }
            for key, value in OVERLAY.items()
        },
    }
    path = ANALYSIS_DIR / "layer_parameter_state_analysis_v1.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_audit(visual_path: Path, raw_path: Path, row_path: Path, analysis_path: Path) -> Path:
    audit = {
        "audit_id": "layer_parameter_state_seed_export_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "tests/codex/stage401_persistence_seed_export.py",
        "linked_outputs": [
            str(visual_path.relative_to(ROOT)),
            str(raw_path.relative_to(ROOT)),
            str(row_path.relative_to(ROOT)),
            str(analysis_path.relative_to(ROOT)),
        ],
    }
    path = AUDIT_DIR / "layer_parameter_state_audit_v1.json"
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    ensure_dirs()
    visual_path = export_visualization()
    raw_path = export_raw_events()
    row_path = export_rows()
    analysis_path = export_analysis()
    audit_path = export_audit(visual_path, raw_path, row_path, analysis_path)
    summary = {
        "stage": "stage401",
        "persisted_dirs": [
            str(RAW_DIR.relative_to(ROOT)),
            str(ROWS_DIR.relative_to(ROOT)),
            str(ANALYSIS_DIR.relative_to(ROOT)),
            str(THEORY_DIR.relative_to(ROOT)),
            str(VIS_DIR.relative_to(ROOT)),
            str(AUDIT_DIR.relative_to(ROOT)),
        ],
        "persisted_files": [
            str(visual_path.relative_to(ROOT)),
            str(raw_path.relative_to(ROOT)),
            str(row_path.relative_to(ROOT)),
            str(analysis_path.relative_to(ROOT)),
            str(audit_path.relative_to(ROOT)),
        ],
        "node_count": sum(len(block["nodes"]) for block in OVERLAY.values()),
        "chain_count": sum(len(block["chains"]) for block in OVERLAY.values()),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

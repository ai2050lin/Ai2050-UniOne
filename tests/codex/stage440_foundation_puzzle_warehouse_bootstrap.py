from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "research" / "gpt5" / "data"
AUDIT_ROOT = DATA_ROOT / "audit"
PUZZLE_ROOT = DATA_ROOT / "puzzle_warehouse"
FRONTEND_DATA_ROOT = ROOT / "frontend" / "src" / "blueprint" / "data"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage440_foundation_puzzle_warehouse_bootstrap_{datetime.now().strftime('%Y%m%d')}"

RAW_PATH = DATA_ROOT / "raw" / "layer_parameter_state_events_v1.jsonl"
ROWS_PATH = DATA_ROOT / "rows" / "layer_parameter_state_rows_v1.jsonl"
ANALYSIS_PATH = DATA_ROOT / "analysis" / "layer_parameter_state_analysis_v1.json"
VIS_PATH = DATA_ROOT / "visualization" / "layer_parameter_state_overlay_v1.json"
AUDIT_PATH = AUDIT_ROOT / "layer_parameter_state_audit_v1.json"

PUZZLE_MANIFEST_PATH = PUZZLE_ROOT / "puzzle_warehouse_manifest_v1.json"
PUZZLE_TEMPLATE_PATH = PUZZLE_ROOT / "puzzle_record_template_v1.json"
PUZZLE_AXES_PATH = PUZZLE_ROOT / "puzzle_focus_axes_v1.json"
PUZZLE_VIEW_REGISTRY_PATH = PUZZLE_ROOT / "puzzle_view_registry_v1.json"
PUZZLE_RECORDS_PATH = PUZZLE_ROOT / "puzzle_records_v1.jsonl"
REPAIR_REPLAY_SLOTS_PATH = PUZZLE_ROOT / "repair_replay_sample_slots_v1.json"

DATA_CATALOG_PATH = AUDIT_ROOT / "data_catalog_v1.json"
ENTITY_REGISTRY_PATH = AUDIT_ROOT / "entity_registry_v1.json"
PERSISTED_DATA_CATALOG_PATH = FRONTEND_DATA_ROOT / "persisted_data_catalog_v1.js"
PERSISTED_ENTITY_REGISTRY_PATH = FRONTEND_DATA_ROOT / "persisted_entity_registry_v1.js"
PERSISTED_PUZZLE_RECORDS_PATH = FRONTEND_DATA_ROOT / "persisted_puzzle_records_v1.js"
PERSISTED_REPAIR_REPLAY_SLOTS_PATH = FRONTEND_DATA_ROOT / "persisted_repair_replay_sample_slots_v1.js"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def file_timestamp(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def count_json_file_records(file_id: str, payload: dict) -> tuple[int, dict]:
    extra: dict = {}
    if file_id == "analysis_v1":
        extra["profile_count"] = payload.get("profile_count", 0)
        extra["chain_count"] = payload.get("chain_count", 0)
        return payload.get("node_count", 0), extra
    if file_id == "visualization_overlay_v1":
        profile_count = 0
        node_count = 0
        chain_count = 0
        for profile in payload.values():
            if isinstance(profile, dict):
                profile_count += 1
                node_count += len(profile.get("nodes", []))
                chain_count += len(profile.get("chains", []))
        extra["profile_count"] = profile_count
        extra["chain_count"] = chain_count
        return node_count, extra
    if file_id == "audit_v1":
        return len(payload.get("outputs", [])), extra
    if file_id == "puzzle_warehouse_manifest_v1":
        return len(payload.get("puzzle_types", [])), extra
    if file_id == "puzzle_record_template_v1":
        return len(payload.get("required_fields", [])), extra
    if file_id == "puzzle_focus_axes_v1":
        return len(payload.get("axes", [])), extra
    if file_id == "puzzle_view_registry_v1":
        return len(payload.get("views", [])), extra
    if file_id == "repair_replay_sample_slots_v1":
        extra["status_count"] = len(payload.get("status_legend", []))
        return len(payload.get("slots", [])), extra
    return len(payload), extra


def build_data_catalog() -> dict:
    file_specs = [
        {
            "id": "raw_events_v1",
            "label": "原始事件数据",
            "layer": "原始抓取层",
            "path": RAW_PATH,
            "description": "参数级原始事件，记录层号、神经元、维度、参数位、激活值与来源阶段。",
            "field_examples": [
                "event_id",
                "layer_index",
                "neuron_index",
                "dim_index",
                "parameter_ids",
                "activation_value",
                "source_stage",
            ],
        },
        {
            "id": "research_rows_v1",
            "label": "研究结构行",
            "layer": "研究结构层",
            "path": ROWS_PATH,
            "description": "把原始事件整理成研究结构行，保留类别、字段签名和来源链。",
            "field_examples": [
                "row_id",
                "category",
                "field_signature",
                "layer_key",
                "source_stage",
                "source_event_id",
            ],
        },
        {
            "id": "analysis_v1",
            "label": "基础分析结果",
            "layer": "分析结果层",
            "path": ANALYSIS_PATH,
            "description": "汇总参数节点数、链路数、剖面数和主要分类。",
            "field_examples": [
                "profile_count",
                "node_count",
                "chain_count",
                "layers",
                "categories",
            ],
        },
        {
            "id": "visualization_overlay_v1",
            "label": "可视化叠加数据",
            "layer": "可视化资产层",
            "path": VIS_PATH,
            "description": "前端当前使用的参数态节点、链路和层级叠加信息。",
            "field_examples": [
                "label",
                "color",
                "nodes",
                "chains",
                "sourceDataPath",
            ],
        },
        {
            "id": "audit_v1",
            "label": "审计溯源数据",
            "layer": "审计与溯源层",
            "path": AUDIT_PATH,
            "description": "记录脚本、生成时间、输出文件和源文件映射。",
            "field_examples": [
                "script_path",
                "generated_at",
                "outputs",
                "source_files",
            ],
        },
        {
            "id": "puzzle_warehouse_manifest_v1",
            "label": "拼图仓总览",
            "layer": "拼图仓总览层",
            "path": PUZZLE_MANIFEST_PATH,
            "description": "定义拼图仓主入口、默认优先轴、默认视图和拼图类型。",
            "field_examples": [
                "warehouse_id",
                "records_path",
                "default_priority_axes",
                "default_views",
                "puzzle_types",
            ],
        },
        {
            "id": "puzzle_records_v1",
            "label": "拼图记录数据",
            "layer": "拼图记录层",
            "path": PUZZLE_RECORDS_PATH,
            "description": "基础拼图仓的核心记录，保存裂缝、反例、修复候选和下一步动作。",
            "field_examples": [
                "puzzle_id",
                "puzzle_type",
                "priority_axis",
                "layer_scope",
                "mapped_variables",
                "next_action",
            ],
        },
        {
            "id": "puzzle_record_template_v1",
            "label": "拼图记录模板",
            "layer": "拼图模板层",
            "path": PUZZLE_TEMPLATE_PATH,
            "description": "规定每条拼图记录必须包含的字段和示例写法。",
            "field_examples": [
                "required_fields",
                "field_notes",
                "example_record",
            ],
        },
        {
            "id": "puzzle_focus_axes_v1",
            "label": "优先轴配置",
            "layer": "优先轴配置层",
            "path": PUZZLE_AXES_PATH,
            "description": "固定当前最值得持续推进的研究主线和目标问题。",
            "field_examples": [
                "axes",
                "priority",
                "reason",
                "target_questions",
            ],
        },
        {
            "id": "puzzle_view_registry_v1",
            "label": "拼图视图注册表",
            "layer": "拼图可视化层",
            "path": PUZZLE_VIEW_REGISTRY_PATH,
            "description": "定义基础总览图、对象对比图、运行回放图、裂缝地图和证据关系图。",
            "field_examples": [
                "views",
                "purpose",
                "default_layers",
                "default_priority_axes",
            ],
        },
        {
            "id": "repair_replay_sample_slots_v1",
            "label": "修复回放槽位",
            "layer": "样本回放槽位层",
            "path": REPAIR_REPLAY_SLOTS_PATH,
            "description": "为修复前后对照预留真实样本回放槽位、缺失资产和验证目标。",
            "field_examples": [
                "slot_id",
                "repair_puzzle_id",
                "baseline_puzzle_id",
                "sample_id",
                "phase_slots",
                "missing_assets",
            ],
        },
    ]

    entries = []
    for spec in file_specs:
        path = spec["path"]
        entry = {
            "id": spec["id"],
            "label": spec["label"],
            "layer": spec["layer"],
            "relative_path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "last_modified": file_timestamp(path),
            "description": spec["description"],
            "field_examples": spec["field_examples"],
        }
        if path.suffix == ".jsonl":
            entry["record_count"] = len(load_jsonl(path))
        elif path.suffix == ".json":
            payload = load_json(path)
            entry["record_count"], extra = count_json_file_records(spec["id"], payload)
            entry.update(extra)
        entries.append(entry)

    catalog = {
        "catalog_id": "data_catalog_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(DATA_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "entry_count": len(entries),
        "entries": entries,
    }
    DATA_CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return catalog


def write_persisted_data_catalog(catalog: dict) -> None:
    items = []
    for entry in catalog["entries"]:
        items.append(
            {
                "id": entry["id"],
                "label": entry["label"],
                "layer": entry["layer"],
                "relativePath": entry["relative_path"],
                "recordCount": entry.get("record_count", 0),
                "description": entry["description"],
                "fields": entry["field_examples"],
            }
        )
    js = "export const PERSISTED_DATA_CATALOG_V1 = " + json.dumps(items, ensure_ascii=False, indent=2) + ";\n"
    PERSISTED_DATA_CATALOG_PATH.write_text(js, encoding="utf-8")


def build_entity_registry() -> dict:
    raw_events = load_jsonl(RAW_PATH)
    research_rows = load_jsonl(ROWS_PATH)
    analysis = load_json(ANALYSIS_PATH)
    visualization = load_json(VIS_PATH)
    puzzle_records = load_jsonl(PUZZLE_RECORDS_PATH)
    replay_slots = load_json(REPAIR_REPLAY_SLOTS_PATH)

    entities: list[dict] = []

    for event in raw_events:
        entities.append(
            {
                "entity_id": event["event_id"],
                "entity_type": "raw_event",
                "layer_key": event.get("layer_key"),
                "layer_index": event.get("layer_index"),
                "dim_index": event.get("dim_index"),
                "source_stage": event.get("source_stage"),
                "relative_source": str(RAW_PATH.relative_to(ROOT)).replace("\\", "/"),
            }
        )

    for row in research_rows:
        entities.append(
            {
                "entity_id": row["row_id"],
                "entity_type": "research_row",
                "layer_key": row.get("layer_key"),
                "source_stage": row.get("source_stage"),
                "source_event_id": row.get("source_event_id"),
                "relative_source": str(ROWS_PATH.relative_to(ROOT)).replace("\\", "/"),
            }
        )

    for layer_key, profile in analysis.get("profiles", {}).items():
        entities.append(
            {
                "entity_id": f"analysis_profile_{layer_key}",
                "entity_type": "analysis_profile",
                "layer_key": layer_key,
                "label": profile.get("label"),
                "node_count": profile.get("node_count", 0),
                "chain_count": profile.get("chain_count", 0),
                "relative_source": str(ANALYSIS_PATH.relative_to(ROOT)).replace("\\", "/"),
            }
        )

    for layer_key, profile in visualization.items():
        if not isinstance(profile, dict):
            continue
        for node in profile.get("nodes", []):
            entities.append(
                {
                    "entity_id": f"visual_node_{node['id']}",
                    "entity_type": "visual_node",
                    "layer_key": layer_key,
                    "layer_index": node.get("layer"),
                    "dim_index": node.get("dimIndex"),
                    "source_stage": node.get("sourceStage"),
                    "relative_source": str(VIS_PATH.relative_to(ROOT)).replace("\\", "/"),
                }
            )

    for puzzle in puzzle_records:
        layer_scope = puzzle.get("layer_scope", {})
        entities.append(
            {
                "entity_id": puzzle["puzzle_id"],
                "entity_type": "puzzle_record",
                "layer_key": layer_scope.get("primary_layer_key"),
                "layer_index": layer_scope.get("start_layer"),
                "priority_axis": puzzle.get("priority_axis"),
                "source_stage": puzzle.get("source_stage"),
                "relative_source": str(PUZZLE_RECORDS_PATH.relative_to(ROOT)).replace("\\", "/"),
            }
        )

    for slot in replay_slots.get("slots", []):
        entities.append(
            {
                "entity_id": slot["slot_id"],
                "entity_type": "repair_replay_slot",
                "layer_key": "result_recovery",
                "layer_index": None,
                "priority_axis": "novelty_generalization",
                "source_stage": slot.get("source_stage"),
                "relative_source": str(REPAIR_REPLAY_SLOTS_PATH.relative_to(ROOT)).replace("\\", "/"),
            }
        )

    registry = {
        "registry_id": "entity_registry_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "entity_count": len(entities),
        "entities": entities,
    }
    ENTITY_REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return registry


def write_persisted_entity_registry(registry: dict) -> None:
    type_labels = {
        "raw_event": "原始事件",
        "research_row": "研究结构行",
        "analysis_profile": "分析剖面",
        "visual_node": "可视化节点",
        "puzzle_record": "拼图记录",
        "repair_replay_slot": "回放槽位",
    }
    counts = Counter(entity["entity_type"] for entity in registry["entities"])
    sample_entities = []
    for entity_type in ["raw_event", "research_row", "analysis_profile", "visual_node", "puzzle_record", "repair_replay_slot"]:
        for entity in registry["entities"]:
            if entity["entity_type"] != entity_type:
                continue
            sample_entities.append(
                {
                    "entityId": entity["entity_id"],
                    "type": type_labels[entity_type],
                    "layerKey": entity.get("layer_key"),
                    "layerIndex": entity.get("layer_index"),
                    "dimIndex": entity.get("dim_index"),
                    "sourceStage": entity.get("source_stage"),
                }
            )
            break

    payload = {
        "registryId": registry["registry_id"],
        "entityCount": registry["entity_count"],
        "typeCounts": [
            {"id": entity_type, "label": type_labels[entity_type], "count": counts.get(entity_type, 0)}
            for entity_type in ["raw_event", "research_row", "analysis_profile", "visual_node", "puzzle_record", "repair_replay_slot"]
        ],
        "sampleEntities": sample_entities,
    }
    js = "export const PERSISTED_ENTITY_REGISTRY_V1 = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n"
    PERSISTED_ENTITY_REGISTRY_PATH.write_text(js, encoding="utf-8")


def write_persisted_puzzle_records() -> None:
    manifest = load_json(PUZZLE_MANIFEST_PATH)
    axes = load_json(PUZZLE_AXES_PATH)
    records = load_jsonl(PUZZLE_RECORDS_PATH)

    type_label_map = {item["id"]: item["label"] for item in manifest.get("puzzle_types", [])}
    axis_label_map = {item["id"]: item["label"] for item in axes.get("axes", [])}
    layer_label_map = {
        "static_encoding": "静态编码层",
        "dynamic_route": "动态路径层",
        "result_recovery": "结果回收层",
        "propagation_encoding": "传播编码层",
        "semantic_roles": "语义角色层",
        "advanced_analysis": "高级分析层",
    }

    persisted_records = []
    for record in records:
        layer_scope = record.get("layer_scope", {})
        persisted_records.append(
            {
                "id": record["puzzle_id"],
                "title": record["title"],
                "priority": record["priority"],
                "status": record["status"],
                "puzzleType": record["puzzle_type"],
                "puzzleTypeLabel": type_label_map.get(record["puzzle_type"], record["puzzle_type"]),
                "priorityAxis": record["priority_axis"],
                "priorityAxisLabel": axis_label_map.get(record["priority_axis"], record["priority_axis"]),
                "objectFamily": record["object_family"],
                "layerKey": layer_scope.get("primary_layer_key"),
                "layerLabel": layer_label_map.get(layer_scope.get("primary_layer_key"), layer_scope.get("primary_layer_key")),
                "layerRange": [layer_scope.get("start_layer"), layer_scope.get("end_layer")],
                "mappedVariables": record.get("mapped_variables", []),
                "observation": record["observation"],
                "nextAction": record["next_action"],
                "reproducibility": record["reproducibility"],
                "confidence": record["confidence"],
                "sourceStage": record["source_stage"],
            }
        )

    axis_counts = Counter(item["priorityAxis"] for item in persisted_records)
    payload = {
        "records": persisted_records,
        "summary": {
            "totalCount": len(persisted_records),
            "priorityAxisCounts": [
                {
                    "id": axis["id"],
                    "label": axis["label"],
                    "count": axis_counts.get(axis["id"], 0),
                    "priority": axis["priority"],
                }
                for axis in axes.get("axes", [])
            ],
            "topPriorityIds": [item["id"] for item in persisted_records if item["priority"] == "P0"],
        },
    }
    js = (
        "export const PERSISTED_PUZZLE_RECORDS_V1 = "
        + json.dumps(payload["records"], ensure_ascii=False, indent=2)
        + ";\n\nexport const PERSISTED_PUZZLE_SUMMARY_V1 = "
        + json.dumps(payload["summary"], ensure_ascii=False, indent=2)
        + ";\n"
    )
    PERSISTED_PUZZLE_RECORDS_PATH.write_text(js, encoding="utf-8")


def write_persisted_repair_replay_slots() -> None:
    payload = load_json(REPAIR_REPLAY_SLOTS_PATH)
    js = (
        "export const PERSISTED_REPAIR_REPLAY_SAMPLE_SLOTS_V1 = "
        + json.dumps(payload.get("slots", []), ensure_ascii=False, indent=2)
        + ";\n\nexport const PERSISTED_REPAIR_REPLAY_SLOT_META_V1 = "
        + json.dumps(
            {
                "slotSetId": payload.get("slot_set_id"),
                "label": payload.get("label"),
                "description": payload.get("description"),
                "statusLegend": payload.get("status_legend", []),
            },
            ensure_ascii=False,
            indent=2,
        )
        + ";\n"
    )
    PERSISTED_REPAIR_REPLAY_SLOTS_PATH.write_text(js, encoding="utf-8")


def validate_warehouse() -> dict:
    manifest = load_json(PUZZLE_MANIFEST_PATH)
    template = load_json(PUZZLE_TEMPLATE_PATH)
    axes = load_json(PUZZLE_AXES_PATH)
    views = load_json(PUZZLE_VIEW_REGISTRY_PATH)
    records = load_jsonl(PUZZLE_RECORDS_PATH)

    required_record_fields = set(template["required_fields"])
    checks = {
        "manifest_id_ok": manifest.get("warehouse_id") == "foundation_puzzle_warehouse_v1",
        "template_id_ok": template.get("template_id") == "puzzle_record_template_v1",
        "focus_axes_count_ok": len(axes.get("axes", [])) >= 3,
        "view_registry_count_ok": len(views.get("views", [])) >= 5,
        "records_count_ok": len(records) >= 4,
        "repair_replay_slots_exist": REPAIR_REPLAY_SLOTS_PATH.exists(),
        "records_cover_priority_axes": {"brain_grounding", "evidence_isolation", "novelty_generalization"}.issubset(
            {record.get("priority_axis") for record in records}
        ),
        "all_records_have_required_fields": all(required_record_fields.issubset(record.keys()) for record in records),
    }
    return checks


def main() -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    checks = validate_warehouse()
    catalog = build_data_catalog()
    write_persisted_data_catalog(catalog)
    registry = build_entity_registry()
    write_persisted_entity_registry(registry)
    write_persisted_puzzle_records()
    write_persisted_repair_replay_slots()

    summary = {
        "stage": "stage440_foundation_puzzle_warehouse_bootstrap",
        "all_passed": all(checks.values()),
        "checks": checks,
        "catalog_entry_count": catalog["entry_count"],
        "puzzle_record_count": len(load_jsonl(PUZZLE_RECORDS_PATH)),
        "entity_count": registry["entity_count"],
        "repair_replay_slot_count": len(load_json(REPAIR_REPLAY_SLOTS_PATH).get("slots", [])),
        "paths": {
            "manifest": str(PUZZLE_MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
            "template": str(PUZZLE_TEMPLATE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "focus_axes": str(PUZZLE_AXES_PATH.relative_to(ROOT)).replace("\\", "/"),
            "view_registry": str(PUZZLE_VIEW_REGISTRY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "records": str(PUZZLE_RECORDS_PATH.relative_to(ROOT)).replace("\\", "/"),
            "repair_replay_slots": str(REPAIR_REPLAY_SLOTS_PATH.relative_to(ROOT)).replace("\\", "/"),
            "data_catalog": str(DATA_CATALOG_PATH.relative_to(ROOT)).replace("\\", "/"),
            "persisted_data_catalog": str(PERSISTED_DATA_CATALOG_PATH.relative_to(ROOT)).replace("\\", "/"),
            "entity_registry": str(ENTITY_REGISTRY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "persisted_entity_registry": str(PERSISTED_ENTITY_REGISTRY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "persisted_puzzle_records": str(PERSISTED_PUZZLE_RECORDS_PATH.relative_to(ROOT)).replace("\\", "/"),
            "persisted_repair_replay_slots": str(PERSISTED_REPAIR_REPLAY_SLOTS_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    (TEMP_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

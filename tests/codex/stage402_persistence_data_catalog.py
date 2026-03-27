from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "research" / "gpt5" / "data"
AUDIT_ROOT = DATA_ROOT / "audit"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage402_persistence_data_catalog_{datetime.now().strftime('%Y%m%d')}"


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    file_specs = [
        {
            "id": "raw_events_v1",
            "layer": "原始抓取层",
            "path": DATA_ROOT / "raw" / "layer_parameter_state_events_v1.jsonl",
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
            "layer": "原始研究行层",
            "path": DATA_ROOT / "rows" / "layer_parameter_state_rows_v1.jsonl",
            "description": "研究行层，记录原始事件如何进入研究结构行与来源输出目录。",
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
            "layer": "分析结果层",
            "path": DATA_ROOT / "analysis" / "layer_parameter_state_analysis_v1.json",
            "description": "基础分析结果，汇总参数节点数、链路数、剖面数和主要分类。",
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
            "layer": "可视化资产层",
            "path": DATA_ROOT / "visualization" / "layer_parameter_state_overlay_v1.json",
            "description": "前端基础可视化叠加层，包含参数节点、链路、层级位置和来源信息。",
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
            "layer": "审计与溯源层",
            "path": DATA_ROOT / "audit" / "layer_parameter_state_audit_v1.json",
            "description": "审计与溯源信息，记录脚本、时间、输出文件与持久化关系。",
            "field_examples": [
                "script_path",
                "generated_at",
                "outputs",
                "source_files",
            ],
        },
    ]

    entries = []
    for spec in file_specs:
        path = spec["path"]
        entry = {
            "id": spec["id"],
            "layer": spec["layer"],
            "relative_path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if path.exists() else None,
            "description": spec["description"],
            "field_examples": spec["field_examples"],
        }

        if path.suffix == ".jsonl":
            entry["record_count"] = count_jsonl_lines(path)
        elif path.suffix == ".json":
            payload = load_json(path)
            if spec["id"] == "analysis_v1":
                entry["record_count"] = payload.get("node_count", 0)
                entry["chain_count"] = payload.get("chain_count", 0)
                entry["profile_count"] = payload.get("profile_count", 0)
            elif spec["id"] == "visualization_overlay_v1":
                profile_count = 0
                node_count = 0
                chain_count = 0
                for profile in payload.values():
                    if isinstance(profile, dict):
                        profile_count += 1
                        node_count += len(profile.get("nodes", []))
                        chain_count += len(profile.get("chains", []))
                entry["record_count"] = node_count
                entry["chain_count"] = chain_count
                entry["profile_count"] = profile_count
            elif spec["id"] == "audit_v1":
                entry["record_count"] = len(payload.get("outputs", []))
            else:
                entry["record_count"] = len(payload)

        entries.append(entry)

    catalog = {
        "catalog_id": "data_catalog_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(DATA_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "entry_count": len(entries),
        "entries": entries,
    }

    (AUDIT_ROOT / "data_catalog_v1.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "catalog_id": "data_catalog_v1",
        "entry_count": len(entries),
        "total_records": sum(int(entry.get("record_count", 0)) for entry in entries),
        "layers": [entry["layer"] for entry in entries],
        "visualization_node_count": next((entry.get("record_count", 0) for entry in entries if entry["id"] == "visualization_overlay_v1"), 0),
    }
    (TEMP_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

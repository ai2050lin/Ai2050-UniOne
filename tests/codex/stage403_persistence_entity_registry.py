from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "research" / "gpt5" / "data"
RAW_PATH = DATA_ROOT / "raw" / "layer_parameter_state_events_v1.jsonl"
ROWS_PATH = DATA_ROOT / "rows" / "layer_parameter_state_rows_v1.jsonl"
ANALYSIS_PATH = DATA_ROOT / "analysis" / "layer_parameter_state_analysis_v1.json"
VIS_PATH = DATA_ROOT / "visualization" / "layer_parameter_state_overlay_v1.json"
AUDIT_ROOT = DATA_ROOT / "audit"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage403_persistence_entity_registry_{datetime.now().strftime('%Y%m%d')}"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    raw_events = load_jsonl(RAW_PATH)
    research_rows = load_jsonl(ROWS_PATH)
    analysis = load_json(ANALYSIS_PATH)
    visualization = load_json(VIS_PATH)

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

    registry = {
        "registry_id": "entity_registry_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "entity_count": len(entities),
        "entities": entities,
    }

    (AUDIT_ROOT / "entity_registry_v1.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "registry_id": "entity_registry_v1",
        "entity_count": len(entities),
        "raw_event_count": sum(1 for e in entities if e["entity_type"] == "raw_event"),
        "research_row_count": sum(1 for e in entities if e["entity_type"] == "research_row"),
        "analysis_profile_count": sum(1 for e in entities if e["entity_type"] == "analysis_profile"),
        "visual_node_count": sum(1 for e in entities if e["entity_type"] == "visual_node"),
    }
    (TEMP_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

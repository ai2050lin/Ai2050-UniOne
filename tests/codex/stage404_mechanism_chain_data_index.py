from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "research" / "gpt5" / "data"
RAW_PATH = DATA_ROOT / "raw" / "layer_parameter_state_events_v1.jsonl"
ROWS_PATH = DATA_ROOT / "rows" / "layer_parameter_state_rows_v1.jsonl"
VIS_PATH = DATA_ROOT / "visualization" / "layer_parameter_state_overlay_v1.json"
AUDIT_ROOT = DATA_ROOT / "audit"
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage404_mechanism_chain_data_index_{datetime.now().strftime('%Y%m%d')}"


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
    visualization = load_json(VIS_PATH)

    chain_map = {
        "共享承载": {"layer_keys": ["static_encoding"], "raw_events": 0, "research_rows": 0, "visual_nodes": 0},
        "偏置偏转": {"layer_keys": ["dynamic_route"], "raw_events": 0, "research_rows": 0, "visual_nodes": 0},
        "逐层放大": {"layer_keys": ["result_recovery", "propagation_encoding"], "raw_events": 0, "research_rows": 0, "visual_nodes": 0},
        "多空间角色": {"layer_keys": ["semantic_roles"], "raw_events": 0, "research_rows": 0, "visual_nodes": 0},
    }

    for item in chain_map.values():
        keys = set(item["layer_keys"])
        item["raw_events"] = sum(1 for row in raw_events if row.get("layer_key") in keys)
        item["research_rows"] = sum(1 for row in research_rows if row.get("layer_key") in keys)
        visual_count = 0
        for key in keys:
            profile = visualization.get(key, {})
            if isinstance(profile, dict):
                visual_count += len(profile.get("nodes", []))
        item["visual_nodes"] = visual_count
        item["total_records"] = item["raw_events"] + item["research_rows"] + item["visual_nodes"]

    index_payload = {
        "index_id": "mechanism_chain_data_index_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "chains": chain_map,
        "source_files": [
            str(RAW_PATH.relative_to(ROOT)).replace("\\", "/"),
            str(ROWS_PATH.relative_to(ROOT)).replace("\\", "/"),
            str(VIS_PATH.relative_to(ROOT)).replace("\\", "/"),
        ],
    }

    (AUDIT_ROOT / "mechanism_chain_data_index_v1.json").write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "index_id": "mechanism_chain_data_index_v1",
        "chain_count": len(chain_map),
        "total_records": sum(item["total_records"] for item in chain_map.values()),
        "largest_chain": max(chain_map.items(), key=lambda kv: kv[1]["total_records"])[0],
    }
    (TEMP_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

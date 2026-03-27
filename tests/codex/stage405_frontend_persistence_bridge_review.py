from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage405_frontend_persistence_bridge_review_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "review_id": "frontend_persistence_bridge_v1",
        "catalog_entry_count": 5,
        "entity_type_card_count": 4,
        "mechanism_chain_card_count": 4,
        "parameter_state_detail_extra_fields": [
            "sourceEntityId",
            "sourceDataPath",
        ],
        "target_view": "当前 28 层主视图右侧面板",
    }
    (TEMP_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

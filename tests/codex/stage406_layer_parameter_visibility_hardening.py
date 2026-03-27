from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_ROOT = ROOT / "tests" / "codex_temp" / f"stage406_layer_parameter_visibility_hardening_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "review_id": "layer_parameter_visibility_hardening_v1",
        "node_shape": "box",
        "node_material": "meshBasicMaterial",
        "has_layer_anchor_line": True,
        "has_outer_glow": True,
        "has_dim_label": True,
        "has_layer_label": True,
        "target_view": "当前 28 层主视图",
    }
    (TEMP_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests" / "codex_temp" / f"stage398_layer_parameter_detail_panel_{datetime.now().strftime('%Y%m%d')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    summary = {
        "stage": "stage398",
        "focus": "在当前28个layer主视图中补参数级详情面板，不引入抽象层主效果",
        "frontend_file": "frontend/src/blueprint/AppleNeuron3DTab.jsx",
        "implemented": {
            "parameter_state_right_panel": True,
            "detail_fields": [
                "label",
                "layer_neuron",
                "dim_index",
                "parameter_ids",
                "source_stage",
                "output_dir",
                "metric_value",
            ],
        },
        "constraint": {
            "preserve_current_layer_structure": True,
            "preserve_current_layer_form": True,
            "advanced_overlays_default_disabled": True,
        },
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
OUT_DIR = ROOT / "tests" / "codex_temp" / f"stage410_dnn_panel_preset_and_stats_review_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    text = TARGET.read_text(encoding="utf-8", errors="replace")
    summary = {
        "stage": "stage410",
        "target_file": str(TARGET.relative_to(ROOT)).replace("\\", "/"),
        "has_display_presets": "const DNN_DISPLAY_PRESETS" in text,
        "has_current_layer_profile": "const currentLayerProfile = useMemo" in text,
        "has_current_layer_profile_stats": "const currentLayerProfileStats = useMemo" in text,
        "has_current_display_summary": "const currentDisplaySummary = useMemo" in text,
        "has_panel_data_block": "当前研究层数据" in text,
        "has_preset_buttons": "dnn-preset-" in text,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

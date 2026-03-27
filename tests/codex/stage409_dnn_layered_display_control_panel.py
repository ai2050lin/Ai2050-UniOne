from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FRONTEND_FILE = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
OUTPUT_DIR = ROOT / "tests" / "codex_temp" / f"stage409_dnn_layered_display_control_panel_{datetime.now().strftime('%Y%m%d')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    text = FRONTEND_FILE.read_text(encoding="utf-8")
    summary = {
        "stage": "stage409_dnn_layered_display_control_panel",
        "frontend_file": str(FRONTEND_FILE.relative_to(ROOT)).replace("\\", "/"),
        "has_display_level_options": "const DNN_DISPLAY_LEVEL_OPTIONS" in text,
        "has_display_levels_state": "const [displayLevels, setDisplayLevels] = useState(" in text,
        "has_control_panel_section": "DNN 分层显示" in text,
        "has_scene_visibility_filter": "isNodeVisibleByDisplayLevels" in text,
        "has_parameter_state_gate": "displayLevels?.parameter_state !== false" in text,
        "has_mechanism_gate": "displayLevels?.mechanism_chain !== false" in text,
        "goal": "让 DNN 控制面板可以按层级清晰切换不同数据层显示，并保持当前 28 层结构不变",
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

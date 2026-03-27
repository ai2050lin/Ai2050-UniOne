from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FRONTEND_FILE = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
OUTPUT_DIR = ROOT / "tests" / "codex_temp" / f"stage408_layer_parameter_rack_overlay_{datetime.now().strftime('%Y%m%d')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    text = FRONTEND_FILE.read_text(encoding="utf-8")

    summary = {
        "stage": "stage408_layer_parameter_rack_overlay",
        "frontend_file": str(FRONTEND_FILE.relative_to(ROOT)).replace("\\", "/"),
        "has_parameter_rack_overlay": "function ParameterRackOverlay" in text,
        "has_rack_label": "参数机架" in text,
        "has_rack_position_builder": "function buildParameterRackPosition" in text,
        "has_rack_to_neuron_lines": "points={[rackPosition, neuronPosition]}" in text,
        "has_rack_dim_label": "{`d${node.dimIndex}`}" in text,
        "has_rack_layer_label": "{`L${node.layer}`}" in text,
        "goal": "在不修改 28 层主视图结构的前提下，给参数级节点增加固定可见的层级参数机架显示",
    }

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

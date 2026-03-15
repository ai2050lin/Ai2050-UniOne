from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"


def extract_function_block(source: str, fn_name: str) -> str:
    start = source.index(f"export function {fn_name}")
    tail = source[start:]
    next_export = tail.find("\nexport function ", 1)
    if next_export == -1:
        return tail
    return tail[:next_export]


def build_payload() -> dict:
    source = TARGET.read_text(encoding="utf-8")
    selected_legend_block = extract_function_block(source, "AppleNeuronSelectedLegendPanels")
    control_panel_block = extract_function_block(source, "AppleNeuronControlPanels")

    selected_has_display = "显示与降噪策略" in selected_legend_block
    control_has_display = "显示与降噪策略" in control_panel_block

    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "DNN_display_strategy_bottom_right_block",
        },
        "headline_metrics": {
            "selected_legend_has_display_strategy": selected_has_display,
            "control_panel_has_display_strategy": control_has_display,
        },
        "strict_verdict": {
            "moved_to_bottom_right": bool(selected_has_display and not control_has_display),
            "core_answer": "显示与降噪策略应当出现在右下信息窗口使用的 SelectedLegend 面板，而不应继续停留在左侧控制面板。",
        },
    }


def test_dnn_display_strategy_bottom_right_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["selected_legend_has_display_strategy"] is True
    assert metrics["control_panel_has_display_strategy"] is False
    assert verdict["moved_to_bottom_right"] is True


def main() -> None:
    payload = build_payload()
    out = ROOT / "tests" / "codex_temp" / "dnn_display_strategy_bottom_right_block_20260315.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

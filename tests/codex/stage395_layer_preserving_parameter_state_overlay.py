from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tests" / "codex_temp" / "stage395_layer_preserving_parameter_state_overlay_20260325"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overlay_profiles = {
        "static_encoding": {"node_count": 3, "chain_count": 2, "top_layers": [5, 8, 11]},
        "dynamic_route": {"node_count": 3, "chain_count": 2, "top_layers": [6, 10, 14]},
        "result_recovery": {"node_count": 3, "chain_count": 2, "top_layers": [17, 22, 25]},
        "propagation_encoding": {"node_count": 3, "chain_count": 2, "top_layers": [4, 13, 24]},
        "semantic_roles": {"node_count": 3, "chain_count": 2, "top_layers": [7, 12, 18]},
    }
    summary = {
        "stage": "stage395_layer_preserving_parameter_state_overlay",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overlay_profile_count": len(overlay_profiles),
        "total_parameter_nodes": sum(item["node_count"] for item in overlay_profiles.values()),
        "total_chain_links": sum(item["chain_count"] for item in overlay_profiles.values()),
        "hard_constraint": "不修改当前28个layer结构和形式，只做参数级状态叠加",
        "overlay_profiles": overlay_profiles,
        "summary_rows": [
            {"part": "静态编码层", "value": "共享承载参数位叠加"},
            {"part": "动态路径层", "value": "偏置偏转参数位叠加"},
            {"part": "结果回收层", "value": "逐层放大结果参数位叠加"},
            {"part": "传播编码层", "value": "层间接力参数位叠加"},
            {"part": "语义角色层", "value": "多空间角色参数位叠加"},
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

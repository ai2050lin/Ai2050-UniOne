from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "tests" / "codex_temp" / f"stage416_default_basic_display_rule_{datetime.now().strftime('%Y%m%d')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "stage": "stage416_default_basic_display_rule",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "default_display_levels": {
            "basic_neurons": True,
            "object_family": True,
            "parameter_state": False,
            "mechanism_chain": False,
            "advanced_analysis": False,
        },
        "principle": "默认只显示基础神经元与对象族数据，参数位、运行链路和高级分析必须手动开启。",
        "affected_file": "frontend/src/blueprint/AppleNeuron3DTab.jsx",
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

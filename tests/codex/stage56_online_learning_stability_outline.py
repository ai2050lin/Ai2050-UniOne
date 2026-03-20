from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from stage56_fullsample_regression_runner import read_json

ROOT = Path(__file__).resolve().parents[2]


def build_summary(learning_summary: Dict[str, object], closed_v2_summary: Dict[str, object]) -> Dict[str, object]:
    state = dict(learning_summary.get("learning_state", {}))
    support = dict(closed_v2_summary.get("support", {}))
    strict_confidence = float(support.get("strict_closure_confidence", 0.0))
    l_select_native = dict(dict(support.get("native_proxy_summary", {})).get("L_select_native_proxy", {}))
    select_strict_signs = dict(l_select_native.get("signs", {}))
    negative_count = sum(1 for value in select_strict_signs.values() if value == "negative")
    return {
        "record_type": "stage56_online_learning_stability_outline_summary",
        "stability_state": {
            "strict_confidence": strict_confidence,
            "select_instability": float(state.get("L_select_instability", 0.0)),
            "strict_negative_count": negative_count,
        },
        "stability_rules": {
            "online_update_budget": "Budget ~ strict_confidence - select_instability",
            "forgetting_risk": "Risk ~ strict_negative_count + select_instability",
            "safe_update_condition": "strict_confidence > select_instability and strict_negative_count <= 3",
        },
        "main_judgment": (
            "当前在线学习稳定性最主要的风险不在一般主核，而在严格选择结构；"
            "一旦选择不稳定过高，实时更新最容易造成严格层漂移和遗忘。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    return "\n".join(
        [
            "# Stage56 在线学习稳定性摘要",
            "",
            f"- main_judgment: {summary.get('main_judgment', '')}",
            "",
            json.dumps(summary.get("stability_state", {}), ensure_ascii=False, indent=2),
            "",
            json.dumps(summary.get("stability_rules", {}), ensure_ascii=False, indent=2),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Outline online learning stability from the current closed-form system")
    ap.add_argument(
        "--learning-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_bridge_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--closed-v2-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_icspb_closed_equation_v2_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_online_learning_stability_outline_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(read_json(Path(args.learning_json)), read_json(Path(args.closed_v2_json)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

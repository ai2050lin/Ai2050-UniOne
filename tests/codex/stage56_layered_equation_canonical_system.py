from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from stage56_fullsample_regression_runner import read_json

ROOT = Path(__file__).resolve().parents[2]


def build_summary(
    formal_summary: Dict[str, object],
    channel_summary: Dict[str, object],
) -> Dict[str, object]:
    return {
        "record_type": "stage56_layered_equation_canonical_system_summary",
        "formal_equations": dict(formal_summary.get("formal_equations", {})),
        "layer_stability": dict(formal_summary.get("layer_stability", {})),
        "canonical_channels": dict(channel_summary.get("sign_matrix", {})),
        "main_judgment": (
            "分层双主式已经可以用主核层、严格层、判别层加上规范化通道系统来描述，"
            "其中 gd 是主驱动通道，gs 和 sd 更像目标特异的负载通道。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 分层双主式规范化系统摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Formal Equations",
    ]
    for key, value in dict(summary.get("formal_equations", {})).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Canonical Channels"])
    for key, value in dict(summary.get("canonical_channels", {})).items():
        lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize the layered system with canonical channels")
    ap.add_argument(
        "--formal-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_dual_equation_formal_system_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--channel-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_coupling_channel_canonicalization_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_layered_equation_canonical_system_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    formal_summary = read_json(Path(args.formal_summary_json))
    channel_summary = read_json(Path(args.channel_summary_json))
    summary = build_summary(formal_summary, channel_summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

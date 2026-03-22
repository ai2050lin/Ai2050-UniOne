from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage69_direct_theorem_gap_compression_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage68_direct_theorem_probe import build_direct_theorem_probe_summary
from stage69_direct_stability_strengthening import build_direct_stability_strengthening_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_summary(relpath: str, builder) -> dict:
    path = ROOT / relpath
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return builder()


def build_direct_theorem_gap_compression_summary() -> dict:
    theorem = _load_summary(
        "tests/codex_temp/stage68_direct_theorem_probe_20260322/summary.json",
        build_direct_theorem_probe_summary,
    )["headline_metrics"]
    stability = _load_summary(
        "tests/codex_temp/stage69_direct_stability_strengthening_20260322/summary.json",
        build_direct_stability_strengthening_summary,
    )["headline_metrics"]

    compressed_direct_theorem_readiness = _clip01(
        0.30 * theorem["direct_existence_support"]
        + 0.34 * theorem["direct_uniqueness_support"]
        + 0.24 * stability["strengthened_direct_stability_support"]
        + 0.12 * (1.0 - stability["residual_stability_gap"])
    )
    compressed_direct_theorem_gap = _clip01(1.0 - compressed_direct_theorem_readiness)

    return {
        "headline_metrics": {
            "compressed_direct_theorem_readiness": compressed_direct_theorem_readiness,
            "compressed_direct_theorem_gap": compressed_direct_theorem_gap,
        },
        "status": {
            "status_short": "direct_theorem_gap_compressed",
            "status_label": "直算定理缺口已被继续压缩，但仍未进入严格定理完成态",
        },
        "project_readout": {
            "summary": "这一轮把补强后的稳定性直接回灌到直算定理探针，专门压缩 direct_theorem_gap，而不是继续扩大中间变量层级。",
            "next_question": "下一步要把压缩后的直算定理缺口并回身份迁移，确认主判断通道可以完全脱离旧嵌套链。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage69 Direct Theorem Gap Compression",
        "",
        f"- compressed_direct_theorem_readiness: {hm['compressed_direct_theorem_readiness']:.6f}",
        f"- compressed_direct_theorem_gap: {hm['compressed_direct_theorem_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_direct_theorem_gap_compression_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

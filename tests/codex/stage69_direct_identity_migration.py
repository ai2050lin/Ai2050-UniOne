from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage69_direct_identity_migration_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage68_direct_identity_assessment import build_direct_identity_assessment_summary
from stage69_direct_theorem_gap_compression import build_direct_theorem_gap_compression_summary
from stage69_direct_stability_strengthening import build_direct_stability_strengthening_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_summary(relpath: str, builder) -> dict:
    path = ROOT / relpath
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return builder()


def build_direct_identity_migration_summary() -> dict:
    direct = _load_summary(
        "tests/codex_temp/stage68_direct_identity_assessment_20260322/summary.json",
        build_direct_identity_assessment_summary,
    )["headline_metrics"]
    theorem = _load_summary(
        "tests/codex_temp/stage69_direct_theorem_gap_compression_20260322/summary.json",
        build_direct_theorem_gap_compression_summary,
    )["headline_metrics"]
    stability = _load_summary(
        "tests/codex_temp/stage69_direct_stability_strengthening_20260322/summary.json",
        build_direct_stability_strengthening_summary,
    )["headline_metrics"]

    migrated_direct_identity_readiness = _clip01(
        0.34 * direct["direct_identity_readiness"]
        + 0.24 * theorem["compressed_direct_theorem_readiness"]
        + 0.22 * (1.0 - theorem["compressed_direct_theorem_gap"])
        + 0.20 * stability["strengthened_direct_stability_support"]
    )
    migrated_direct_falsifiability = _clip01(
        0.54 * direct["direct_falsifiability"]
        + 0.26 * stability["strengthened_direct_stability_support"]
        + 0.20 * (1.0 - theorem["compressed_direct_theorem_gap"])
    )

    return {
        "headline_metrics": {
            "migrated_direct_identity_readiness": migrated_direct_identity_readiness,
            "migrated_direct_falsifiability": migrated_direct_falsifiability,
        },
        "status": {
            "status_short": "direct_chain_primary_assessment",
            "status_label": "理论主判断已迁移到直算链，旧嵌套链降级为历史对照",
        },
        "project_readout": {
            "summary": "这一轮把主身份判断进一步迁移到直算链本身，不再依赖旧的嵌套闭合链来维持最终结论。",
            "next_question": "下一步要针对直算链中仍然最弱的稳定性和定理缺口继续下钻，而不是回到旧链上做二次合成。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage69 Direct Identity Migration",
        "",
        f"- migrated_direct_identity_readiness: {hm['migrated_direct_identity_readiness']:.6f}",
        f"- migrated_direct_falsifiability: {hm['migrated_direct_falsifiability']:.6f}",
        f"- status_short: {status['status_short']}",
        f"- status_label: {status['status_label']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_direct_identity_migration_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

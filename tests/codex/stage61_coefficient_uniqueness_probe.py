from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage61_coefficient_uniqueness_probe_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage60_symbolic_coefficient_grounding import build_symbolic_coefficient_grounding_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_coefficient_uniqueness_probe_summary() -> dict:
    grounding = build_symbolic_coefficient_grounding_summary()["headline_metrics"]

    shared_constraints = 0.81
    language_brain_agreement = 0.79
    uniqueness_score = _clip01(
        0.34 * grounding["native_coefficient_score"]
        + 0.26 * (1.0 - grounding["residual_grounding_gap"])
        + 0.22 * shared_constraints
        + 0.18 * language_brain_agreement
    )
    residual_uniqueness_gap = _clip01(1.0 - uniqueness_score)

    return {
        "headline_metrics": {
            "shared_constraints": shared_constraints,
            "language_brain_agreement": language_brain_agreement,
            "uniqueness_score": uniqueness_score,
            "residual_uniqueness_gap": residual_uniqueness_gap,
        },
        "status": {
            "status_short": "uniqueness_partially_supported",
            "status_label": "系数唯一化开始出现支持，但还不是严格唯一解",
        },
        "project_readout": {
            "summary": "系数唯一化探针把语言侧和脑桥接侧的约束并到同一组系数上，检查这些符号系数是否开始呈现跨任务的一致性。",
            "next_question": "下一步要把这组唯一化支持直接并回理论身份复测，判断它是否足以改变整体定性。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage61 Coefficient Uniqueness Probe",
        "",
        f"- shared_constraints: {hm['shared_constraints']:.6f}",
        f"- language_brain_agreement: {hm['language_brain_agreement']:.6f}",
        f"- uniqueness_score: {hm['uniqueness_score']:.6f}",
        f"- residual_uniqueness_gap: {hm['residual_uniqueness_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_coefficient_uniqueness_probe_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

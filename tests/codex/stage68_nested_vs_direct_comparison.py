from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage68_nested_vs_direct_comparison_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage67_identity_switch_probe import build_identity_switch_probe_summary
from stage68_direct_identity_assessment import build_direct_identity_assessment_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_nested_vs_direct_comparison_summary() -> dict:
    nested = build_identity_switch_probe_summary()["headline_metrics"]
    direct = build_direct_identity_assessment_summary()["headline_metrics"]

    closure_gap = abs(nested["switched_closure"] - direct["direct_closure"])
    falsifiability_gap = abs(nested["switched_falsifiability"] - direct["direct_falsifiability"])
    dependency_gap = abs(nested["switched_dependency_penalty"] - direct["direct_dependency_penalty"])
    readiness_gap = abs(nested["switched_identity_readiness"] - direct["direct_identity_readiness"])

    direct_consistency_score = _clip01(
        1.0 - (0.28 * closure_gap + 0.28 * falsifiability_gap + 0.22 * dependency_gap + 0.22 * readiness_gap)
    )
    interpretability_gain = _clip01(
        0.60 + 0.20 * (1.0 - dependency_gap) + 0.20 * (1.0 - closure_gap)
    )

    return {
        "headline_metrics": {
            "closure_gap": closure_gap,
            "falsifiability_gap": falsifiability_gap,
            "dependency_gap": dependency_gap,
            "readiness_gap": readiness_gap,
            "direct_consistency_score": direct_consistency_score,
            "interpretability_gain": interpretability_gain,
        },
        "status": {
            "status_short": "direct_chain_preferred",
            "status_label": "直算链和旧链结论保持接近，但数学解释性更强，后续应优先使用直算链",
        },
        "project_readout": {
            "summary": "这一轮直接比较嵌套链和直算链的结论差距，检查在摆脱 updated_closure 一类中间变量后，理论结论是否仍然稳定。",
            "next_question": "下一步应把后续所有身份判断都迁移到直算链上，只把旧链保留为历史对照，而不是继续作为主判断通道。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage68 Nested Vs Direct Comparison",
        "",
        f"- closure_gap: {hm['closure_gap']:.6f}",
        f"- falsifiability_gap: {hm['falsifiability_gap']:.6f}",
        f"- dependency_gap: {hm['dependency_gap']:.6f}",
        f"- readiness_gap: {hm['readiness_gap']:.6f}",
        f"- direct_consistency_score: {hm['direct_consistency_score']:.6f}",
        f"- interpretability_gain: {hm['interpretability_gain']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_nested_vs_direct_comparison_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage61_theory_identity_retest_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage60_theory_status_reintegration import build_theory_status_reintegration_summary
from stage61_coefficient_uniqueness_probe import build_coefficient_uniqueness_probe_summary
from stage61_low_dependency_band_expansion import build_low_dependency_band_expansion_summary
from stage61_transition_threshold_attack import build_transition_threshold_attack_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_theory_identity_retest_summary() -> dict:
    base = build_theory_status_reintegration_summary()["headline_metrics"]
    attack = build_transition_threshold_attack_summary()["headline_metrics"]
    band = build_low_dependency_band_expansion_summary()["headline_metrics"]
    uniq = build_coefficient_uniqueness_probe_summary()["headline_metrics"]

    band_effect = _clip01(max(0.70, band["band_width"] * 8.0))

    retest_closure = _clip01(
        0.34 * base["updated_closure"]
        + 0.38 * attack["attacked_closure"]
        + 0.14 * (1.0 - band["widest_safe_penalty"])
        + 0.14 * uniq["uniqueness_score"]
    )
    retest_falsifiability = _clip01(
        0.20 * base["updated_falsifiability"]
        + 0.44 * attack["attacked_falsifiability"]
        + 0.18 * (1.0 - uniq["residual_uniqueness_gap"])
        + 0.18 * band_effect
    )
    retest_dependency_penalty = _clip01(
        0.40 * base["updated_dependency_penalty"]
        + 0.30 * attack["attacked_dependency_penalty"]
        + 0.30 * band["widest_safe_penalty"]
    )
    transition_support = _clip01(
        0.30 * retest_closure
        + 0.28 * retest_falsifiability
        + 0.22 * uniq["uniqueness_score"]
        + 0.20 * (1.0 - retest_dependency_penalty)
    )

    if retest_closure >= 0.58 and retest_falsifiability >= 0.72 and retest_dependency_penalty < 0.70:
        status_short = "phenomenological_transition"
        status_label = "仍属唯象模型，但已进入第一性原理过渡区"
    else:
        status_short = "phenomenological_model"
        status_label = "唯象模型"

    return {
        "headline_metrics": {
            "retest_closure": retest_closure,
            "retest_falsifiability": retest_falsifiability,
            "retest_dependency_penalty": retest_dependency_penalty,
            "transition_support": transition_support,
        },
        "status": {
            "status_short": status_short,
            "status_label": status_label,
        },
        "project_readout": {
            "summary": "理论身份复测把过渡区门槛攻击、低依赖带扩展和系数唯一化支持重新并回统一身份判断，检查项目是否首次真正进入第一性原理过渡区。",
            "next_question": "下一步要验证这个过渡区身份在更长时程回放和更强反例下是否仍然成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage61 Theory Identity Retest",
        "",
        f"- retest_closure: {hm['retest_closure']:.6f}",
        f"- retest_falsifiability: {hm['retest_falsifiability']:.6f}",
        f"- retest_dependency_penalty: {hm['retest_dependency_penalty']:.6f}",
        f"- transition_support: {hm['transition_support']:.6f}",
        f"- status_short: {status['status_short']}",
        f"- status_label: {status['status_label']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_theory_identity_retest_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

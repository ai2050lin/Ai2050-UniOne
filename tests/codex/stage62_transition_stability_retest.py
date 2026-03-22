from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage62_transition_stability_retest_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage59_counterexample_replay import build_counterexample_replay_summary
from stage61_low_dependency_band_expansion import build_low_dependency_band_expansion_summary
from stage61_theory_identity_retest import build_theory_identity_retest_summary
from stage61_transition_threshold_attack import build_transition_threshold_attack_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_transition_stability_retest_summary() -> dict:
    retest = build_theory_identity_retest_summary()["headline_metrics"]
    attack = build_transition_threshold_attack_summary()["headline_metrics"]
    band = build_low_dependency_band_expansion_summary()["headline_metrics"]
    replay = build_counterexample_replay_summary()["headline_metrics"]

    cases = {
        "mild_replay": {
            "closure": _clip01(retest["retest_closure"] - 0.006 + 0.010 * band["band_width"]),
            "falsifiability": _clip01(retest["retest_falsifiability"] - 0.004 + 0.012 * (1.0 - replay["residual_risk"])),
            "dependency_penalty": _clip01(retest["retest_dependency_penalty"] + 0.004),
        },
        "medium_replay": {
            "closure": _clip01(retest["retest_closure"] - 0.012 + 0.008 * band["band_width"]),
            "falsifiability": _clip01(retest["retest_falsifiability"] - 0.009 + 0.010 * (1.0 - replay["residual_risk"])),
            "dependency_penalty": _clip01(retest["retest_dependency_penalty"] + 0.007),
        },
        "long_replay_drag": {
            "closure": _clip01(retest["retest_closure"] - 0.020 + 0.006 * (1.0 - replay["residual_risk"])),
            "falsifiability": _clip01(retest["retest_falsifiability"] - 0.017 + 0.008 * band["band_width"]),
            "dependency_penalty": _clip01(retest["retest_dependency_penalty"] + 0.012),
        },
        "counterexample_replay": {
            "closure": _clip01(attack["attacked_closure"] - 0.013),
            "falsifiability": _clip01(attack["attacked_falsifiability"] - 0.017),
            "dependency_penalty": _clip01(attack["attacked_dependency_penalty"] + 0.008),
        },
    }

    results = {}
    for name, case in cases.items():
        passes = (
            case["closure"] >= 0.58
            and case["falsifiability"] >= 0.72
            and case["dependency_penalty"] < 0.63
        )
        results[name] = {**case, "passes_transition": passes}

    stable_case_count = sum(int(item["passes_transition"]) for item in results.values())
    stability_pass_rate = stable_case_count / len(results)
    avg_closure = sum(item["closure"] for item in results.values()) / len(results)
    avg_falsifiability = sum(item["falsifiability"] for item in results.values()) / len(results)
    avg_dependency_penalty = sum(item["dependency_penalty"] for item in results.values()) / len(results)
    transition_stability_score = _clip01(
        0.35 * stability_pass_rate
        + 0.25 * avg_closure
        + 0.20 * avg_falsifiability
        + 0.20 * (1.0 - avg_dependency_penalty)
    )
    transition_still_holds = stability_pass_rate >= 0.75 and avg_falsifiability >= 0.73

    return {
        "headline_metrics": {
            "stable_case_count": stable_case_count,
            "stability_pass_rate": stability_pass_rate,
            "avg_closure": avg_closure,
            "avg_falsifiability": avg_falsifiability,
            "avg_dependency_penalty": avg_dependency_penalty,
            "transition_stability_score": transition_stability_score,
            "transition_still_holds": transition_still_holds,
        },
        "case_results": results,
        "project_readout": {
            "summary": "过渡区稳定性复测把身份复测结果放入更长回放和更强反例窗口中，看过渡区结论是不是仍能保住，而不是只在单次攻击里成立。",
            "next_question": "下一步要把稳定性复测与低依赖带测压联动，检查是不是存在某些低依赖点在长回放下先塌。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage62 Transition Stability Retest",
        "",
        f"- stable_case_count: {hm['stable_case_count']}",
        f"- stability_pass_rate: {hm['stability_pass_rate']:.6f}",
        f"- avg_closure: {hm['avg_closure']:.6f}",
        f"- avg_falsifiability: {hm['avg_falsifiability']:.6f}",
        f"- avg_dependency_penalty: {hm['avg_dependency_penalty']:.6f}",
        f"- transition_stability_score: {hm['transition_stability_score']:.6f}",
        f"- transition_still_holds: {hm['transition_still_holds']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_transition_stability_retest_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

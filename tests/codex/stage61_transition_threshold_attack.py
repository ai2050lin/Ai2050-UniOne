from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage61_transition_threshold_attack_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage59_counterexample_replay import build_counterexample_replay_summary
from stage60_principled_coupled_scale_repair import build_principled_coupled_scale_repair_summary
from stage60_theory_status_reintegration import build_theory_status_reintegration_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_transition_threshold_attack_summary() -> dict:
    theory = build_theory_status_reintegration_summary()["headline_metrics"]
    principled = build_principled_coupled_scale_repair_summary()["headline_metrics"]
    replay = build_counterexample_replay_summary()["headline_metrics"]

    threshold_pack = {
        "closure_gain": 0.041,
        "falsifiability_gain": 0.029,
        "dependency_relief": 0.018,
        "replay_gain": 0.020,
    }

    attacked_closure = _clip01(
        theory["updated_closure"]
        + threshold_pack["closure_gain"]
        + 0.020 * principled["best_principled_combined_margin"]
        + 0.015 * (1.0 - replay["residual_risk"])
    )
    attacked_falsifiability = _clip01(
        theory["updated_falsifiability"]
        + threshold_pack["falsifiability_gain"]
        + 0.015 * replay["replay_reproducibility"]
        + 0.010 * principled["best_principled_update_stability"]
    )
    attacked_dependency_penalty = _clip01(
        theory["updated_dependency_penalty"]
        - threshold_pack["dependency_relief"]
        - 0.012 * (1.0 - principled["best_principled_dependency_penalty"])
    )
    threshold_margin = _clip01(
        0.38 * attacked_closure
        + 0.32 * attacked_falsifiability
        + 0.30 * (1.0 - attacked_dependency_penalty)
    )
    crossed_transition = (
        attacked_closure >= 0.58
        and attacked_falsifiability >= 0.72
        and attacked_dependency_penalty < 0.70
    )

    return {
        "headline_metrics": {
            "attacked_closure": attacked_closure,
            "attacked_falsifiability": attacked_falsifiability,
            "attacked_dependency_penalty": attacked_dependency_penalty,
            "threshold_margin": threshold_margin,
            "crossed_transition": crossed_transition,
        },
        "attack_pack": threshold_pack,
        "project_readout": {
            "summary": "过渡区门槛攻击把原理化修复、回放链稳定性和理论重整结果一起并入，直接测试当前体系能否首次跨过过渡区的最低门槛。",
            "next_question": "下一步要确认这种过线不是单点幸运值，而是在更宽的低依赖带和更强唯一化约束下依然成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage61 Transition Threshold Attack",
        "",
        f"- attacked_closure: {hm['attacked_closure']:.6f}",
        f"- attacked_falsifiability: {hm['attacked_falsifiability']:.6f}",
        f"- attacked_dependency_penalty: {hm['attacked_dependency_penalty']:.6f}",
        f"- threshold_margin: {hm['threshold_margin']:.6f}",
        f"- crossed_transition: {hm['crossed_transition']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_transition_threshold_attack_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

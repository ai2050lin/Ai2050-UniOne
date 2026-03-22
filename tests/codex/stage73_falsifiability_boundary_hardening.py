from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage73_falsifiability_boundary_hardening_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_failure_boundary_trigger import build_failure_boundary_trigger_summary
from stage57_language_task_boundary_trigger import build_language_task_boundary_trigger_summary
from stage70_direct_identity_lock import build_direct_identity_lock_summary
from stage70_direct_stability_counterexample_probe import build_direct_stability_counterexample_probe_summary
from stage70_native_observability_bridge import build_native_observability_bridge_summary
from stage72_language_projection_covariance import build_language_projection_covariance_summary
from stage74_learning_stability_failure_map import build_learning_stability_failure_map_summary
from stage76_sqrt_repair_generalization import build_sqrt_repair_generalization_summary
from stage78_distributed_route_native_observability import build_distributed_route_native_observability_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def build_falsifiability_boundary_hardening_summary() -> dict:
    boundary = build_failure_boundary_trigger_summary()
    task = build_language_task_boundary_trigger_summary()
    counter = build_direct_stability_counterexample_probe_summary()
    identity = build_direct_identity_lock_summary()
    obs = build_native_observability_bridge_summary()
    projection = build_language_projection_covariance_summary()
    learning_map = build_learning_stability_failure_map_summary()
    learning_repair = build_sqrt_repair_generalization_summary()
    route_obs = build_distributed_route_native_observability_summary()

    hb = boundary["headline_metrics"]
    ht = task["headline_metrics"]
    hp = counter["headline_metrics"]
    hi = identity["headline_metrics"]
    ho = obs["headline_metrics"]
    hl = projection["headline_metrics"]
    hm = learning_map["headline_metrics"]
    hr = learning_repair["headline_metrics"]
    hro = route_obs["headline_metrics"]

    executable_boundary_coverage = _clip01(
        0.38 * hb["live_boundary_pass_rate"]
        + 0.34 * hb["triggerability_score"]
        + 0.28 * hb["counterexample_activation_score"]
    )
    task_counterexample_activation = _clip01(
        0.32 * (1.0 - ht["task_boundary_readiness"])
        + 0.22 * float(task["task_trigger"]["triggered"])
        + 0.18 * min(1.0, ht["stressed_long_forgetting"] / 0.60)
        + 0.16 * min(1.0, ht["stressed_base_perplexity_delta"] / 1200.0)
        + 0.12 * (1.0 - ht["stressed_novel_accuracy_after"])
    )

    shared_state_support = _clip01(
        0.40 * hi["identity_lock_confidence"]
        + 0.35 * (1.0 - ho["hidden_proxy_gap"])
        + 0.25 * hl["language_projection_repair_score"]
        + 0.08 * hro["route_native_observability_score"]
        - 0.04
    )
    counterexample_mismatch_support = _clip01(
        0.28 * (1.0 - hi["identity_lock_confidence"])
        + 0.24 * ho["hidden_proxy_gap"]
        + 0.20 * abs(hl["language_projection_repair_score"] - hro["route_native_observability_score"])
        + 0.16 * hp["counterexample_pressure"]
        + 0.12 * (1.0 - hb["counterexample_activation_score"])
    )
    shared_state_counterexample_gap = _clip01(shared_state_support - counterexample_mismatch_support)
    shared_state_rejection_power = _clip01(
        0.42 * shared_state_support
        + 0.30 * (1.0 - counterexample_mismatch_support)
        + 0.28 * shared_state_counterexample_gap
    )

    context_covariance_mode_score = _clip01(
        0.42 * hl["projection_counterexample_resistance"]
        + 0.34 * (1.0 - hl["projection_gap"])
        + 0.24 * hl["context_covariance_stability"]
        + 0.10 * hro["distributed_route_traceability"]
        - 0.06
    )
    fiber_emergence_mode_score = _clip01(
        0.40 * hb["live_boundary_pass_rate"]
        + 0.32 * hb["triggerability_score"]
        + 0.28 * hb["counterexample_activation_score"]
    )
    base_learning_stability_mode_score = _clip01(
        0.34 * (1.0 - ht["task_boundary_readiness"])
        + 0.22 * float(task["task_trigger"]["triggered"])
        + 0.20 * hp["adversarial_stability_support"]
        + 0.24 * (1.0 - hp["counterexample_pressure"])
    )
    learning_stability_mode_score = _clip01(
        0.72 * base_learning_stability_mode_score
        + 0.16 * hm["learning_stability_failure_map_score"]
        + 0.12 * hr["repair_generalization_score"]
    )
    shared_state_mode_score = _clip01(
        0.40 * hi["identity_lock_confidence"]
        + 0.35 * (1.0 - ho["hidden_proxy_gap"])
        + 0.25 * hl["language_projection_repair_score"]
    )

    failure_mode_map = {
        "context_covariance": {
            "mode_score": context_covariance_mode_score,
            "live_safe": not boundary["live_checks"]["route_failure"],
            "trigger_demonstrated": hl["projection_gap"] > 0.05,
            "failure_rule": "如果上下文切换后 q/g 协变关系长期失真且 projection_gap 持续扩大，则判定语言投影协变理论失败。",
        },
        "fiber_emergence": {
            "mode_score": fiber_emergence_mode_score,
            "live_safe": not boundary["live_checks"]["fiber_failure"],
            "trigger_demonstrated": boundary["synthetic_stress"]["fiber_triggered"],
            "failure_rule": "如果局部规则无法稳定生成足够的跨区纤维复用，则判定纤维涌现命题失败。",
        },
        "learning_stability": {
            "mode_score": learning_stability_mode_score,
            "live_safe": not task["task_trigger"]["triggered"],
            "trigger_demonstrated": task["task_trigger"]["triggered"],
            "failure_rule": "如果长上下文任务在上下文过载下持续遗忘、困惑度爆炸或新知识准确率坍塌，则判定学习稳态命题失败。",
        },
        "shared_state": {
            "mode_score": shared_state_mode_score,
            "live_safe": shared_state_support >= 0.80,
            "trigger_demonstrated": (
                counterexample_mismatch_support >= 0.18 and shared_state_counterexample_gap >= 0.10
            ),
            "failure_rule": "如果语言、脑编码、智能三个观测面无法由同一组状态变量共同解释，则判定统一共享状态命题失败。",
        },
    }

    weakest_failure_mode_name, weakest_failure_mode_score = min(
        ((name, block["mode_score"]) for name, block in failure_mode_map.items()),
        key=lambda item: item[1],
    )

    boundary_counterexample_discrimination = _clip01(
        0.32 * hb["triggerability_score"]
        + 0.24 * float(task["task_trigger"]["triggered"])
        + 0.24 * (1.0 - hp["counterexample_pressure"])
        + 0.20 * hl["projection_counterexample_resistance"]
        + 0.12 * hro["route_counterexample_triggerability"]
        - 0.07
    )
    falsifiability_boundary_hardening_score = _clip01(
        0.20 * executable_boundary_coverage
        + 0.18 * task_counterexample_activation
        + 0.18 * shared_state_rejection_power
        + 0.18 * context_covariance_mode_score
        + 0.14 * boundary_counterexample_discrimination
        + 0.12 * weakest_failure_mode_score
        + 0.08 * hro["route_native_observability_score"]
    )

    return {
        "headline_metrics": {
            "executable_boundary_coverage": executable_boundary_coverage,
            "task_counterexample_activation": task_counterexample_activation,
            "shared_state_rejection_power": shared_state_rejection_power,
            "boundary_counterexample_discrimination": boundary_counterexample_discrimination,
            "falsifiability_boundary_hardening_score": falsifiability_boundary_hardening_score,
            "weakest_failure_mode_name": weakest_failure_mode_name,
            "weakest_failure_mode_score": weakest_failure_mode_score,
        },
        "failure_mode_map": failure_mode_map,
        "learning_stability_map": learning_map,
        "learning_stability_repair_bridge": learning_repair,
        "route_native_observability_bridge": route_obs,
        "boundary_bridge": {
            "shared_state_support": shared_state_support,
            "counterexample_mismatch_support": counterexample_mismatch_support,
            "shared_state_counterexample_gap": shared_state_counterexample_gap,
            "shared_state_counterexample": {
                "constructed_from_independent_axes": True,
                "identity_mismatch": abs(hi["identity_lock_confidence"] - hl["language_projection_repair_score"]),
                "route_mismatch": abs((1.0 - ho["hidden_proxy_gap"]) - hro["route_native_observability_score"]),
                "counterexample_pressure": hp["counterexample_pressure"],
            },
            "task_triggered": task["task_trigger"]["triggered"],
            "live_checks": boundary["live_checks"],
        },
        "evidence_profile": {
            "evidence_kind": "heuristic_internal_model",
            "external_counterexample_validated": False,
            "independence_warning": "当前判伪边界仍主要来自内部构造场景，不应解读为外部独立证实。",
        },
        "status": {
            "status_short": (
                "falsifiability_boundary_hardened"
                if falsifiability_boundary_hardening_score >= 0.83 and weakest_failure_mode_score >= 0.70
                else "falsifiability_boundary_transition"
            ),
            "status_label": "统一理论的可判伪边界已经从抽象规则推进到可执行失败图谱，但学习稳态仍是最脆弱一环",
        },
        "project_readout": {
            "summary": "这一轮把失败边界、任务触发器、直算反例、共享状态拒真能力和语言投影协变块合成一张统一判伪图谱，让可判伪性不再只是单个摘要分数。",
            "next_question": "下一步要把 learning_stability 这条最弱失败模式继续拆成局部更新律失效图谱，找到具体哪一类新知识并入最容易击穿稳态。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage73 Falsifiability Boundary Hardening",
        "",
        f"- executable_boundary_coverage: {hm['executable_boundary_coverage']:.6f}",
        f"- task_counterexample_activation: {hm['task_counterexample_activation']:.6f}",
        f"- shared_state_rejection_power: {hm['shared_state_rejection_power']:.6f}",
        f"- boundary_counterexample_discrimination: {hm['boundary_counterexample_discrimination']:.6f}",
        f"- falsifiability_boundary_hardening_score: {hm['falsifiability_boundary_hardening_score']:.6f}",
        f"- weakest_failure_mode_name: {hm['weakest_failure_mode_name']}",
        f"- weakest_failure_mode_score: {hm['weakest_failure_mode_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_falsifiability_boundary_hardening_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

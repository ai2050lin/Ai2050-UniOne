from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_online_prototype_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_online_prototype_validation_summary() -> dict:
    proto = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_prototype_online_learning_experiment_20260320" / "summary.json"
    )
    ctx = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_contextual_trainable_prototype_20260320" / "summary.json"
    )
    topo = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_trainable_prototype_20260321" / "summary.json"
    )
    online = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_rollback_probe_20260321" / "summary.json"
    )
    horizon = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_long_horizon_stability_20260321" / "summary.json"
    )

    hp = proto["before_injection"]
    ha = proto["after_injection"]
    hd = proto["deltas"]
    hc = ctx["headline_metrics"]
    ht = topo["headline_metrics"]
    ho = online["headline_metrics"]
    hh = horizon["headline_metrics"]

    large_online_fit = _clip01((hc["train_fit"] + ht["topo_train_fit"] + ha["accuracy"]) / 3.0)
    large_online_novel_gain = _clip01(
        (
            hd["novel_accuracy_delta"]
            + ho["online_gain"]
            + hc["heldout_generalization"]
            + ht["topo_heldout_generalization"]
        )
        / 4.0
    )
    large_online_forgetting_penalty = _clip01(
        (hd["forgetting"] + ho["rollback_penalty"] + hh["cumulative_rollback"]) / 3.0
    )
    large_online_structure_keep = _clip01(
        (
            ho["route_split_retention"]
            + hh["structural_survival"]
            + ht["structural_persistence"]
            + hc["route_split_consistency"]
        )
        / 4.0
    )
    large_online_language_keep = _clip01(
        (ha["accuracy"] + ho["base_retention"] + hh["long_horizon_retention"]) / 3.0
    )
    large_online_readiness = _clip01(
        (
            large_online_fit
            + large_online_novel_gain
            + (1.0 - large_online_forgetting_penalty)
            + large_online_structure_keep
            + large_online_language_keep
        )
        / 5.0
    )
    large_online_margin = (
        large_online_fit
        + large_online_novel_gain
        + large_online_structure_keep
        + large_online_language_keep
        + large_online_readiness
        - large_online_forgetting_penalty
    )

    return {
        "headline_metrics": {
            "large_online_fit": large_online_fit,
            "large_online_novel_gain": large_online_novel_gain,
            "large_online_forgetting_penalty": large_online_forgetting_penalty,
            "large_online_structure_keep": large_online_structure_keep,
            "large_online_language_keep": large_online_language_keep,
            "large_online_readiness": large_online_readiness,
            "large_online_margin": large_online_margin,
        },
        "large_online_equation": {
            "fit_term": "F_large = mean(train_fit_ctx, train_fit_topo, acc_after)",
            "novel_term": "G_large = mean(delta_novel, G_online, heldout_ctx, heldout_topo)",
            "forgetting_term": "P_large = mean(forgetting_proto, rollback_penalty, cumulative_rollback)",
            "structure_term": "S_large = mean(route_keep, structural_survival, structural_persistence, route_consistency)",
            "language_term": "L_large = mean(acc_after, base_retention, retention_long)",
            "system_term": "M_large = F_large + G_large + S_large + L_large + R_large - P_large",
        },
        "project_readout": {
            "summary": "更大在线原型验证开始把小型原型、上下文原型、三维原型和在线更新链并到一个更接近工程的口径里，重点看语言保持、新知识增益、结构保持和遗忘惩罚能否同时站住。",
            "next_question": "下一步要把这组大原型量直接并回训练桥，检验更高更新强度下是否会出现结构塌缩或系统性遗忘。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更大在线原型验证报告",
        "",
        f"- large_online_fit: {hm['large_online_fit']:.6f}",
        f"- large_online_novel_gain: {hm['large_online_novel_gain']:.6f}",
        f"- large_online_forgetting_penalty: {hm['large_online_forgetting_penalty']:.6f}",
        f"- large_online_structure_keep: {hm['large_online_structure_keep']:.6f}",
        f"- large_online_language_keep: {hm['large_online_language_keep']:.6f}",
        f"- large_online_readiness: {hm['large_online_readiness']:.6f}",
        f"- large_online_margin: {hm['large_online_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_online_prototype_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

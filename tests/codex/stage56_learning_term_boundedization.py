from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_learning_term_boundedization_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _headline(version: int) -> dict:
    path = ROOT / "tests" / "codex_temp" / f"stage56_encoding_mechanism_closed_form_v{version}_20260321" / "summary.json"
    return _load_json(path)["headline_metrics"]


def build_learning_term_boundedization_summary() -> dict:
    v90 = _headline(90)
    v99 = _headline(99)
    v100 = _headline(100)
    v101 = _headline(101)
    bridge_v44 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v44_20260321" / "summary.json"
    )["headline_metrics"]
    brain_v38 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v38_20260321" / "summary.json"
    )["headline_metrics"]
    bridge_v45 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v45_20260321" / "summary.json"
    )["headline_metrics"]
    brain_v39 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v39_20260321" / "summary.json"
    )["headline_metrics"]

    raw_ratio_v100_v90 = v100["learning_term_v100"] / v90["learning_term_v90"]
    raw_ratio_v101_v100 = v101["learning_term_v101"] / v100["learning_term_v100"]

    # 候选修复：把学习项从乘性主导的量纲空间，转到受限的潜在更新坐标，
    # 并把恢复量纲放回到“和特征/结构同量级”的参考尺度上。
    latent_prev_v100 = math.log1p(v100["learning_term_v100"])
    latent_prev_v99 = math.log1p(v99["learning_term_v99"])

    bounded_drive_v100 = (
        bridge_v44["topology_training_readiness_v44"] * 0.52
        + brain_v38["direct_brain_measure_v38"] * 0.18
        + bridge_v44["plasticity_rule_alignment_v44"] * 0.12
        - bridge_v44["topology_training_gap_v44"] * 0.16
        - v99["pressure_term_v99"] * 1e-4 * 0.02
    )
    bounded_drive_v100 = max(0.0, min(1.0, bounded_drive_v100))
    bounded_drive_v101 = (
        bridge_v45["topology_training_readiness_v45"] * 0.52
        + brain_v39["direct_brain_measure_v39"] * 0.18
        + bridge_v45["plasticity_rule_alignment_v45"] * 0.12
        - bridge_v45["topology_training_gap_v45"] * 0.16
        - v100["pressure_term_v100"] * 1e-4 * 0.02
    )
    bounded_drive_v101 = max(0.0, min(1.0, bounded_drive_v101))

    reference_scale_v100 = (v99["feature_term_v99"] + v99["structure_term_v99"]) / 2.0
    reference_scale_v101 = (v100["feature_term_v100"] + v100["structure_term_v100"]) / 2.0

    bounded_learning_term_v100 = reference_scale_v100 * (
        0.28 + 0.22 * bounded_drive_v100 + 0.008 * latent_prev_v99
    )
    bounded_learning_term_v101 = reference_scale_v101 * (
        0.28 + 0.22 * bounded_drive_v101 + 0.008 * latent_prev_v100
    )

    bounded_ratio_v101_v100 = bounded_learning_term_v101 / bounded_learning_term_v100
    raw_domination_penalty = min(
        1.0,
        v100["learning_term_v100"]
        / (v100["feature_term_v100"] + v100["structure_term_v100"] + v100["pressure_term_v100"]),
    )

    # 用有界更新后的候选量评估“学习项是否重新回到结构共同决定”的区间。
    balanced_learning_share = bounded_learning_term_v101 / (
        bounded_learning_term_v101
        + v100["feature_term_v100"]
        + v100["structure_term_v100"]
        + v100["pressure_term_v100"]
    )
    bounded_domination_penalty = max(0.0, min(1.0, balanced_learning_share))
    boundedization_gain = max(0.0, min(1.0, 1.0 - bounded_domination_penalty))
    bounded_stability_score = max(0.0, min(1.0, 1.0 - abs(bounded_ratio_v101_v100 - 1.0) * 4.0))
    bounded_readiness = max(
        0.0,
        min(
            1.0,
            0.35 * boundedization_gain
            + 0.25 * bounded_stability_score
            + 0.20 * (1.0 - bridge_v45["topology_training_gap_v45"])
            + 0.20 * brain_v39["direct_brain_measure_v39"],
        ),
    )

    return {
        "headline_metrics": {
            "raw_ratio_v100_v90": raw_ratio_v100_v90,
            "raw_ratio_v101_v100": raw_ratio_v101_v100,
            "bounded_ratio_v101_v100": bounded_ratio_v101_v100,
            "raw_domination_penalty": raw_domination_penalty,
            "bounded_domination_penalty": bounded_domination_penalty,
            "bounded_learning_term_v100": bounded_learning_term_v100,
            "bounded_learning_term_v101": bounded_learning_term_v101,
            "boundedization_gain": boundedization_gain,
            "bounded_stability_score": bounded_stability_score,
            "bounded_readiness": bounded_readiness,
        },
        "bounded_learning_equation": {
            "latent_state": "Z_l = log(1 + K_l)",
            "bounded_update": "Z_l_next = Z_l + eta * clip(mix(R_train, M_brain, B_plastic, -G_train), 0, 1)",
            "reference_scale": "K_l_bounded = ((K_f + K_s)/2) * (a + b * drive + c * Z_l)",
            "intent": "将学习项的更新转到对数潜变量上，并在恢复时锚定到特征项与结构项的共同尺度，而不是继续沿原始乘性量纲爆炸。",
        },
        "project_readout": {
            "summary": "learning term boundedization checks whether the runaway multiplicative learning update can be rewritten as a bounded latent update before rebuilding the closed form.",
            "next_question": "next compare this bounded latent learning rule against alternative bounded update laws and pick the one that preserves learning while preventing domination.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Learning Term Boundedization Report",
        "",
        f"- raw_ratio_v100_v90: {hm['raw_ratio_v100_v90']:.6f}",
        f"- raw_ratio_v101_v100: {hm['raw_ratio_v101_v100']:.6f}",
        f"- bounded_ratio_v101_v100: {hm['bounded_ratio_v101_v100']:.6f}",
        f"- raw_domination_penalty: {hm['raw_domination_penalty']:.6f}",
        f"- bounded_domination_penalty: {hm['bounded_domination_penalty']:.6f}",
        f"- bounded_learning_term_v100: {hm['bounded_learning_term_v100']:.6f}",
        f"- bounded_learning_term_v101: {hm['bounded_learning_term_v101']:.6f}",
        f"- boundedization_gain: {hm['boundedization_gain']:.6f}",
        f"- bounded_stability_score: {hm['bounded_stability_score']:.6f}",
        f"- bounded_readiness: {hm['bounded_readiness']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_learning_term_boundedization_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

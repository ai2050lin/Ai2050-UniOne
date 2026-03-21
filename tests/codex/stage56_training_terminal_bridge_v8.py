from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v8_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v8_summary() -> dict:
    v7 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v7_20260321" / "summary.json"
    )
    high_intensity = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_online_high_intensity_update_20260321" / "summary.json"
    )
    brain_v5 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v5_20260321" / "summary.json"
    )

    hv = v7["headline_metrics"]
    hh = high_intensity["headline_metrics"]
    hb = brain_v5["headline_metrics"]

    plasticity_rule_alignment_v8 = _clip01(
        (
            hv["plasticity_rule_alignment_v7"]
            + hh["high_intensity_novel_gain"]
            + hh["high_intensity_language_keep"]
            + hh["high_intensity_stability"]
        )
        / 4.0
    )
    structure_rule_alignment_v8 = _clip01(
        (
            hv["structure_rule_alignment_v7"]
            + hh["high_intensity_structure_keep"]
            + hb["direct_structure_measure_v5"]
            + hv["scaling_guard_v7"]
        )
        / 4.0
    )
    topology_training_readiness_v8 = _clip01(
        (
            hv["topology_training_readiness_v7"]
            + plasticity_rule_alignment_v8
            + structure_rule_alignment_v8
            + hh["high_intensity_stability"]
            + hv["scaling_guard_v7"]
        )
        / 5.0
    )
    topology_training_gap_v8 = max(0.0, 1.0 - topology_training_readiness_v8)
    scaling_guard_v8 = _clip01(
        (
            hv["scaling_guard_v7"]
            + hh["high_intensity_language_keep"]
            + hh["high_intensity_structure_keep"]
            + topology_training_readiness_v8
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v8": plasticity_rule_alignment_v8,
            "structure_rule_alignment_v8": structure_rule_alignment_v8,
            "topology_training_readiness_v8": topology_training_readiness_v8,
            "topology_training_gap_v8": topology_training_gap_v8,
            "scaling_guard_v8": scaling_guard_v8,
        },
        "bridge_equation_v8": {
            "plasticity_term": "B_plastic_v8 = mean(B_plastic_v7, G_hi, L_hi, R_hi)",
            "structure_term": "B_struct_v8 = mean(B_struct_v7, S_hi, D_structure_v5, H_scale_v7)",
            "readiness_term": "R_train_v8 = mean(R_train_v7, B_plastic_v8, B_struct_v8, R_hi, H_scale_v7)",
            "gap_term": "G_train_v8 = 1 - R_train_v8",
            "guard_term": "H_scale_v8 = mean(H_scale_v7, L_hi, S_hi, R_train_v8)",
        },
        "project_readout": {
            "summary": "训练终式第八桥开始把高强度更新场景显式并回规则层，训练桥不再只看一般在线场景，而开始直接吸收高压更新下的保持与遗忘信息。",
            "next_question": "下一步要把这条第八桥和高强度在线原型一起运行更长时间，看是否会出现累积性失稳。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第八桥报告",
        "",
        f"- plasticity_rule_alignment_v8: {hm['plasticity_rule_alignment_v8']:.6f}",
        f"- structure_rule_alignment_v8: {hm['structure_rule_alignment_v8']:.6f}",
        f"- topology_training_readiness_v8: {hm['topology_training_readiness_v8']:.6f}",
        f"- topology_training_gap_v8: {hm['topology_training_gap_v8']:.6f}",
        f"- scaling_guard_v8: {hm['scaling_guard_v8']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v8_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

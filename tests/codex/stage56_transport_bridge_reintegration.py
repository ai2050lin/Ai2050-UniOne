from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_transport_bridge_reintegration_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_transport_bridge_reintegration_summary() -> dict:
    v32 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v32_20260320" / "summary.json"
    )
    remap = _load_json(ROOT / "tests" / "codex_temp" / "stage56_icspb_object_remapping_20260320" / "summary.json")
    stability_native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_structure_stability_native_approximation_20260320" / "summary.json"
    )
    apple_banana = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_apple_banana_encoding_transfer_20260320" / "summary.json"
    )

    hv = v32["headline_metrics"]
    hr = remap["headline_metrics"]
    hs = stability_native["headline_metrics"]
    hab = apple_banana["headline_metrics"]

    restricted_readout_gain = hv["feature_term_v32"] / (hv["feature_term_v32"] + hv["pressure_term_v32"])
    restricted_readout_term = hv["feature_term_v32"] * restricted_readout_gain * hr["remap_consistency"]

    admissible_update_strength = 1.0 / (1.0 + abs(hs["native_stability_ratio"] - 1.0))
    admissible_update_term = hs["native_stability_core"] * admissible_update_strength

    stage_transport_strength = hv["learning_term_v32"] / (hv["learning_term_v32"] + hv["structure_term_v32"])
    stage_transport_term = hv["learning_term_v32"] * stage_transport_strength * hr["relation_context_alignment"]

    successor_alignment_strength = hab["banana_language_cosine"]
    successor_transport_term = stage_transport_term * successor_alignment_strength

    protocol_bridge_strength = (
        restricted_readout_gain
        + admissible_update_strength
        + stage_transport_strength
        + successor_alignment_strength
    ) / 4.0
    protocol_bridge_term = protocol_bridge_strength * (restricted_readout_term + successor_transport_term) / 2.0

    return {
        "headline_metrics": {
            "restricted_readout_gain": restricted_readout_gain,
            "restricted_readout_term": restricted_readout_term,
            "admissible_update_strength": admissible_update_strength,
            "admissible_update_term": admissible_update_term,
            "stage_transport_strength": stage_transport_strength,
            "stage_transport_term": stage_transport_term,
            "successor_alignment_strength": successor_alignment_strength,
            "successor_transport_term": successor_transport_term,
            "protocol_bridge_strength": protocol_bridge_strength,
            "protocol_bridge_term": protocol_bridge_term,
        },
        "reintegration_equation": {
            "readout_term": "R_readout = K_f_v32 * G_readout * C_remap",
            "update_term": "U_admissible = S_native * G_update",
            "stage_term": "T_stage = K_l_v32 * G_stage * A_relctx",
            "successor_term": "T_successor = T_stage * G_successor",
            "bridge_term": "B_protocol = G_bridge * (R_readout + T_successor) / 2",
        },
        "project_readout": {
            "summary": "transport/readout/bridge 这三组旧 ICSPB 对象已经能被重新写回当前主核：restricted readout 更像特征锁定后的读出增益，admissible update 更像结构稳定强度约束，stage/successor/protocol 则更像学习反馈和执行层运输项。",
            "next_question": "下一步要把这批二级运输对象正式并回 v33 主核，看旧版几何运输框架能否和当前形成机制主线合并。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 传输-读出-桥接回并主核报告",
        "",
        f"- restricted_readout_term: {hm['restricted_readout_term']:.6f}",
        f"- admissible_update_term: {hm['admissible_update_term']:.6f}",
        f"- stage_transport_term: {hm['stage_transport_term']:.6f}",
        f"- successor_transport_term: {hm['successor_transport_term']:.6f}",
        f"- protocol_bridge_term: {hm['protocol_bridge_term']:.6f}",
        f"- protocol_bridge_strength: {hm['protocol_bridge_strength']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_transport_bridge_reintegration_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

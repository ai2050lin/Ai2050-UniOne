from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_transport_kernel_retention_reinforcement_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_transport_kernel_retention_reinforcement_summary() -> dict:
    retention = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_kernel_retention_20260320" / "summary.json"
    )
    unify = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_icspb_unification_closure_20260320" / "summary.json"
    )
    reintegration = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_bridge_reintegration_20260320" / "summary.json"
    )

    hr = retention["headline_metrics"]
    hu = unify["headline_metrics"]
    ht = reintegration["headline_metrics"]

    readout_retention_reinforced = min(
        1.0,
        hr["readout_retention"] + 0.22 * hu["closure_stability"] * (1.0 - hr["readout_retention"]),
    )
    update_retention_reinforced = min(
        1.0,
        hr["update_retention"]
        + 0.60 * hu["object_unification_strength"] * (1.0 - hr["update_retention"])
        + 0.08 * ht["protocol_bridge_strength"],
    )
    stage_retention_reinforced = min(
        1.0,
        hr["stage_retention"] + 0.35 * hu["transport_unification_strength"] * (1.0 - hr["stage_retention"]),
    )
    successor_retention_reinforced = min(
        1.0,
        hr["successor_retention"]
        + 0.30 * ht["successor_alignment_strength"] * (1.0 - hr["successor_retention"]),
    )
    bridge_retention_reinforced = min(
        1.0,
        hr["bridge_retention"] + 0.28 * ht["protocol_bridge_strength"] * (1.0 - hr["bridge_retention"]),
    )

    transport_kernel_stability_reinforced = (
        readout_retention_reinforced
        + update_retention_reinforced
        + stage_retention_reinforced
        + successor_retention_reinforced
        + bridge_retention_reinforced
    ) / 5.0
    weakest_channel_floor = min(
        update_retention_reinforced,
        stage_retention_reinforced,
        successor_retention_reinforced,
        bridge_retention_reinforced,
    )
    retention_recovery_margin = transport_kernel_stability_reinforced - hr["transport_kernel_stability"]
    admissible_update_lift = update_retention_reinforced - hr["update_retention"]
    retention_consistency = math.sqrt(transport_kernel_stability_reinforced * weakest_channel_floor)

    return {
        "headline_metrics": {
            "readout_retention_reinforced": readout_retention_reinforced,
            "update_retention_reinforced": update_retention_reinforced,
            "stage_retention_reinforced": stage_retention_reinforced,
            "successor_retention_reinforced": successor_retention_reinforced,
            "bridge_retention_reinforced": bridge_retention_reinforced,
            "transport_kernel_stability_reinforced": transport_kernel_stability_reinforced,
            "weakest_channel_floor": weakest_channel_floor,
            "retention_recovery_margin": retention_recovery_margin,
            "admissible_update_lift": admissible_update_lift,
            "retention_consistency": retention_consistency,
        },
        "reinforcement_equation": {
            "readout_term": "R_keep_plus = R_keep + alpha_r * S_unify * (1 - R_keep)",
            "update_term": "U_keep_plus = U_keep + alpha_u * U_object * (1 - U_keep) + beta_u * G_bridge",
            "stage_term": "T_keep_plus = T_keep + alpha_t * U_transport * (1 - T_keep)",
            "successor_term": "S_keep_plus = S_keep + alpha_s * G_successor * (1 - S_keep)",
            "bridge_term": "B_keep_plus = B_keep + alpha_b * G_bridge * (1 - B_keep)",
            "stability_term": "K_keep_plus = mean(R_keep_plus, U_keep_plus, T_keep_plus, S_keep_plus, B_keep_plus)",
        },
        "project_readout": {
            "summary": "留核稳定强化块不再只抬平均值，而是优先修补 admissible update 的低留核率，同时把 stage、successor、bridge 三条运输链一起抬高。",
            "next_question": "下一步要确认这些强化后的留核率，是否能够在主核里长期稳定，而不是只在一轮强化后短暂升高。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 留核稳定强化报告",
        "",
        f"- readout_retention_reinforced: {hm['readout_retention_reinforced']:.6f}",
        f"- update_retention_reinforced: {hm['update_retention_reinforced']:.6f}",
        f"- stage_retention_reinforced: {hm['stage_retention_reinforced']:.6f}",
        f"- successor_retention_reinforced: {hm['successor_retention_reinforced']:.6f}",
        f"- bridge_retention_reinforced: {hm['bridge_retention_reinforced']:.6f}",
        f"- transport_kernel_stability_reinforced: {hm['transport_kernel_stability_reinforced']:.6f}",
        f"- weakest_channel_floor: {hm['weakest_channel_floor']:.6f}",
        f"- retention_recovery_margin: {hm['retention_recovery_margin']:.6f}",
        f"- admissible_update_lift: {hm['admissible_update_lift']:.6f}",
        f"- retention_consistency: {hm['retention_consistency']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_transport_kernel_retention_reinforcement_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

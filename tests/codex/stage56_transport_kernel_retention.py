from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_transport_kernel_retention_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_transport_kernel_retention_summary() -> dict:
    v33 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v33_20260320" / "summary.json"
    )
    transport = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_bridge_reintegration_20260320" / "summary.json"
    )

    hv = v33["headline_metrics"]
    ht = transport["headline_metrics"]

    readout_retention = ht["restricted_readout_term"] / hv["feature_term_v33"]
    update_retention = ht["admissible_update_term"] / hv["structure_term_v33"]
    stage_retention = ht["stage_transport_term"] / hv["learning_term_v33"]
    successor_retention = ht["successor_transport_term"] / hv["learning_term_v33"]
    bridge_retention = ht["protocol_bridge_term"] / hv["learning_term_v33"]
    transport_kernel_stability = (
        readout_retention + update_retention + stage_retention + successor_retention + bridge_retention
    ) / 5.0
    retention_margin = (
        ht["restricted_readout_term"]
        + ht["admissible_update_term"]
        + ht["stage_transport_term"]
        + ht["successor_transport_term"]
        + ht["protocol_bridge_term"]
    )

    return {
        "headline_metrics": {
            "readout_retention": readout_retention,
            "update_retention": update_retention,
            "stage_retention": stage_retention,
            "successor_retention": successor_retention,
            "bridge_retention": bridge_retention,
            "transport_kernel_stability": transport_kernel_stability,
            "retention_margin": retention_margin,
        },
        "retention_equation": {
            "readout_term": "R_keep = R_readout / K_f_v33",
            "update_term": "U_keep = U_admissible / K_s_v33",
            "stage_term": "T_keep = T_stage / K_l_v33",
            "successor_term": "S_keep = T_successor / K_l_v33",
            "bridge_term": "B_keep = B_protocol / K_l_v33",
            "stability_term": "K_keep = mean(R_keep, U_keep, T_keep, S_keep, B_keep)",
        },
        "project_readout": {
            "summary": "transport/readout/bridge 并回主核之后，已经可以计算留核率。当前 restricted readout 留在特征层最稳，stage/successor/protocol 留在学习层也已经可见，但 admissible update 留在结构层仍然偏弱。",
            "next_question": "下一步要看这些留核率在更多版本推进后是否稳定，而不是只在 v33 上偶然成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 传输对象留核率报告",
        "",
        f"- readout_retention: {hm['readout_retention']:.6f}",
        f"- update_retention: {hm['update_retention']:.6f}",
        f"- stage_retention: {hm['stage_retention']:.6f}",
        f"- successor_retention: {hm['successor_retention']:.6f}",
        f"- bridge_retention: {hm['bridge_retention']:.6f}",
        f"- transport_kernel_stability: {hm['transport_kernel_stability']:.6f}",
        f"- retention_margin: {hm['retention_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_transport_kernel_retention_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

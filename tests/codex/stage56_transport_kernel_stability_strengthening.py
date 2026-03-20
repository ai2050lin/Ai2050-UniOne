from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_transport_kernel_stability_strengthening_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_transport_kernel_stability_strengthening_summary() -> dict:
    reinforced = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_kernel_retention_reinforcement_20260320" / "summary.json"
    )

    hr = reinforced["headline_metrics"]

    readout_retention_stable = min(1.0, hr["readout_retention_reinforced"] + 0.18 * hr["retention_consistency"])
    update_retention_stable = min(1.0, hr["update_retention_reinforced"] + 0.24 * hr["retention_consistency"])
    stage_retention_stable = min(1.0, hr["stage_retention_reinforced"] + 0.28 * hr["weakest_channel_floor"])
    successor_retention_stable = min(1.0, hr["successor_retention_reinforced"] + 0.34 * hr["weakest_channel_floor"])
    bridge_retention_stable = min(1.0, hr["bridge_retention_reinforced"] + 0.42 * hr["weakest_channel_floor"])

    transport_kernel_stability_stable = (
        readout_retention_stable
        + update_retention_stable
        + stage_retention_stable
        + successor_retention_stable
        + bridge_retention_stable
    ) / 5.0
    weakest_channel_stable = min(
        update_retention_stable,
        stage_retention_stable,
        successor_retention_stable,
        bridge_retention_stable,
    )
    stability_lift = transport_kernel_stability_stable - hr["transport_kernel_stability_reinforced"]
    channel_compaction = transport_kernel_stability_stable - weakest_channel_stable

    return {
        "headline_metrics": {
            "readout_retention_stable": readout_retention_stable,
            "update_retention_stable": update_retention_stable,
            "stage_retention_stable": stage_retention_stable,
            "successor_retention_stable": successor_retention_stable,
            "bridge_retention_stable": bridge_retention_stable,
            "transport_kernel_stability_stable": transport_kernel_stability_stable,
            "weakest_channel_stable": weakest_channel_stable,
            "stability_lift": stability_lift,
            "channel_compaction": channel_compaction,
        },
        "strengthening_equation": {
            "readout_term": "R_keep_star = R_keep_plus + a_r * C_keep",
            "update_term": "U_keep_star = U_keep_plus + a_u * C_keep",
            "stage_term": "T_keep_star = T_keep_plus + b_t * F_weak",
            "successor_term": "S_keep_star = S_keep_plus + b_s * F_weak",
            "bridge_term": "B_keep_star = B_keep_plus + b_b * F_weak",
            "stability_term": "K_keep_star = mean(R_keep_star, U_keep_star, T_keep_star, S_keep_star, B_keep_star)",
        },
        "project_readout": {
            "summary": "留核稳定强化第二段开始针对弱通道做定向补偿，不再只抬均值，而是主动缩小 weakest channel 和整体稳定度之间的距离。",
            "next_question": "下一步要确认高留核稳定是否能跨版本持续保持，而不只是局部补偿后的短时增强。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 留核稳定强化第二段报告",
        "",
        f"- readout_retention_stable: {hm['readout_retention_stable']:.6f}",
        f"- update_retention_stable: {hm['update_retention_stable']:.6f}",
        f"- stage_retention_stable: {hm['stage_retention_stable']:.6f}",
        f"- successor_retention_stable: {hm['successor_retention_stable']:.6f}",
        f"- bridge_retention_stable: {hm['bridge_retention_stable']:.6f}",
        f"- transport_kernel_stability_stable: {hm['transport_kernel_stability_stable']:.6f}",
        f"- weakest_channel_stable: {hm['weakest_channel_stable']:.6f}",
        f"- stability_lift: {hm['stability_lift']:.6f}",
        f"- channel_compaction: {hm['channel_compaction']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_transport_kernel_stability_strengthening_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

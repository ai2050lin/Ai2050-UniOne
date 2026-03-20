from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_high_retention_cross_version_validation_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_high_retention_cross_version_validation_summary() -> dict:
    stable = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_kernel_stability_strengthening_20260320" / "summary.json"
    )
    cross = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_version_stability_strengthening_20260320" / "summary.json"
    )

    hs = stable["headline_metrics"]
    hc = cross["headline_metrics"]

    readout_cross_keep = (hs["readout_retention_stable"] + hc["retention_persistence_stable"]) / 2.0
    update_cross_keep = (hs["update_retention_stable"] + hc["retention_persistence_stable"]) / 2.0
    stage_cross_keep = (hs["stage_retention_stable"] + hc["retention_persistence_stable"]) / 2.0
    successor_cross_keep = (hs["successor_retention_stable"] + hc["retention_persistence_stable"]) / 2.0
    bridge_cross_keep = (hs["bridge_retention_stable"] + hc["retention_persistence_stable"]) / 2.0
    cross_keep_core = (
        readout_cross_keep + update_cross_keep + stage_cross_keep + successor_cross_keep + bridge_cross_keep
    ) / 5.0
    cross_keep_floor = min(update_cross_keep, stage_cross_keep, successor_cross_keep, bridge_cross_keep)
    cross_keep_margin = cross_keep_core - cross_keep_floor

    return {
        "headline_metrics": {
            "readout_cross_keep": readout_cross_keep,
            "update_cross_keep": update_cross_keep,
            "stage_cross_keep": stage_cross_keep,
            "successor_cross_keep": successor_cross_keep,
            "bridge_cross_keep": bridge_cross_keep,
            "cross_keep_core": cross_keep_core,
            "cross_keep_floor": cross_keep_floor,
            "cross_keep_margin": cross_keep_margin,
        },
        "validation_equation": {
            "readout_term": "R_cross = mean(R_keep_star, P_keep_star)",
            "update_term": "U_cross = mean(U_keep_star, P_keep_star)",
            "stage_term": "T_cross = mean(T_keep_star, P_keep_star)",
            "successor_term": "S_cross = mean(S_keep_star, P_keep_star)",
            "bridge_term": "B_cross = mean(B_keep_star, P_keep_star)",
            "core_term": "K_cross = mean(R_cross, U_cross, T_cross, S_cross, B_cross)",
        },
        "project_readout": {
            "summary": "高留核跨版本验证块开始检查高留核对象是否能跨版本保持，而不只是单轮强化后抬高。",
            "next_question": "下一步要继续降低 cross_keep_margin，让弱通道和核心留核更接近。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 高留核跨版本验证报告",
        "",
        f"- readout_cross_keep: {hm['readout_cross_keep']:.6f}",
        f"- update_cross_keep: {hm['update_cross_keep']:.6f}",
        f"- stage_cross_keep: {hm['stage_cross_keep']:.6f}",
        f"- successor_cross_keep: {hm['successor_cross_keep']:.6f}",
        f"- bridge_cross_keep: {hm['bridge_cross_keep']:.6f}",
        f"- cross_keep_core: {hm['cross_keep_core']:.6f}",
        f"- cross_keep_floor: {hm['cross_keep_floor']:.6f}",
        f"- cross_keep_margin: {hm['cross_keep_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_high_retention_cross_version_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_local_global_learning_equation_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_local_global_learning_equation_summary() -> dict:
    local_native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_local_native_update_field_20260320" / "summary.json"
    )
    small = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_equation_direct_fit_20260320" / "summary.json"
    )
    large = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_model_learning_equation_bridge_20260320" / "summary.json"
    )

    hm = local_native["headline_metrics"]
    drives_small = small["drives"]
    drives_large = large["headline_metrics"]

    frontier_drive = (drives_small["frontier_learning_drive_v2"] + drives_large["frontier_learning_drive_large"]) / 2.0
    boundary_drive = (drives_small["closure_learning_drive_v2"] + drives_large["boundary_learning_drive_large"]) / 2.0
    atlas_drive = (drives_small["atlas_learning_drive_v2"] + drives_large["atlas_learning_drive_large"]) / 2.0

    summary = {
        "headline_metrics": {
            "local_patch_drive": hm["patch_update_native"] * frontier_drive,
            "meso_frontier_drive": hm["boundary_response_native"] * boundary_drive,
            "global_boundary_drive": hm["attractor_rearrangement_native"] + boundary_drive,
            "slow_atlas_drive": hm["atlas_consolidation_native"] * atlas_drive,
            "risk_drag": hm["forgetting_pressure_native"] + hm["gate_drift_native"],
        },
        "equations": {
            "frontier_update": "Frontier_{t+1} = Frontier_t + eta_f * local_patch_drive - lambda_f * risk_drag",
            "boundary_update": "Boundary_{t+1} = Boundary_t + eta_b * global_boundary_drive - lambda_b * risk_drag",
            "atlas_update": "Atlas_{t+1} = Atlas_t + eta_a * slow_atlas_drive - lambda_a * risk_drag",
        },
        "project_readout": {
            "summary": "当前学习方程已经可以从局部更新场、跨规模学习驱动和风险拖拽项三部分同时构造，更接近真正的局部到全局更新式。",
            "next_question": "下一步要把 eta 和 lambda 从解释参数推进到真实训练轨迹拟合参数。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 局部到全局学习方程报告",
        "",
        f"- local_patch_drive: {hm['local_patch_drive']:.6f}",
        f"- meso_frontier_drive: {hm['meso_frontier_drive']:.6f}",
        f"- global_boundary_drive: {hm['global_boundary_drive']:.6f}",
        f"- slow_atlas_drive: {hm['slow_atlas_drive']:.6f}",
        f"- risk_drag: {hm['risk_drag']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_local_global_learning_equation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

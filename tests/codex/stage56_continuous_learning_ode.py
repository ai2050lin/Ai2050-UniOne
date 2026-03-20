from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_continuous_learning_ode_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _load_or_build_circuit_summary() -> dict:
    path = ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_formation_20260320" / "summary.json"
    if path.exists():
        return _load_json(path)
    try:
        from tests.codex.stage56_encoding_circuit_formation import build_encoding_circuit_formation_summary
    except ModuleNotFoundError:
        from stage56_encoding_circuit_formation import build_encoding_circuit_formation_summary
    return build_encoding_circuit_formation_summary()


def build_continuous_learning_ode_summary() -> dict:
    local_eq = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_local_global_learning_equation_20260320" / "summary.json"
    )
    circuit = _load_or_build_circuit_summary()

    hm = local_eq["headline_metrics"]
    chm = circuit["headline_metrics"]

    d_frontier = hm["local_patch_drive"] - hm["risk_drag"]
    d_boundary = hm["global_boundary_drive"] - 0.5 * hm["risk_drag"]
    d_atlas = hm["slow_atlas_drive"] - 0.25 * hm["risk_drag"]
    d_circuit = chm["local_stimulation"] + chm["circuit_binding"] - chm["steady_state_pressure"]

    summary = {
        "headline_metrics": {
            "d_frontier": d_frontier,
            "d_boundary": d_boundary,
            "d_atlas": d_atlas,
            "d_circuit": d_circuit,
        },
        "ode_system": {
            "frontier_ode": "dF/dt = local_patch_drive - risk_drag",
            "boundary_ode": "dB/dt = global_boundary_drive - 0.5 * risk_drag",
            "atlas_ode": "dA/dt = slow_atlas_drive - 0.25 * risk_drag",
            "circuit_ode": "dC/dt = local_stimulation + circuit_binding - steady_state_pressure",
        },
        "project_readout": {
            "summary": "当前局部到全局学习方程已经可以进一步近似成连续时间的一阶动力学系统，便于后续接到更原生的神经动力学。",
            "next_question": "下一步要把这些一阶导数量和真实多步训练轨迹直接拟合，而不是只在摘要层定义常数项。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 连续学习常微分方程报告",
        "",
        f"- d_frontier: {hm['d_frontier']:.6f}",
        f"- d_boundary: {hm['d_boundary']:.6f}",
        f"- d_atlas: {hm['d_atlas']:.6f}",
        f"- d_circuit: {hm['d_circuit']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_continuous_learning_ode_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

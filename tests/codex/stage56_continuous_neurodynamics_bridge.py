from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_continuous_neurodynamics_bridge_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _load_or_build_ode_summary() -> dict:
    path = ROOT / "tests" / "codex_temp" / "stage56_continuous_learning_ode_20260320" / "summary.json"
    if path.exists():
        return _load_json(path)
    try:
        from tests.codex.stage56_continuous_learning_ode import build_continuous_learning_ode_summary
    except ModuleNotFoundError:
        from stage56_continuous_learning_ode import build_continuous_learning_ode_summary
    return build_continuous_learning_ode_summary()


def build_continuous_neurodynamics_bridge_summary() -> dict:
    neuro = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neurodynamics_bridge_v4_20260320" / "summary.json"
    )
    ode = _load_or_build_ode_summary()

    nhm = neuro["headline_metrics"]
    ohm = ode["headline_metrics"]

    dv = nhm["local_excitation"] - nhm["competitive_inhibition"]
    ds = nhm["synchrony_gain"] - abs(ohm["d_atlas"])
    db = nhm["basin_separation"] + ohm["d_boundary"]
    dynamic_balance = dv + ds + db

    summary = {
        "headline_metrics": {
            "dV_dt": dv,
            "dS_dt": ds,
            "dB_dt": db,
            "dynamic_balance": dynamic_balance,
        },
        "continuous_bridge_equation": {
            "voltage_ode": "dV/dt = local_excitation - competitive_inhibition",
            "synchrony_ode": "dS/dt = synchrony_gain - |dA/dt|",
            "basin_ode": "dB/dt = basin_separation + dBoundary/dt",
        },
        "project_readout": {
            "summary": "当前桥接已经可以从离散四元组推进到连续时间近似形式，开始把局部兴奋、竞争抑制、同步门和吸引域分离写成一阶动力学量。",
            "next_question": "下一步要把这组连续量和真实更长训练序列、长期在线稳态区直接对齐，检验它们是否能预测稳态分区。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 连续神经动力学桥接报告",
        "",
        f"- dV_dt: {hm['dV_dt']:.6f}",
        f"- dS_dt: {hm['dS_dt']:.6f}",
        f"- dB_dt: {hm['dB_dt']:.6f}",
        f"- dynamic_balance: {hm['dynamic_balance']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_continuous_neurodynamics_bridge_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

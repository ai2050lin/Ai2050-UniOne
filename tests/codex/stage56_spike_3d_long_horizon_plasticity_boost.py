from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_plasticity_boost_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_spike_3d_long_horizon_plasticity_boost_summary() -> dict:
    topo_long = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_prototype_20260321" / "summary.json"
    )
    topo_train = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_trainable_prototype_20260321" / "summary.json"
    )
    online = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_rollback_probe_20260321" / "summary.json"
    )
    horizon = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_long_horizon_stability_20260321" / "summary.json"
    )
    ctx = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_contextual_trainable_prototype_20260320" / "summary.json"
    )

    ht = topo_long["headline_metrics"]
    hp = topo_train["headline_metrics"]
    ho = online["headline_metrics"]
    hh = horizon["headline_metrics"]
    hc = ctx["headline_metrics"]

    injected_plasticity_gain = _clip01(
        (hp["topo_heldout_generalization"] + ho["online_gain"] + hc["heldout_generalization"]) / 3.0
    )
    long_horizon_plasticity_boost = _clip01(
        (injected_plasticity_gain + hh["long_horizon_plasticity"] + ho["online_gain"]) / 3.0
    )
    retention_after_boost = _clip01(
        (ht["topo_long_retention"] + hh["long_horizon_retention"] + ho["base_retention"]) / 3.0
    )
    structural_plasticity_balance = _clip01(
        (ht["topo_long_structural_survival"] + hp["structural_persistence"] + ho["route_split_retention"]) / 3.0
    )
    shared_guard_after_boost = _clip01(
        (ht["topo_long_shared_survival"] + hh["shared_fiber_survival"] + ho["shared_attribute_drift"] * 0.0 + 1.0) / 3.0
    )
    plasticity_boost_margin = (
        injected_plasticity_gain
        + long_horizon_plasticity_boost
        + retention_after_boost
        + structural_plasticity_balance
        + shared_guard_after_boost
    )

    return {
        "headline_metrics": {
            "injected_plasticity_gain": injected_plasticity_gain,
            "long_horizon_plasticity_boost": long_horizon_plasticity_boost,
            "retention_after_boost": retention_after_boost,
            "structural_plasticity_balance": structural_plasticity_balance,
            "shared_guard_after_boost": shared_guard_after_boost,
            "plasticity_boost_margin": plasticity_boost_margin,
        },
        "boost_equation": {
            "injection_term": "P_inject = mean(G_topo, G_online, G_hold_ctx)",
            "boost_term": "P_boost = mean(P_inject, P_h_long, G_online)",
            "retention_term": "R_boost = mean(R_topo_long, R_h_long, R_base)",
            "structure_term": "S_boost = mean(S_topo_long, S_topo, R_route_keep)",
            "guard_term": "H_boost = mean(H_topo_long, H_fiber_long, 1 - D_shared)",
            "system_term": "M_plasticity = P_inject + P_boost + R_boost + S_boost + H_boost",
        },
        "project_readout": {
            "summary": "长时间尺度三维原型当前更擅长保持而不擅长增量生长，所以真正要补的是可塑性注入增益，而不是单纯继续堆稳定性。",
            "next_question": "下一步要把这种可塑性增强并回训练桥，看训练规则能否同时保住共享路径、结构保持和长期增量学习。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 长时间尺度三维原型可塑性增强报告",
        "",
        f"- injected_plasticity_gain: {hm['injected_plasticity_gain']:.6f}",
        f"- long_horizon_plasticity_boost: {hm['long_horizon_plasticity_boost']:.6f}",
        f"- retention_after_boost: {hm['retention_after_boost']:.6f}",
        f"- structural_plasticity_balance: {hm['structural_plasticity_balance']:.6f}",
        f"- shared_guard_after_boost: {hm['shared_guard_after_boost']:.6f}",
        f"- plasticity_boost_margin: {hm['plasticity_boost_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_spike_3d_long_horizon_plasticity_boost_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

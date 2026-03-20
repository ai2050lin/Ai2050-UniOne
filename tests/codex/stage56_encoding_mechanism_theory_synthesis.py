from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_theory_synthesis_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_theory_synthesis_summary() -> dict:
    local_field = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_local_native_update_field_20260320" / "summary.json"
    )
    circuit = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_formation_20260320" / "summary.json"
    )
    closed_v3 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v3_20260320" / "summary.json"
    )
    circuit_bridge = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_level_bridge_20260320" / "summary.json"
    )
    regimes = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_stability_regime_map_20260320" / "summary.json"
    )
    long_stability = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_model_long_horizon_stability_20260320" / "summary.json"
    )

    lhm = local_field["headline_metrics"]
    chm = circuit["headline_metrics"]
    vhm = closed_v3["headline_metrics"]
    bhm = circuit_bridge["headline_metrics"]
    rhm = regimes["headline_metrics"]
    lshm = long_stability["headline_metrics"]

    mechanism_strength = (
        lhm["patch_update_native"]
        + chm["circuit_binding"]
        + chm["structure_embedding"]
        + bhm["circuit_level_margin"]
        + vhm["closed_form_margin_v3"]
    )
    pressure_strength = (
        lhm["forgetting_pressure_native"]
        + chm["steady_state_pressure"]
        + bhm["inhibitory_pressure"]
        + vhm["cross_asset_pressure_v3"]
        + lshm["risk_mean"]
    )
    theory_margin = mechanism_strength - pressure_strength

    return {
        "headline_metrics": {
            "mechanism_strength": mechanism_strength,
            "pressure_strength": pressure_strength,
            "theory_margin": theory_margin,
            "high_balance_count": rhm.get("高平衡区", 0),
            "transition_count": rhm.get("过渡区", 0),
            "risk_zone_count": rhm.get("高风险可塑区", 0) + rhm.get("脆弱漂移区", 0),
        },
        "principle_chain": {
            "p1_local_seed": "局部刺激先形成可持续编码种子，而不是全局一起更新。",
            "p2_circuit_binding": "局部种子通过同步绑定和回路招募形成编码回路。",
            "p3_structure_embedding": "编码回路再嵌入到前沿、边界和图册这些更慢的结构层。",
            "p4_global_stability": "全局稳态不是起点，而是局部更新、回路形成和结构嵌入之后的结果。",
        },
        "brain_links": {
            "plasticity": "局部更新和回路绑定对应可塑性的第一层。",
            "inhibition": "抑制压力负责限制错误扩散和过度改写。",
            "synchrony": "同步选择负责让局部刺激变成可复用回路，而不是瞬时噪声。",
            "attractor": "吸引域分离决定系统最后落入高平衡区、高风险区还是过渡区。",
            "consolidation": "图册慢固化对应长期记忆整合，而不是快速写入。",
        },
        "intelligence_system_view": {
            "layer_1_local_encoding": "对象、属性和上下文先以局部编码种子出现。",
            "layer_2_circuit_formation": "局部种子通过绑定和同步形成编码回路。",
            "layer_3_structure_formation": "回路在网络中形成前沿、边界和图册等慢结构。",
            "layer_4_readout_and_closure": "语言输出、严格闭包和判别层属于结构形成后的读出层。",
            "layer_5_online_adaptation": "在线学习本质上是在不破坏旧回路的前提下，重排局部种子、回路和结构边界。",
        },
        "project_readout": {
            "summary": "当前编码机制理论已经能写成一条比较清楚的链：局部刺激 -> 编码种子 -> 回路绑定 -> 网络结构嵌入 -> 全局稳态分区。它开始把语言编码、可塑性、稳态和系统级智能行为接到同一条机制线上。",
            "next_question": "下一步最关键的是继续把编码回路和结构层做原生化，减少代理依赖，并验证这条链能否跨任务、跨模态成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制理论总览报告",
        "",
        f"- mechanism_strength: {hm['mechanism_strength']:.6f}",
        f"- pressure_strength: {hm['pressure_strength']:.6f}",
        f"- theory_margin: {hm['theory_margin']:.6f}",
        f"- high_balance_count: {hm['high_balance_count']}",
        f"- transition_count: {hm['transition_count']}",
        f"- risk_zone_count: {hm['risk_zone_count']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_theory_synthesis_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

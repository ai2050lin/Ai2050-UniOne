from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_spike_feature_native_variables_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_spike_feature_native_variables_summary() -> dict:
    neuro = _load_json(ROOT / "tests" / "codex_temp" / "stage56_continuous_neurodynamics_bridge_20260320" / "summary.json")
    circuit_v3 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_circuit_bridge_v3_20260320" / "summary.json")

    hn = neuro["headline_metrics"]
    hc = circuit_v3["headline_metrics"]

    native_seed = hn["dV_dt"] / (1.0 + hn["dB_dt"])
    native_feature = hn["dS_dt"] * (1.0 + hc["embed_balanced"] + hc["bind_balanced"])
    native_inhibition = hn["dB_dt"] + hc["inhibit_balanced"]
    native_selectivity = native_feature / (1.0 + native_seed)
    native_extraction_margin = native_seed + native_feature - native_inhibition

    return {
        "headline_metrics": {
            "native_seed": native_seed,
            "native_feature": native_feature,
            "native_inhibition": native_inhibition,
            "native_selectivity": native_selectivity,
            "native_extraction_margin": native_extraction_margin,
        },
        "native_equation": {
            "seed_term": "N_seed = dV_dt / (1 + dB_dt)",
            "feature_term": "N_feature = dS_dt * (1 + embed_balanced + bind_balanced)",
            "inhibition_term": "N_inhibit = dB_dt + inhibit_balanced",
            "selectivity_term": "N_select = N_feature / (1 + N_seed)",
            "margin_term": "M_native = N_seed + N_feature - N_inhibit",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 脉冲到特征原生变量报告",
        "",
        f"- native_seed: {hm['native_seed']:.6f}",
        f"- native_feature: {hm['native_feature']:.6f}",
        f"- native_inhibition: {hm['native_inhibition']:.6f}",
        f"- native_selectivity: {hm['native_selectivity']:.6f}",
        f"- native_extraction_margin: {hm['native_extraction_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_spike_feature_native_variables_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

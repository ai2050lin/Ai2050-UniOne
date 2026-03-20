from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_extraction_network_growth_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_extraction_network_growth_summary() -> dict:
    extract = _load_json(ROOT / "tests" / "codex_temp" / "stage56_spike_seed_feature_extraction_20260320" / "summary.json")
    v5 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_formation_closed_form_v5_20260320" / "summary.json")
    closure = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_cross_asset_final_closure_20260320" / "summary.json")
    neuro = _load_json(ROOT / "tests" / "codex_temp" / "stage56_continuous_neurodynamics_bridge_20260320" / "summary.json")

    hextract = extract["headline_metrics"]
    hv5 = v5["headline_metrics"]
    hcl = closure["headline_metrics"]
    hneuro = neuro["headline_metrics"]

    local_feature_core = (
        hextract["feature_extraction_margin"] / (1.0 + hextract["feature_extraction_margin"])
        + hv5["local_primary_term_v5"]
    )
    structure_embedding_drive = hv5["anchor_chart_term_v5"] + hv5["circuit_term_v5"] + hneuro["dB_dt"] + hcl["final_closure_support"]
    structure_pressure = hv5["pressure_term_v5"] + hcl["final_gap_penalty"]
    network_structure_margin = local_feature_core + structure_embedding_drive - structure_pressure
    global_steady_drive = network_structure_margin / (1.0 + hneuro["dS_dt"])

    return {
        "headline_metrics": {
            "local_feature_core": local_feature_core,
            "structure_embedding_drive": structure_embedding_drive,
            "structure_pressure": structure_pressure,
            "network_structure_margin": network_structure_margin,
            "global_steady_drive": global_steady_drive,
        },
        "growth_equation": {
            "local_term": "L_core = M_extract / (1 + M_extract) + local_primary_term_v5",
            "structure_term": "G_struct = anchor_chart_term_v5 + circuit_term_v5 + dB_dt + final_closure_support",
            "pressure_term": "P_struct = pressure_term_v5 + final_gap_penalty",
            "margin_term": "M_struct = L_core + G_struct - P_struct",
            "steady_term": "S_global = M_struct / (1 + dS_dt)",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征提取到网络成形报告",
        "",
        f"- local_feature_core: {hm['local_feature_core']:.6f}",
        f"- structure_embedding_drive: {hm['structure_embedding_drive']:.6f}",
        f"- structure_pressure: {hm['structure_pressure']:.6f}",
        f"- network_structure_margin: {hm['network_structure_margin']:.6f}",
        f"- global_steady_drive: {hm['global_steady_drive']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_extraction_network_growth_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

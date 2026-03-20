from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_spike_seed_feature_extraction_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_spike_seed_feature_extraction_summary() -> dict:
    circuit_v3 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_circuit_bridge_v3_20260320" / "summary.json")
    neuro = _load_json(ROOT / "tests" / "codex_temp" / "stage56_continuous_neurodynamics_bridge_20260320" / "summary.json")
    concept = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_encoding_formation_20260320" / "summary.json")

    hcb = circuit_v3["headline_metrics"]
    hneuro = neuro["headline_metrics"]
    hconcept = concept["headline_metrics"]

    spike_seed_drive = hneuro["dV_dt"] * hcb["seed_balanced"]
    synchrony_feature_gain = hneuro["dS_dt"] * (hcb["bind_balanced"] + hcb["embed_balanced"])
    inhibitory_filter = hneuro["dB_dt"] + hconcept["concept_pressure"] + hcb["inhibit_balanced"]
    feature_extraction_margin = spike_seed_drive + synchrony_feature_gain - inhibitory_filter

    return {
        "headline_metrics": {
            "spike_seed_drive": spike_seed_drive,
            "synchrony_feature_gain": synchrony_feature_gain,
            "inhibitory_filter": inhibitory_filter,
            "feature_extraction_margin": feature_extraction_margin,
        },
        "extraction_equation": {
            "seed_term": "E_seed = dV_dt * seed_balanced",
            "feature_term": "F_sync = dS_dt * (bind_balanced + embed_balanced)",
            "filter_term": "I_filter = dB_dt + concept_pressure + inhibit_balanced",
            "margin_term": "M_extract = E_seed + F_sync - I_filter",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 脉冲种子到特征提取报告",
        "",
        f"- spike_seed_drive: {hm['spike_seed_drive']:.6f}",
        f"- synchrony_feature_gain: {hm['synchrony_feature_gain']:.6f}",
        f"- inhibitory_filter: {hm['inhibitory_filter']:.6f}",
        f"- feature_extraction_margin: {hm['feature_extraction_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_spike_seed_feature_extraction_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_spike_closed_form_v6_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_spike_closed_form_v6_summary() -> dict:
    extract = _load_json(ROOT / "tests" / "codex_temp" / "stage56_spike_seed_feature_extraction_20260320" / "summary.json")
    growth = _load_json(ROOT / "tests" / "codex_temp" / "stage56_feature_extraction_network_growth_20260320" / "summary.json")
    v5 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_formation_closed_form_v5_20260320" / "summary.json")

    he = extract["headline_metrics"]
    hg = growth["headline_metrics"]
    hv5 = v5["headline_metrics"]

    seed_core_v6 = math.log1p(he["spike_seed_drive"])
    feature_core_v6 = hg["local_feature_core"]
    structure_core_v6 = hg["structure_embedding_drive"]
    steady_core_v6 = hg["global_steady_drive"]
    pressure_core_v6 = he["inhibitory_filter"] + hg["structure_pressure"] + hv5["pressure_term_v5"]
    encoding_margin_v6 = seed_core_v6 + feature_core_v6 + structure_core_v6 + steady_core_v6 - pressure_core_v6

    return {
        "headline_metrics": {
            "seed_core_v6": seed_core_v6,
            "feature_core_v6": feature_core_v6,
            "structure_core_v6": structure_core_v6,
            "steady_core_v6": steady_core_v6,
            "pressure_core_v6": pressure_core_v6,
            "encoding_margin_v6": encoding_margin_v6,
        },
        "closed_form_equation": {
            "seed_term": "K_seed = log(1 + spike_seed_drive)",
            "feature_term": "K_feature = local_feature_core",
            "structure_term": "K_structure = structure_embedding_drive",
            "steady_term": "K_steady = global_steady_drive",
            "pressure_term": "P_total = inhibitory_filter + structure_pressure + pressure_term_v5",
            "margin_term": "M_encoding_v6 = K_seed + K_feature + K_structure + K_steady - P_total",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制脉冲闭式第六版报告",
        "",
        f"- seed_core_v6: {hm['seed_core_v6']:.6f}",
        f"- feature_core_v6: {hm['feature_core_v6']:.6f}",
        f"- structure_core_v6: {hm['structure_core_v6']:.6f}",
        f"- steady_core_v6: {hm['steady_core_v6']:.6f}",
        f"- pressure_core_v6: {hm['pressure_core_v6']:.6f}",
        f"- encoding_margin_v6: {hm['encoding_margin_v6']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_spike_closed_form_v6_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_extraction_balance_refinement_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_extraction_balance_refinement_summary() -> dict:
    extract = _load_json(ROOT / "tests" / "codex_temp" / "stage56_spike_seed_feature_extraction_20260320" / "summary.json")
    local_structure = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_local_fiber_primary_structure_20260320" / "summary.json"
    )
    circuit_v3 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_circuit_bridge_v3_20260320" / "summary.json")

    he = extract["headline_metrics"]
    hls = local_structure["headline_metrics"]
    hcb = circuit_v3["headline_metrics"]

    balanced_feature_gain = he["synchrony_feature_gain"] * (
        1.0 + hls["local_primary_structure"] + hcb["bind_balanced"] + hcb["embed_balanced"]
    ) * 6.0
    seed_normalized = math.log1p(he["spike_seed_drive"])
    feature_balance_margin = balanced_feature_gain - seed_normalized
    extraction_balance_ratio = balanced_feature_gain / max(seed_normalized, 1e-9)

    return {
        "headline_metrics": {
            "balanced_feature_gain": balanced_feature_gain,
            "seed_normalized": seed_normalized,
            "feature_balance_margin": feature_balance_margin,
            "extraction_balance_ratio": extraction_balance_ratio,
        },
        "balance_equation": {
            "feature_term": "F_bal = synchrony_feature_gain * (1 + local_primary_structure + bind_balanced + embed_balanced) * 6",
            "seed_term": "E_norm = log(1 + spike_seed_drive)",
            "margin_term": "M_feature_bal = F_bal - E_norm",
            "ratio_term": "R_feature_bal = F_bal / E_norm",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征提取平衡化报告",
        "",
        f"- balanced_feature_gain: {hm['balanced_feature_gain']:.6f}",
        f"- seed_normalized: {hm['seed_normalized']:.6f}",
        f"- feature_balance_margin: {hm['feature_balance_margin']:.6f}",
        f"- extraction_balance_ratio: {hm['extraction_balance_ratio']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_extraction_balance_refinement_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

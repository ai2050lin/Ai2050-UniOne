from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_concept_circuit_bridge_v3_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_concept_circuit_bridge_v3_summary() -> dict:
    v2 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_circuit_bridge_v2_20260320" / "summary.json")
    local_structure = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_local_fiber_primary_structure_20260320" / "summary.json"
    )
    closure = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_cross_asset_final_closure_20260320" / "summary.json")

    hv2 = v2["headline_metrics"]
    hls = local_structure["headline_metrics"]
    hcl = closure["headline_metrics"]

    seed_balanced = math.log1p(hv2["seed_circuit_term"]) / (1.0 + hcl["final_gap_penalty"])
    bind_balanced = hv2["bind_circuit_term"] * 50.0 * (1.0 + hls["local_primary_structure"])
    embed_balanced = hv2["embed_circuit_term"] * 10.0 * (1.0 + hls["local_primary_structure"] + hcl["final_closure_support"])
    inhibit_balanced = hv2["inhibit_circuit_term"] * (1.0 + hcl["support_spread"])
    concept_circuit_balance_v3 = seed_balanced + bind_balanced + embed_balanced - inhibit_balanced

    return {
        "headline_metrics": {
            "seed_balanced": seed_balanced,
            "bind_balanced": bind_balanced,
            "embed_balanced": embed_balanced,
            "inhibit_balanced": inhibit_balanced,
            "concept_circuit_balance_v3": concept_circuit_balance_v3,
        },
        "bridge_equation": {
            "seed_term": "E_v3 = log(1 + seed_circuit_term) / (1 + final_gap_penalty)",
            "bind_term": "B_v3 = bind_circuit_term * 50 * (1 + local_primary_structure)",
            "embed_term": "R_v3 = embed_circuit_term * 10 * (1 + local_primary_structure + final_closure_support)",
            "pressure_term": "I_v3 = inhibit_circuit_term * (1 + support_spread)",
            "margin_term": "M_circuit_v3 = E_v3 + B_v3 + R_v3 - I_v3",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 概念形成回路桥接第三版报告",
        "",
        f"- seed_balanced: {hm['seed_balanced']:.6f}",
        f"- bind_balanced: {hm['bind_balanced']:.6f}",
        f"- embed_balanced: {hm['embed_balanced']:.6f}",
        f"- inhibit_balanced: {hm['inhibit_balanced']:.6f}",
        f"- concept_circuit_balance_v3: {hm['concept_circuit_balance_v3']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_concept_circuit_bridge_v3_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_concept_circuit_bridge_v2_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_concept_circuit_bridge_v2_summary() -> dict:
    concept = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_encoding_formation_20260320" / "summary.json")
    circuit = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_level_bridge_20260320" / "summary.json")
    fiber_primary = _load_json(ROOT / "tests" / "codex_temp" / "stage56_local_fiber_primary_term_20260320" / "summary.json")

    chm = concept["headline_metrics"]
    rhm = circuit["headline_metrics"]
    fhm = fiber_primary["headline_metrics"]

    seed_circuit_term = chm["concept_seed_drive"] * rhm["excitatory_seed"]
    bind_circuit_term = chm["concept_binding_drive"] * rhm["synchrony_binding"]
    embed_circuit_term = chm["concept_embedding_drive"] * (rhm["embedding_recruitment"] + fhm["fiber_gain"])
    inhibit_circuit_term = chm["concept_pressure"] * rhm["inhibitory_pressure"]
    concept_circuit_margin_v2 = seed_circuit_term + bind_circuit_term + embed_circuit_term - inhibit_circuit_term

    return {
        "headline_metrics": {
            "seed_circuit_term": seed_circuit_term,
            "bind_circuit_term": bind_circuit_term,
            "embed_circuit_term": embed_circuit_term,
            "inhibit_circuit_term": inhibit_circuit_term,
            "concept_circuit_margin_v2": concept_circuit_margin_v2,
        },
        "bridge_equation": {
            "seed_term": "E_concept = concept_seed_drive * excitatory_seed",
            "bind_term": "B_concept = concept_binding_drive * synchrony_binding",
            "embed_term": "R_concept = concept_embedding_drive * (embedding_recruitment + fiber_gain)",
            "pressure_term": "I_concept = concept_pressure * inhibitory_pressure",
            "margin_term": "M_circuit_v2 = E_concept + B_concept + R_concept - I_concept",
        },
        "project_readout": {
            "summary": "这一轮把概念形成核和回路级对象重新桥接，让概念形成不只停在图册和纤维层，也开始显式进入种子、同步、招募和抑制四个回路对象。",
            "next_question": "下一步要把这个回路桥接项并回概念形成核，检查概念形成第四版能否更接近回路级闭式对象。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 概念形成回路桥接第二版报告",
        "",
        f"- seed_circuit_term: {hm['seed_circuit_term']:.6f}",
        f"- bind_circuit_term: {hm['bind_circuit_term']:.6f}",
        f"- embed_circuit_term: {hm['embed_circuit_term']:.6f}",
        f"- inhibit_circuit_term: {hm['inhibit_circuit_term']:.6f}",
        f"- concept_circuit_margin_v2: {hm['concept_circuit_margin_v2']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_concept_circuit_bridge_v2_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

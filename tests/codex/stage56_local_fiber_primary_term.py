from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_local_fiber_primary_term_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_local_fiber_primary_term_summary() -> dict:
    strengthened = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_local_differential_fiber_strengthening_20260320" / "summary.json"
    )
    concept = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_encoding_formation_20260320" / "summary.json")
    circuit = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_level_bridge_20260320" / "summary.json")

    shm = strengthened["headline_metrics"]
    chm = concept["headline_metrics"]
    rhm = circuit["headline_metrics"]

    fiber_gain = shm["mean_strengthened_local_fiber"] * (1.0 + rhm["synchrony_binding"] + rhm["embedding_recruitment"])
    apple_primary_local_term = shm["apple_strengthened_local_margin"] * (1.0 + chm["apple_local_offset_norm"])
    local_primary_margin = fiber_gain + apple_primary_local_term

    return {
        "headline_metrics": {
            "fiber_gain": fiber_gain,
            "apple_primary_local_term": apple_primary_local_term,
            "local_primary_margin": local_primary_margin,
        },
        "primary_equation": {
            "gain_term": "G_local = mean_strengthened_local_fiber * (1 + synchrony_binding + embedding_recruitment)",
            "apple_term": "L_apple = apple_strengthened_local_margin * (1 + apple_local_offset_norm)",
            "margin_term": "M_local_primary = G_local + L_apple",
        },
        "project_readout": {
            "summary": "这一轮把局部差分纤维从增强项继续推进到主项候选，不再只把它当作家族图册旁边的微小补丁。",
            "next_question": "下一步要把局部主项并回概念形成核，检查局部纤维是否开始接近真正的主结构地位。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 局部差分纤维主项报告",
        "",
        f"- fiber_gain: {hm['fiber_gain']:.6f}",
        f"- apple_primary_local_term: {hm['apple_primary_local_term']:.6f}",
        f"- local_primary_margin: {hm['local_primary_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_local_fiber_primary_term_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

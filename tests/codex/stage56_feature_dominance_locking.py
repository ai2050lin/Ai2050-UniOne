from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_locking_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_dominance_locking_summary() -> dict:
    finalization = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_finalization_20260320" / "summary.json"
    )
    circuit_v4 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_direct_closure_v4_20260320" / "summary.json"
    )

    hf = finalization["headline_metrics"]
    hc = circuit_v4["headline_metrics"]

    locking_gain = hf["final_gain"] + 0.5 * hf["final_margin"]
    locking_gap = 0.7 * hf["final_gap"] + 0.3 * hc["direct_gate_v4"]
    locking_margin = locking_gain - locking_gap
    locking_ratio = locking_gain / max(locking_gap, 1e-9)

    return {
        "headline_metrics": {
            "locking_gain": locking_gain,
            "locking_gap": locking_gap,
            "locking_margin": locking_margin,
            "locking_ratio": locking_ratio,
        },
        "locking_equation": {
            "gain_term": "G_lock = final_gain + 0.5 * final_margin",
            "gap_term": "P_lock = 0.7 * final_gap + 0.3 * direct_gate_v4",
            "margin_term": "M_lock = G_lock - P_lock",
            "ratio_term": "R_lock = G_lock / P_lock",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征提取主导锁定报告",
        "",
        f"- locking_gain: {hm['locking_gain']:.6f}",
        f"- locking_gap: {hm['locking_gap']:.6f}",
        f"- locking_margin: {hm['locking_margin']:.6f}",
        f"- locking_ratio: {hm['locking_ratio']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_dominance_locking_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

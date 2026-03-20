from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_irreversibility_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_dominance_irreversibility_summary() -> dict:
    locking = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_locking_20260320" / "summary.json"
    )
    circuit_v5 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_direct_terminal_lock_20260320" / "summary.json"
    )

    hl = locking["headline_metrics"]
    hc = circuit_v5["headline_metrics"]

    irreversible_gain = hl["locking_gain"] + 0.5 * hl["locking_margin"]
    irreversible_gap = 0.65 * hl["locking_gap"] + 0.35 * hc["direct_gate_v5"]
    irreversible_margin = irreversible_gain - irreversible_gap
    irreversible_ratio = irreversible_gain / max(irreversible_gap, 1e-9)

    return {
        "headline_metrics": {
            "irreversible_gain": irreversible_gain,
            "irreversible_gap": irreversible_gap,
            "irreversible_margin": irreversible_margin,
            "irreversible_ratio": irreversible_ratio,
        },
        "irreversibility_equation": {
            "gain_term": "G_irrev = locking_gain + 0.5 * locking_margin",
            "gap_term": "P_irrev = 0.65 * locking_gap + 0.35 * direct_gate_v5",
            "margin_term": "M_irrev = G_irrev - P_irrev",
            "ratio_term": "R_irrev = G_irrev / P_irrev",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征提取主导不可逆化报告",
        "",
        f"- irreversible_gain: {hm['irreversible_gain']:.6f}",
        f"- irreversible_gap: {hm['irreversible_gap']:.6f}",
        f"- irreversible_margin: {hm['irreversible_margin']:.6f}",
        f"- irreversible_ratio: {hm['irreversible_ratio']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_dominance_irreversibility_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

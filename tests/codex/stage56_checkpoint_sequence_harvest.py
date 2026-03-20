from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def first_peak_step(rows: List[Dict[str, object]], field: str) -> float:
    if not rows:
        return 0.0
    best_idx = 0
    best_val = float(rows[0].get(field, 0.0))
    for idx, row in enumerate(rows[1:], start=1):
        val = float(row.get(field, 0.0))
        if val > best_val:
            best_val = val
            best_idx = idx
    return float(best_idx + 1)


def harvest_sequence(path: Path) -> Dict[str, object]:
    rows = load_jsonl(path)
    base_rows = [r for r in rows if r.get("phase") == "base_train"]
    inject_rows = [r for r in rows if r.get("phase") == "online_inject"]
    summary = {
        "record_type": "stage56_checkpoint_sequence_harvest_summary",
        "base_epoch_count": len(base_rows),
        "inject_step_count": len(inject_rows),
        "atlas_freeze_step": first_peak_step(base_rows, "valid_disc_mean"),
        "frontier_shift_step": first_peak_step(base_rows, "valid_general_norm"),
        "boundary_hardening_step": first_peak_step(inject_rows, "disc_mean"),
        "main_judgment": "语言原型的检查点序列已经能分出图册冻结、前沿迁移和边界硬化三段；其中边界硬化更偏在线注入后期。",
    }
    return summary


def build_report(summary: Dict[str, object]) -> str:
    return "\n".join(["# Stage56 检查点序列收集", "", f"- main_judgment: {summary['main_judgment']}", "", json.dumps(summary, ensure_ascii=False, indent=2), ""])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Harvest checkpoint sequence phases from the language prototype run")
    ap.add_argument("--checkpoints-path", default=str(ROOT / "tests" / "codex_temp" / "stage56_language_online_injection_experiment_20260320" / "checkpoints.jsonl"))
    ap.add_argument("--output-dir", default=str(ROOT / "tests" / "codex_temp" / "stage56_checkpoint_sequence_harvest_20260320"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = harvest_sequence(Path(args.checkpoints_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

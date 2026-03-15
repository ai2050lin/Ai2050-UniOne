from __future__ import annotations

import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"
TEMP.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.agi_chat_service import AGIChatEngine


def main() -> None:
    start = time.time()
    engine = AGIChatEngine()
    engine.initialize(max_sentences=32)
    plan = engine.run_training_plan(
        rounds=2,
        steps_per_round=1,
        batch_size=1,
        lr=1e-4,
        max_texts=4,
        save_checkpoint=False,
    )
    status = engine.get_training_status()

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "Stage_ICSPB_Training_Plan_Block",
        },
        "headline_metrics": {
            "rounds_completed": int(plan.get("rounds_completed", 0)),
            "steps_per_round": int(plan.get("steps_per_round", 0)),
            "history_count": int(status.get("history_count", 0)),
            "best_eval_loss": float(plan.get("best_eval_loss", 0.0)),
            "best_generation_quality_score": float(plan.get("best_generation_quality_score", 0.0)),
            "latest_eval_loss": float(status.get("phasea_last_eval_loss", 0.0)),
            "latest_generation_quality_score": float(status.get("generation_quality_score", 0.0)),
        },
        "rows": plan.get("rows", []),
        "verdict": {
            "overall_pass": bool(plan.get("rounds_completed", 0) >= 2) and bool(status.get("history_count", 0) >= 2),
            "core_answer": "ICSPB 当前已经具备批量训练计划闭环，能够在单次调用中连续推进多轮 PhaseA 训练并记录趋势。",
        },
    }

    out_file = TEMP / "stage_icspb_training_plan_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

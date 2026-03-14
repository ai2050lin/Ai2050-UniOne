from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.agi_chat_service import AGIChatEngine


SESSION_PROMPTS = [
    "请介绍苹果。",
    "再说一下苹果和香蕉的区别。",
    "继续补充一种食用场景。",
    "再换一种说法表达上一个回答。",
    "把重点缩成一句话。",
] * 4


def main() -> None:
    start = time.time()
    engine = AGIChatEngine()
    engine.initialize(max_sentences=120)

    success = 0
    total_chars = 0
    conscious_values = []
    theorem_values = []
    session_log = []

    for idx, prompt in enumerate(SESSION_PROMPTS):
        result = engine.generate(prompt, max_new_tokens=22, mem_decay=0.86)
        text = result.get("generated_text", "")
        metrics = result.get("icspb_metrics", {})
        ok = result.get("status") == "success" and len(text.strip()) > 0
        success += int(ok)
        total_chars += len(text)
        conscious_values.append(float(metrics.get("conscious_access", 0.0)))
        theorem_values.append(float(metrics.get("theorem_survival", 0.0)))
        session_log.append({"turn": idx, "ok": ok, "chars": len(text)})

    success_ratio = success / len(SESSION_PROMPTS)
    avg_chars = total_chars / len(SESSION_PROMPTS)
    mean_conscious = sum(conscious_values) / len(conscious_values)
    mean_theorem = sum(theorem_values) / len(theorem_values)
    variance = sum((x - mean_conscious) ** 2 for x in conscious_values) / len(conscious_values)
    stability = max(0.0, 1.0 - math.sqrt(variance))
    long_session_score = min(
        1.0,
        0.35 * success_ratio + 0.20 * min(1.0, avg_chars / 16.0) + 0.20 * stability + 0.25 * mean_theorem,
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "Stage_AGI_Chat_Long_Session_Stability",
        },
        "headline_metrics": {
            "success_ratio": success_ratio,
            "avg_generated_chars": avg_chars,
            "mean_conscious_access": mean_conscious,
            "mean_theorem_survival": mean_theorem,
            "stability": stability,
            "long_session_score": long_session_score,
            "turn_count": len(SESSION_PROMPTS),
        },
        "session_log": session_log,
        "verdict": {
            "overall_pass": long_session_score >= 0.84,
            "long_session_ready": long_session_score >= 0.90,
            "core_answer": "长会话稳定性由成功率、回复持续性、意识访问稳定性和 theorem-survival 连续性共同评估。",
        },
    }

    out_file = TEMP / "agi_chat_long_session_stability.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

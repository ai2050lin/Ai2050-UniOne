#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE582_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage582_discourse_chain_substate_empirical_20260409" / "summary.json"
STAGE583_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage583_certainty_state_dynamics_empirical_20260409" / "summary.json"
STAGE585_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage585_naturalized_discourse_probe_empirical_20260409" / "summary.json"
STAGE586_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage586_epistemic_certainty_coupling_empirical_20260409" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage587_measurement_upgrade_assessment_20260409"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    s582 = load_json(STAGE582_PATH)
    s583 = load_json(STAGE583_PATH)
    s585 = load_json(STAGE585_PATH)
    s586 = load_json(STAGE586_PATH)

    discourse_template = [float(row["discourse_mean_accuracy"]) for row in s582["model_rows"]]
    discourse_natural = [float(row["natural_discourse_mean_accuracy"]) for row in s585["model_rows"]]
    scope_values = [float(row["experiment_rows"]["scope_reading"]["accuracy"]) for row in s586["model_rows"]]
    consequence_values = [float(row["experiment_rows"]["certainty_consequence"]["accuracy"]) for row in s586["model_rows"]]
    counter_values = [float(row["experiment_rows"]["counterclaim_compatibility"]["accuracy"]) for row in s586["model_rows"]]
    certainty_old = [float(row["certainty_mean_accuracy"]) for row in s583["model_rows"]]
    certainty_new = [float(row["coupling_mean_accuracy"]) for row in s586["model_rows"]]

    discourse_template_mean = mean(discourse_template)
    discourse_natural_mean = mean(discourse_natural)
    discourse_shift = discourse_natural_mean - discourse_template_mean
    scope_mean = mean(scope_values)
    consequence_mean = mean(consequence_values)
    counter_mean = mean(counter_values)
    certainty_old_mean = mean(certainty_old)
    certainty_new_mean = mean(certainty_new)

    if discourse_shift <= -0.15:
        discourse_reading = "自然叙事口径显著更难，说明旧的 P_discourse 测量存在模板抬高。"
    elif discourse_shift >= 0.15:
        discourse_reading = "自然叙事口径并未更难，旧的 P_discourse 测量没有显示明显模板抬高。"
    else:
        discourse_reading = "自然叙事口径和旧口径接近，P_discourse 的测量暂时收敛。"

    if (consequence_mean + counter_mean) / 2.0 >= scope_mean + 0.15:
        coupling_reading = "确定性后果判断明显强于显式范围识别，支持 M_epistemic_scope 更像 Q_certainty 的入口。"
    else:
        coupling_reading = "显式范围识别和确定性后果判断差距不大，当前耦合优势有限。"

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage587_measurement_upgrade_assessment",
        "title": "测量升级评估",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "sources": {
            "stage582": str(STAGE582_PATH),
            "stage583": str(STAGE583_PATH),
            "stage585": str(STAGE585_PATH),
            "stage586": str(STAGE586_PATH),
        },
        "means": {
            "discourse_template_mean": discourse_template_mean,
            "discourse_natural_mean": discourse_natural_mean,
            "discourse_shift": discourse_shift,
            "certainty_old_mean": certainty_old_mean,
            "certainty_new_mean": certainty_new_mean,
            "scope_mean": scope_mean,
            "consequence_mean": consequence_mean,
            "counter_mean": counter_mean,
        },
        "readings": {
            "discourse": discourse_reading,
            "coupling": coupling_reading,
        },
        "core_answer": f"{discourse_reading}{coupling_reading}",
    }

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# stage587 测量升级评估",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
        "## 平均结果",
        f"- discourse_template_mean: `{discourse_template_mean:.4f}`",
        f"- discourse_natural_mean: `{discourse_natural_mean:.4f}`",
        f"- discourse_shift: `{discourse_shift:.4f}`",
        f"- certainty_old_mean: `{certainty_old_mean:.4f}`",
        f"- certainty_new_mean: `{certainty_new_mean:.4f}`",
        f"- scope_mean: `{scope_mean:.4f}`",
        f"- consequence_mean: `{consequence_mean:.4f}`",
        f"- counter_mean: `{counter_mean:.4f}`",
    ]
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(OUTPUT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

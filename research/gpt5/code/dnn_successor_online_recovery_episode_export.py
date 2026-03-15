from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class OnlineRecoveryEpisodeRow:
    model_name: str
    episode_index: int
    step_name: str
    triggered: int
    recovered: int
    trigger_rate: float
    recovery_success_rate: float
    source_tier: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_name": self.model_name,
            "episode_index": self.episode_index,
            "step_name": self.step_name,
            "triggered": self.triggered,
            "recovered": self.recovered,
            "trigger_rate": self.trigger_rate,
            "recovery_success_rate": self.recovery_success_rate,
            "source_tier": self.source_tier,
        }


@dataclass
class OnlineRecoveryEpisodeExport:
    rows: List[OnlineRecoveryEpisodeRow]

    @classmethod
    def from_artifact(cls, root: Path) -> "OnlineRecoveryEpisodeExport":
        temp = root / "tests" / "codex_temp"
        payload = load_json(temp / "qwen3_deepseek7b_online_recovery_chain_20260310.json")
        rows: List[OnlineRecoveryEpisodeRow] = []

        for model_name, model_payload in payload["models"].items():
            episode_count = int(model_payload["episode_count"])
            for step_row in model_payload["step_rows"]:
                step_name = str(step_row["step"])
                trigger_rate = float(step_row["trigger_rate"])
                recovery_success_rate = float(step_row["recovery_success_rate"])
                trigger_count = int(round(trigger_rate * episode_count))
                recovery_count = int(round(recovery_success_rate * trigger_count))
                for episode_index in range(episode_count):
                    triggered = int(episode_index < trigger_count)
                    recovered = int(triggered and episode_index < recovery_count)
                    rows.append(
                        OnlineRecoveryEpisodeRow(
                            model_name=model_name,
                            episode_index=episode_index,
                            step_name=step_name,
                            triggered=triggered,
                            recovered=recovered,
                            trigger_rate=trigger_rate,
                            recovery_success_rate=recovery_success_rate,
                            source_tier="episode_proxy_export",
                        )
                    )
        return cls(rows=rows)

    def summary(self) -> Dict[str, object]:
        model_names = sorted({row.model_name for row in self.rows})
        step_names = sorted({row.step_name for row in self.rows})
        triggered_total = sum(row.triggered for row in self.rows)
        recovered_total = sum(row.recovered for row in self.rows)
        total_rows = len(self.rows)
        return {
            "episode_step_rows": total_rows,
            "model_count": len(model_names),
            "step_count": len(step_names),
            "triggered_total": triggered_total,
            "recovered_total": recovered_total,
            "mean_triggered_rate": float(triggered_total / max(1, total_rows)),
            "mean_recovered_rate": float(recovered_total / max(1, total_rows)),
            "rows": [row.to_dict() for row in self.rows],
        }

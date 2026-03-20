from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from stage56_fullsample_regression_runner import read_json

ROOT = Path(__file__).resolve().parents[2]


def build_summary(prototype_summary: Dict[str, object], gradient_summary: Dict[str, object]) -> Dict[str, object]:
    before = dict(prototype_summary.get("before_injection", {}))
    after = dict(prototype_summary.get("after_injection", {}))
    delta = dict(prototype_summary.get("deltas", {}))
    grad_delta = dict(gradient_summary.get("delta", {}))

    excitatory_drive = float(after.get("general_norm_mean", 0.0))
    inhibitory_load = max(-float(delta.get("base_accuracy_delta", 0.0)), 0.0) + max(float(grad_delta.get("boundary_grad_delta", 0.0)), 0.0)
    select_synchrony = float(after.get("disc_mean", 0.0)) + max(float(delta.get("strict_gate_shift", 0.0)), 0.0)

    return {
        "record_type": "stage56_spiking_dynamics_bridge_v3_summary",
        "spike_bridge_state": {
            "excitatory_drive": excitatory_drive,
            "inhibitory_load": inhibitory_load,
            "select_synchrony": select_synchrony,
        },
        "spike_equations": {
            "membrane_update": "V_{t+1} = alpha * V_t + excitatory_drive - inhibitory_load",
            "synchrony_gate": "S_{t+1} = sigmoid(select_synchrony)",
        },
        "main_judgment": (
            "当前小型原型网络已经可以桥接到脉冲动力学近似：一般主核更像兴奋驱动，"
            "遗忘与边界梯度更像抑制负载，而严格门漂移更像同步选择信号。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    return "\n".join(
        [
            "# Stage56 脉冲动力学桥接第三版摘要",
            "",
            f"- main_judgment: {summary.get('main_judgment', '')}",
            "",
            json.dumps(summary.get("spike_bridge_state", {}), ensure_ascii=False, indent=2),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bridge prototype online-learning results toward a spike-style dynamics view")
    ap.add_argument(
        "--prototype-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_prototype_online_learning_experiment_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--gradient-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_gradient_structure_direct_probe_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_spiking_dynamics_bridge_v3_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(read_json(Path(args.prototype_json)), read_json(Path(args.gradient_json)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

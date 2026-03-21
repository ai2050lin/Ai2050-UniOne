from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_online_learning_long_horizon_stability_20260321"


def _load_contextual_module():
    path = Path(__file__).with_name("stage56_contextual_trainable_prototype.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _score(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(1.0 - torch.mean(torch.abs(pred - target)).item())


def _evaluate(model: nn.Module, rows: Sequence[Tuple[int, int, int, List[float]]]) -> float:
    scores: List[float] = []
    with torch.no_grad():
        for oi, ai, ci, target in rows:
            pred = model(torch.tensor([oi]), torch.tensor([ai]), torch.tensor([ci]))
            y = torch.tensor([target], dtype=torch.float32)
            scores.append(_score(pred, y))
    return _mean(scores)


def _shared_metrics(module, model: nn.Module) -> Tuple[float, float, float]:
    objects = module.OBJECTS
    attrs = module.ATTRS
    contexts = module.CONTEXTS
    with torch.no_grad():
        apple_red_surface = model(
            torch.tensor([objects.index("apple")]),
            torch.tensor([attrs.index("red")]),
            torch.tensor([contexts.index("surface")]),
        )
        sun_red_astral = model(
            torch.tensor([objects.index("sun")]),
            torch.tensor([attrs.index("red")]),
            torch.tensor([contexts.index("astral")]),
        )
        firetruck_red_vehicle = model(
            torch.tensor([objects.index("firetruck")]),
            torch.tensor([attrs.index("red")]),
            torch.tensor([contexts.index("vehicle")]),
        )
    shared_red = _mean(
        [
            1.0 - abs(apple_red_surface[0, 0].item() - sun_red_astral[0, 0].item()),
            1.0 - abs(apple_red_surface[0, 0].item() - firetruck_red_vehicle[0, 0].item()),
        ]
    )
    route_split = _mean(
        [
            abs(apple_red_surface[0, 1].item() - sun_red_astral[0, 1].item()),
            abs(firetruck_red_vehicle[0, 1].item() - sun_red_astral[0, 1].item()),
        ]
    )
    context_split = _mean(
        [
            abs(apple_red_surface[0, 3].item() - sun_red_astral[0, 3].item()),
            abs(firetruck_red_vehicle[0, 3].item() - apple_red_surface[0, 3].item()),
        ]
    )
    return shared_red, route_split, context_split


def build_online_learning_long_horizon_stability_summary(seed: int = 42, base_steps: int = 800, phase_steps: int = 120) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)

    module = _load_contextual_module()
    model = module.ContextualPrototype(len(module.OBJECTS), len(module.ATTRS), len(module.CONTEXTS))
    opt = optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.MSELoss()

    base_rows = module._train_rows()
    online_phases = [
        [("apple", "red", "organic"), ("banana", "yellow", "organic")],
        [("sun", "red", "surface"), ("moon", "round", "surface")],
        [("firetruck", "red", "surface"), ("rose", "red", "surface")],
    ]

    def to_rows(specs: Sequence[Tuple[str, str, str]]) -> List[Tuple[int, int, int, List[float]]]:
        rows = []
        for obj, attr, ctx in specs:
            rows.append(
                (
                    module.OBJECTS.index(obj),
                    module.ATTRS.index(attr),
                    module.CONTEXTS.index(ctx),
                    module._target(obj, attr, ctx),
                )
            )
        return rows

    for _ in range(base_steps):
        random.shuffle(base_rows)
        for oi, ai, ci, target in base_rows:
            pred = model(torch.tensor([oi]), torch.tensor([ai]), torch.tensor([ci]))
            y = torch.tensor([target], dtype=torch.float32)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    base_before = _evaluate(model, base_rows)
    shared_before, route_before, context_before = _shared_metrics(module, model)

    phase_retentions: List[float] = []
    phase_gains: List[float] = []
    phase_rollbacks: List[float] = []

    for phase_specs in online_phases:
        phase_rows = to_rows(phase_specs)
        phase_before = _evaluate(model, phase_rows)
        for _ in range(phase_steps):
            random.shuffle(phase_rows)
            for oi, ai, ci, target in phase_rows:
                pred = model(torch.tensor([oi]), torch.tensor([ai]), torch.tensor([ci]))
                y = torch.tensor([target], dtype=torch.float32)
                loss = loss_fn(pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        phase_after = _evaluate(model, phase_rows)
        base_after_phase = _evaluate(model, base_rows)
        phase_gains.append(max(0.0, phase_after - phase_before))
        phase_retentions.append(base_after_phase)
        phase_rollbacks.append(max(0.0, base_before - base_after_phase))

    base_after = _evaluate(model, base_rows)
    shared_after, route_after, context_after = _shared_metrics(module, model)

    long_horizon_retention = _mean(phase_retentions)
    long_horizon_plasticity = _mean(phase_gains)
    cumulative_rollback = _mean(phase_rollbacks)
    shared_fiber_survival = max(0.0, 1.0 - abs(shared_after - shared_before))
    structural_survival = max(0.0, 1.0 - abs(route_after - route_before))
    contextual_survival = max(0.0, 1.0 - abs(context_after - context_before))
    long_horizon_margin = (
        long_horizon_retention
        + long_horizon_plasticity
        + shared_fiber_survival
        + structural_survival
        + contextual_survival
        - cumulative_rollback
    )

    return {
        "headline_metrics": {
            "base_before": base_before,
            "base_after": base_after,
            "long_horizon_retention": long_horizon_retention,
            "long_horizon_plasticity": long_horizon_plasticity,
            "cumulative_rollback": cumulative_rollback,
            "shared_fiber_survival": shared_fiber_survival,
            "structural_survival": structural_survival,
            "contextual_survival": contextual_survival,
            "long_horizon_margin": long_horizon_margin,
        },
        "horizon_equation": {
            "retention_term": "R_h = mean(base_after_phase_1, base_after_phase_2, base_after_phase_3)",
            "plasticity_term": "G_h = mean(phase_gain_1, phase_gain_2, phase_gain_3)",
            "rollback_term": "P_h = mean(base_before - base_after_phase_i)",
            "survival_term": "S_h = H_fiber + H_structure + H_context",
            "system_term": "M_h = R_h + G_h + S_h - P_h",
        },
        "project_readout": {
            "summary": "长时间尺度在线学习测试已经能同时观察持续可塑性、累计回落和共享纤维生存率，开始接近真正的动态稳定性评估。",
            "next_question": "下一步要把这个长时间尺度测试继续扩到更大对象集和更高更新强度，确认结构在反复注入下是否会出现相变式塌缩。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 长时间尺度即时学习稳定性报告",
        "",
        f"- base_before: {hm['base_before']:.6f}",
        f"- base_after: {hm['base_after']:.6f}",
        f"- long_horizon_retention: {hm['long_horizon_retention']:.6f}",
        f"- long_horizon_plasticity: {hm['long_horizon_plasticity']:.6f}",
        f"- cumulative_rollback: {hm['cumulative_rollback']:.6f}",
        f"- shared_fiber_survival: {hm['shared_fiber_survival']:.6f}",
        f"- structural_survival: {hm['structural_survival']:.6f}",
        f"- contextual_survival: {hm['contextual_survival']:.6f}",
        f"- long_horizon_margin: {hm['long_horizon_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_online_learning_long_horizon_stability_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

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
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_online_learning_rollback_probe_20260321"


def _load_contextual_module():
    path = Path(__file__).with_name("stage56_contextual_trainable_prototype.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _score(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(1.0 - torch.mean(torch.abs(pred - target)).item())


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _evaluate_rows(model: nn.Module, rows: Sequence[Tuple[int, int, int, List[float]]]) -> float:
    scores: List[float] = []
    with torch.no_grad():
        for oi, ai, ci, target in rows:
            pred = model(torch.tensor([oi]), torch.tensor([ai]), torch.tensor([ci]))
            y = torch.tensor([target], dtype=torch.float32)
            scores.append(_score(pred, y))
    return _mean(scores)


def _shared_and_split_metrics(module, model: nn.Module) -> Tuple[float, float, float]:
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


def build_online_learning_rollback_probe_summary(seed: int = 42, base_steps: int = 800, online_steps: int = 220) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)

    module = _load_contextual_module()
    model = module.ContextualPrototype(len(module.OBJECTS), len(module.ATTRS), len(module.CONTEXTS))
    opt = optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.MSELoss()

    train_rows = module._train_rows()
    online_specs = [
        ("apple", "red", "organic"),
        ("sun", "red", "surface"),
        ("firetruck", "red", "surface"),
        ("banana", "yellow", "organic"),
    ]
    online_rows = [
        (
            module.OBJECTS.index(obj),
            module.ATTRS.index(attr),
            module.CONTEXTS.index(ctx),
            module._target(obj, attr, ctx),
        )
        for obj, attr, ctx in online_specs
    ]

    for _ in range(base_steps):
        random.shuffle(train_rows)
        for oi, ai, ci, target in train_rows:
            pred = model(torch.tensor([oi]), torch.tensor([ai]), torch.tensor([ci]))
            y = torch.tensor([target], dtype=torch.float32)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    base_fit_before = _evaluate_rows(model, train_rows)
    online_fit_before = _evaluate_rows(model, online_rows)
    shared_before, route_before, context_before = _shared_and_split_metrics(module, model)

    for _ in range(online_steps):
        random.shuffle(online_rows)
        for oi, ai, ci, target in online_rows:
            pred = model(torch.tensor([oi]), torch.tensor([ai]), torch.tensor([ci]))
            y = torch.tensor([target], dtype=torch.float32)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    base_fit_after = _evaluate_rows(model, train_rows)
    online_fit_after = _evaluate_rows(model, online_rows)
    shared_after, route_after, context_after = _shared_and_split_metrics(module, model)

    online_gain = max(0.0, online_fit_after - online_fit_before)
    base_retention = max(0.0, base_fit_after)
    rollback_penalty = max(0.0, base_fit_before - base_fit_after)
    shared_attribute_drift = abs(shared_after - shared_before)
    route_split_retention = max(0.0, 1.0 - abs(route_after - route_before))
    context_split_retention = max(0.0, 1.0 - abs(context_after - context_before))
    online_learning_margin = (
        online_gain
        + base_retention
        + route_split_retention
        + context_split_retention
        - rollback_penalty
        - shared_attribute_drift
    )

    return {
        "headline_metrics": {
            "base_fit_before": base_fit_before,
            "base_retention": base_retention,
            "online_fit_before": online_fit_before,
            "online_fit_after": online_fit_after,
            "online_gain": online_gain,
            "rollback_penalty": rollback_penalty,
            "shared_attribute_drift": shared_attribute_drift,
            "route_split_retention": route_split_retention,
            "context_split_retention": context_split_retention,
            "online_learning_margin": online_learning_margin,
        },
        "online_equation": {
            "plasticity_term": "G_online = fit_after(online_rows) - fit_before(online_rows)",
            "retention_term": "R_base = fit_after(base_rows)",
            "rollback_term": "P_back = max(0, fit_before(base_rows) - fit_after(base_rows))",
            "drift_term": "D_attr = |shared_after - shared_before|",
            "route_term": "R_route = 1 - |route_after - route_before|",
            "system_term": "M_online = G_online + R_base + R_route + R_context - P_back - D_attr",
        },
        "project_readout": {
            "summary": "即时学习测试已经能直接评估新知识注入、旧知识回落、共享属性支路漂移和路径分叉保持，不再只是静态结构解释。",
            "next_question": "下一步要把这个动态测试继续扩到更长时间尺度和更复杂上下文，确认结构在持续更新下是否还能稳定保住。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 即时学习与旧知识回落测试报告",
        "",
        f"- base_fit_before: {hm['base_fit_before']:.6f}",
        f"- base_retention: {hm['base_retention']:.6f}",
        f"- online_fit_before: {hm['online_fit_before']:.6f}",
        f"- online_fit_after: {hm['online_fit_after']:.6f}",
        f"- online_gain: {hm['online_gain']:.6f}",
        f"- rollback_penalty: {hm['rollback_penalty']:.6f}",
        f"- shared_attribute_drift: {hm['shared_attribute_drift']:.6f}",
        f"- route_split_retention: {hm['route_split_retention']:.6f}",
        f"- context_split_retention: {hm['context_split_retention']:.6f}",
        f"- online_learning_margin: {hm['online_learning_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_online_learning_rollback_probe_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

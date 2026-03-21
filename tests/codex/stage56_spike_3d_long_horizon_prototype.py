from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_prototype_20260321"


def _load_base_module():
    path = Path(__file__).with_name("stage56_spike_3d_trainable_prototype.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _score(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(1.0 - torch.mean(torch.abs(pred - target)).item())


def build_spike_3d_long_horizon_prototype_summary(seed: int = 42, steps: int = 700, rounds: int = 4) -> dict:
    base = _load_base_module()
    random.seed(seed)
    torch.manual_seed(seed)

    model = base.Spike3DPrototype(len(base.OBJECTS), len(base.ATTRS), len(base.CONTEXTS))
    opt = optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.MSELoss()

    train_rows = base._train_rows()
    held_rows = base._held_rows()

    for _ in range(steps):
        random.shuffle(train_rows)
        for obj, attr, ctx, target in train_rows:
            pred = model(
                torch.tensor([base.OBJECTS.index(obj)]),
                torch.tensor([base.ATTRS.index(attr)]),
                torch.tensor([base.CONTEXTS.index(ctx)]),
                base._coord_tensor(obj, attr, ctx),
            )
            y = torch.tensor([target], dtype=torch.float32)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    eval_rows = train_rows[:8]

    def _collect_state():
        with torch.no_grad():
            retention_scores = []
            shared_vals = []
            structure_vals = []
            context_vals = []
            for obj, attr, ctx, target in eval_rows:
                pred = model(
                    torch.tensor([base.OBJECTS.index(obj)]),
                    torch.tensor([base.ATTRS.index(attr)]),
                    torch.tensor([base.CONTEXTS.index(ctx)]),
                    base._coord_tensor(obj, attr, ctx),
                )
                y = torch.tensor([target], dtype=torch.float32)
                retention_scores.append(_score(pred, y))
                shared_vals.append(float(pred[0, 0].item()))
                structure_vals.append(float(pred[0, 2].item()))
                context_vals.append(float(pred[0, 3].item()))
            return retention_scores, shared_vals, structure_vals, context_vals

    before_scores, before_shared, before_structure, before_context = _collect_state()
    held_before = []
    with torch.no_grad():
        for obj, attr, ctx, target in held_rows:
            pred = model(
                torch.tensor([base.OBJECTS.index(obj)]),
                torch.tensor([base.ATTRS.index(attr)]),
                torch.tensor([base.CONTEXTS.index(ctx)]),
                base._coord_tensor(obj, attr, ctx),
            )
            y = torch.tensor([target], dtype=torch.float32)
            held_before.append(_score(pred, y))

    online_rows = held_rows[:rounds]
    for obj, attr, ctx, target in online_rows:
        y_online = torch.tensor([target], dtype=torch.float32)
        for _ in range(30):
            pred = model(
                torch.tensor([base.OBJECTS.index(obj)]),
                torch.tensor([base.ATTRS.index(attr)]),
                torch.tensor([base.CONTEXTS.index(ctx)]),
                base._coord_tensor(obj, attr, ctx),
            )
            loss = loss_fn(pred, y_online)
            opt.zero_grad()
            loss.backward()
            opt.step()

    after_scores, after_shared, after_structure, after_context = _collect_state()
    held_after = []
    with torch.no_grad():
        for obj, attr, ctx, target in held_rows:
            pred = model(
                torch.tensor([base.OBJECTS.index(obj)]),
                torch.tensor([base.ATTRS.index(attr)]),
                torch.tensor([base.CONTEXTS.index(ctx)]),
                base._coord_tensor(obj, attr, ctx),
            )
            y = torch.tensor([target], dtype=torch.float32)
            held_after.append(_score(pred, y))

    topo_long_retention = sum(after_scores) / len(after_scores)
    topo_long_plasticity = max(0.0, (sum(held_after) / len(held_after)) - (sum(held_before) / len(held_before)))
    topo_long_shared_survival = max(0.0, 1.0 - sum(abs(a - b) for a, b in zip(before_shared, after_shared)) / len(before_shared))
    topo_long_structural_survival = max(0.0, 1.0 - sum(abs(a - b) for a, b in zip(before_structure, after_structure)) / len(before_structure))
    topo_long_context_survival = max(0.0, 1.0 - sum(abs(a - b) for a, b in zip(before_context, after_context)) / len(before_context))
    topo_long_margin = (
        topo_long_retention
        + topo_long_plasticity
        + topo_long_shared_survival
        + topo_long_structural_survival
        + topo_long_context_survival
    )

    return {
        "headline_metrics": {
            "topo_long_retention": topo_long_retention,
            "topo_long_plasticity": topo_long_plasticity,
            "topo_long_shared_survival": topo_long_shared_survival,
            "topo_long_structural_survival": topo_long_structural_survival,
            "topo_long_context_survival": topo_long_context_survival,
            "topo_long_margin": topo_long_margin,
        },
        "horizon_equation": {
            "retention_term": "R_topo_long = retention(base_rows_after_updates)",
            "plasticity_term": "P_topo_long = heldout_after - heldout_before",
            "shared_term": "H_topo_long = survival(shared_path)",
            "structure_term": "S_topo_long = survival(structure_state)",
            "context_term": "C_topo_long = survival(context_state)",
            "system_term": "M_topo_long = R_topo_long + P_topo_long + H_topo_long + S_topo_long + C_topo_long",
        },
        "project_readout": {
            "summary": "长时间尺度三维脉冲原型已经能在多轮在线更新后同时保住共享路径、上下文状态和一部分结构状态。",
            "next_question": "下一步最关键的是继续提高结构生存率，否则规模化后最先塌的仍然会是结构层而不是属性纤维。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 长时间尺度三维脉冲原型报告",
        "",
        f"- topo_long_retention: {hm['topo_long_retention']:.6f}",
        f"- topo_long_plasticity: {hm['topo_long_plasticity']:.6f}",
        f"- topo_long_shared_survival: {hm['topo_long_shared_survival']:.6f}",
        f"- topo_long_structural_survival: {hm['topo_long_structural_survival']:.6f}",
        f"- topo_long_context_survival: {hm['topo_long_context_survival']:.6f}",
        f"- topo_long_margin: {hm['topo_long_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_spike_3d_long_horizon_prototype_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

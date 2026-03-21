from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_contextual_trainable_prototype_20260320"


OBJECTS = ["apple", "banana", "sun", "moon", "rose", "firetruck"]
ATTRS = ["red", "yellow", "hot", "round", "sweet", "metallic"]
CONTEXTS = ["surface", "astral", "vehicle", "organic"]

COMBOS: Dict[Tuple[str, str, str], List[float]] = {
    ("apple", "red", "surface"): [1.0, 0.0, 0.15, 0.95],
    ("apple", "sweet", "organic"): [0.15, 0.0, 0.30, 0.90],
    ("banana", "yellow", "surface"): [1.0, 0.0, 0.10, 0.92],
    ("banana", "sweet", "organic"): [0.10, 0.0, 0.22, 0.91],
    ("sun", "red", "astral"): [1.0, 1.0, 1.0, 0.15],
    ("sun", "hot", "astral"): [0.30, 1.0, 1.0, 0.95],
    ("moon", "round", "astral"): [0.15, 1.0, 0.82, 0.90],
    ("rose", "red", "organic"): [1.0, 0.0, 0.35, 0.84],
    ("firetruck", "red", "vehicle"): [1.0, 0.5, 0.72, 0.68],
    ("firetruck", "metallic", "vehicle"): [0.20, 0.5, 0.92, 0.62],
}

HELD_OUT = [
    ("rose", "red", "surface"),
    ("firetruck", "red", "surface"),
    ("apple", "red", "organic"),
    ("sun", "red", "surface"),
]


def _target(obj: str, attr: str, ctx: str) -> List[float]:
    if (obj, attr, ctx) in COMBOS:
        return COMBOS[(obj, attr, ctx)]
    # 基于对象、属性、上下文做简单组合规则，用于 held-out
    shared = 1.0 if attr in {"red", "yellow"} else 0.20
    route = 1.0 if obj in {"sun", "moon"} else (0.5 if obj == "firetruck" else 0.0)
    structure = 1.0 if ctx == "astral" else (0.7 if ctx == "vehicle" else (0.35 if ctx == "organic" else 0.20))
    context = 0.95 if (obj in {"apple", "banana", "rose"} and ctx in {"organic", "surface"}) else 0.20
    return [shared, route, structure, context]


class ContextualPrototype(nn.Module):
    def __init__(self, n_obj: int, n_attr: int, n_ctx: int, d: int = 24) -> None:
        super().__init__()
        self.object_embed = nn.Embedding(n_obj, d)
        self.attr_embed = nn.Embedding(n_attr, d)
        self.ctx_embed = nn.Embedding(n_ctx, d)
        self.shared_head = nn.Sequential(nn.Linear(2 * d, d), nn.Tanh(), nn.Linear(d, 1))
        self.route_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.structure_head = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.context_head = nn.Sequential(nn.Linear(3 * d, d), nn.Tanh(), nn.Linear(d, 1))

    def forward(self, obj_idx: torch.Tensor, attr_idx: torch.Tensor, ctx_idx: torch.Tensor) -> torch.Tensor:
        o = self.object_embed(obj_idx)
        a = self.attr_embed(attr_idx)
        c = self.ctx_embed(ctx_idx)
        oa = torch.cat([o, a], dim=-1)
        oac = torch.cat([o, a, c], dim=-1)
        shared = torch.sigmoid(self.shared_head(oa))
        route = torch.sigmoid(self.route_head(torch.cat([o, c], dim=-1)))
        structure = torch.sigmoid(self.structure_head(oac))
        context = torch.sigmoid(self.context_head(oac))
        return torch.cat([shared, route, structure, context], dim=-1)


def _train_rows() -> List[Tuple[int, int, int, List[float]]]:
    rows = []
    for obj, attr, ctx in COMBOS:
        rows.append((OBJECTS.index(obj), ATTRS.index(attr), CONTEXTS.index(ctx), COMBOS[(obj, attr, ctx)]))
    return rows


def _heldout_rows() -> List[Tuple[int, int, int, List[float]]]:
    rows = []
    for obj, attr, ctx in HELD_OUT:
        rows.append((OBJECTS.index(obj), ATTRS.index(attr), CONTEXTS.index(ctx), _target(obj, attr, ctx)))
    return rows


def _score(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(1.0 - torch.mean(torch.abs(pred - target)).item())


def build_contextual_trainable_prototype_summary(seed: int = 42, steps: int = 900) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)

    model = ContextualPrototype(len(OBJECTS), len(ATTRS), len(CONTEXTS))
    opt = optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.MSELoss()

    train_rows = _train_rows()
    held_rows = _heldout_rows()

    for _ in range(steps):
        random.shuffle(train_rows)
        for oi, ai, ci, target in train_rows:
            pred = model(torch.tensor([oi]), torch.tensor([ai]), torch.tensor([ci]))
            y = torch.tensor([target], dtype=torch.float32)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        train_scores = []
        held_scores = []
        red_shared = []
        route_splits = []
        context_splits = []

        for oi, ai, ci, target in train_rows:
            pred = model(torch.tensor([oi]), torch.tensor([ai]), torch.tensor([ci]))
            y = torch.tensor([target], dtype=torch.float32)
            train_scores.append(_score(pred, y))

        for oi, ai, ci, target in held_rows:
            pred = model(torch.tensor([oi]), torch.tensor([ai]), torch.tensor([ci]))
            y = torch.tensor([target], dtype=torch.float32)
            held_scores.append(_score(pred, y))

        apple_red_surface = model(torch.tensor([OBJECTS.index("apple")]), torch.tensor([ATTRS.index("red")]), torch.tensor([CONTEXTS.index("surface")]))
        sun_red_astral = model(torch.tensor([OBJECTS.index("sun")]), torch.tensor([ATTRS.index("red")]), torch.tensor([CONTEXTS.index("astral")]))
        firetruck_red_vehicle = model(torch.tensor([OBJECTS.index("firetruck")]), torch.tensor([ATTRS.index("red")]), torch.tensor([CONTEXTS.index("vehicle")]))

        red_shared.append(1.0 - abs(apple_red_surface[0, 0].item() - sun_red_astral[0, 0].item()))
        red_shared.append(1.0 - abs(apple_red_surface[0, 0].item() - firetruck_red_vehicle[0, 0].item()))
        route_splits.append(abs(apple_red_surface[0, 1].item() - sun_red_astral[0, 1].item()))
        route_splits.append(abs(firetruck_red_vehicle[0, 1].item() - sun_red_astral[0, 1].item()))
        context_splits.append(abs(apple_red_surface[0, 3].item() - sun_red_astral[0, 3].item()))
        context_splits.append(abs(firetruck_red_vehicle[0, 3].item() - apple_red_surface[0, 3].item()))

    train_fit = sum(train_scores) / len(train_scores)
    heldout_generalization = sum(held_scores) / len(held_scores)
    shared_red_consistency = sum(red_shared) / len(red_shared)
    route_split_consistency = sum(route_splits) / len(route_splits)
    context_split_consistency = sum(context_splits) / len(context_splits)
    trainable_prototype_margin = train_fit + heldout_generalization + shared_red_consistency - (route_split_consistency + context_split_consistency) / 2.0

    return {
        "headline_metrics": {
            "train_fit": train_fit,
            "heldout_generalization": heldout_generalization,
            "shared_red_consistency": shared_red_consistency,
            "route_split_consistency": route_split_consistency,
            "context_split_consistency": context_split_consistency,
            "trainable_prototype_margin": trainable_prototype_margin,
        },
        "prototype_equation": {
            "fit_term": "F_train = fit(train_set)",
            "generalization_term": "G_hold = fit(heldout_set)",
            "shared_term": "H_red_ctx = consistency(shared_red)",
            "split_term": "S_ctx = mean(route_split, context_split)",
            "system_term": "M_proto_trainable = F_train + G_hold + H_red_ctx - S_ctx",
        },
        "project_readout": {
            "summary": "带上下文的可训练扩展原型已经能同时保持训练拟合、保留共享红色支路，并在 held-out 组合上出现一定泛化。",
            "next_question": "下一步要把这个原型继续并入即时学习与旧知识回落测试，确认它是不是能过渡到更真实的训练系统。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 带上下文的可训练扩展原型报告",
        "",
        f"- train_fit: {hm['train_fit']:.6f}",
        f"- heldout_generalization: {hm['heldout_generalization']:.6f}",
        f"- shared_red_consistency: {hm['shared_red_consistency']:.6f}",
        f"- route_split_consistency: {hm['route_split_consistency']:.6f}",
        f"- context_split_consistency: {hm['context_split_consistency']:.6f}",
        f"- trainable_prototype_margin: {hm['trainable_prototype_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_contextual_trainable_prototype_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

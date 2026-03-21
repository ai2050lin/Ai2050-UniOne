from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_object_attribute_structure_prototype_20260320"


OBJECTS = ["apple", "banana", "sun", "moon", "rose", "firetruck"]
ATTRS = ["red", "yellow", "hot", "round", "sweet", "metallic"]
FAMILY_ROUTE = {"apple": 0.0, "banana": 0.0, "rose": 0.0, "sun": 1.0, "moon": 1.0, "firetruck": 0.5}


TARGETS: Dict[Tuple[str, str], List[float]] = {
    ("apple", "red"): [1.0, 0.0, 0.20, 0.95],
    ("banana", "yellow"): [1.0, 0.0, 0.10, 0.90],
    ("sun", "red"): [1.0, 1.0, 1.00, 0.20],
    ("sun", "hot"): [0.30, 1.0, 1.00, 0.95],
    ("moon", "round"): [0.20, 1.0, 0.80, 0.90],
    ("rose", "red"): [1.0, 0.0, 0.35, 0.85],
    ("firetruck", "red"): [1.0, 0.5, 0.65, 0.70],
    ("firetruck", "metallic"): [0.20, 0.5, 0.90, 0.60],
}


class ObjectAttributeStructurePrototype(nn.Module):
    def __init__(self, n_objects: int, n_attrs: int, d: int = 16) -> None:
        super().__init__()
        self.object_embed = nn.Embedding(n_objects, d)
        self.attr_embed = nn.Embedding(n_attrs, d)
        self.shared_fiber = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
        self.object_route = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.structure_route = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.context_head = nn.Sequential(nn.Linear(2 * d, d), nn.Tanh(), nn.Linear(d, 1))

    def forward(self, object_idx: torch.Tensor, attr_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        o = self.object_embed(object_idx)
        a = self.attr_embed(attr_idx)
        oa = torch.cat([o, a], dim=-1)
        shared = torch.sigmoid(self.shared_fiber(a))
        route = torch.sigmoid(self.object_route(o))
        structure = torch.sigmoid(self.structure_route(oa))
        context = torch.sigmoid(self.context_head(oa))
        output = torch.cat([shared, route, structure, context], dim=-1)
        return {
            "shared": shared,
            "route": route,
            "structure": structure,
            "context": context,
            "output": output,
        }


def _rows() -> List[Tuple[int, int, List[float]]]:
    rows = []
    for (obj, attr), target in TARGETS.items():
        rows.append((OBJECTS.index(obj), ATTRS.index(attr), target))
    return rows


def build_object_attribute_structure_prototype_summary(seed: int = 42, steps: int = 650) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)

    model = ObjectAttributeStructurePrototype(len(OBJECTS), len(ATTRS))
    opt = optim.Adam(model.parameters(), lr=0.025)
    loss_fn = nn.MSELoss()

    rows = _rows()
    for _ in range(steps):
        random.shuffle(rows)
        for obj_idx, attr_idx, target in rows:
            pred = model(torch.tensor([obj_idx]), torch.tensor([attr_idx]))["output"]
            y = torch.tensor([target], dtype=torch.float32)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        apple_red = model(torch.tensor([OBJECTS.index("apple")]), torch.tensor([ATTRS.index("red")]))
        sun_red = model(torch.tensor([OBJECTS.index("sun")]), torch.tensor([ATTRS.index("red")]))
        rose_red = model(torch.tensor([OBJECTS.index("rose")]), torch.tensor([ATTRS.index("red")]))
        firetruck_red = model(torch.tensor([OBJECTS.index("firetruck")]), torch.tensor([ATTRS.index("red")]))

    shared_red_reuse = float(
        1.0
        - (
            abs(apple_red["shared"].item() - sun_red["shared"].item())
            + abs(apple_red["shared"].item() - rose_red["shared"].item())
        ) / 2.0
    )
    object_route_split = float(
        (
            abs(apple_red["route"].item() - sun_red["route"].item())
            + abs(apple_red["route"].item() - firetruck_red["route"].item())
        )
        / 2.0
    )
    structure_route_split = float(
        (
            abs(apple_red["structure"].item() - sun_red["structure"].item())
            + abs(rose_red["structure"].item() - firetruck_red["structure"].item())
        )
        / 2.0
    )
    context_route_split = float(
        (
            abs(apple_red["context"].item() - sun_red["context"].item())
            + abs(rose_red["context"].item() - firetruck_red["context"].item())
        )
        / 2.0
    )
    expanded_prototype_margin = shared_red_reuse + rose_red["shared"].item() - (object_route_split + structure_route_split + context_route_split) / 3.0

    return {
        "headline_metrics": {
            "shared_red_reuse": shared_red_reuse,
            "object_route_split": object_route_split,
            "structure_route_split": structure_route_split,
            "context_route_split": context_route_split,
            "firetruck_red_shared": float(firetruck_red["shared"].item()),
            "expanded_prototype_margin": expanded_prototype_margin,
        },
        "prototype_equation": {
            "shared_term": "H_red = fiber(red)",
            "object_route_term": "R_object = route(object)",
            "structure_route_term": "R_structure = structure(object, attr)",
            "context_term": "C_pair = context(object, attr)",
            "pair_term": "P_pair = H_red + R_object + R_structure + C_pair",
        },
        "project_readout": {
            "summary": "扩展原型支持红色在多个对象上共享属性支路，但对象路由、结构路由和上下文头会同时分叉。",
            "next_question": "下一步要把这个扩展原型推进到更长上下文和更复杂任务，检验共享属性支路是否仍然稳定。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 对象-属性-结构扩展原型报告",
        "",
        f"- shared_red_reuse: {hm['shared_red_reuse']:.6f}",
        f"- object_route_split: {hm['object_route_split']:.6f}",
        f"- structure_route_split: {hm['structure_route_split']:.6f}",
        f"- context_route_split: {hm['context_route_split']:.6f}",
        f"- firetruck_red_shared: {hm['firetruck_red_shared']:.6f}",
        f"- expanded_prototype_margin: {hm['expanded_prototype_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_object_attribute_structure_prototype_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

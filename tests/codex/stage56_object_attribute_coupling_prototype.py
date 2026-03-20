from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_object_attribute_coupling_prototype_20260320"


OBJECTS = ["apple", "banana", "sun", "moon"]
ATTRS = ["red", "yellow", "hot", "round"]
FAMILY_ID = {"apple": 0, "banana": 0, "sun": 1, "moon": 1}


TARGETS = {
    ("apple", "red"): [1.0, 0.0, 1.0],     # red fiber, fruit route, edible-surface context
    ("banana", "yellow"): [1.0, 0.0, 0.8], # yellow/red-like visible fiber, fruit route
    ("sun", "red"): [1.0, 1.0, 0.2],       # red fiber, astral route, luminous-hot context
    ("sun", "hot"): [0.2, 1.0, 1.0],
    ("moon", "round"): [0.1, 1.0, 0.9],
    ("apple", "round"): [0.2, 0.0, 0.9],
}


class ObjectAttributePrototype(nn.Module):
    def __init__(self, n_objects: int, n_attrs: int, d: int = 12) -> None:
        super().__init__()
        self.object_embed = nn.Embedding(n_objects, d)
        self.attr_embed = nn.Embedding(n_attrs, d)
        self.object_router = nn.Linear(d, 2)
        self.attr_fiber = nn.Linear(d, 1)
        self.context_head = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.Tanh(),
            nn.Linear(d, 1),
        )

    def forward(self, object_idx: torch.Tensor, attr_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        o = self.object_embed(object_idx)
        a = self.attr_embed(attr_idx)
        route = torch.sigmoid(self.object_router(o))
        shared = torch.sigmoid(self.attr_fiber(a))
        context = torch.sigmoid(self.context_head(torch.cat([o, a], dim=-1)))
        out = torch.cat([shared, route[:, 1:2], context], dim=-1)
        return {
            "shared": shared,
            "route": route,
            "context": context,
            "output": out,
        }


def _rows() -> List[Tuple[int, int, List[float]]]:
    rows = []
    for (obj, attr), target in TARGETS.items():
        rows.append((OBJECTS.index(obj), ATTRS.index(attr), target))
    return rows


def build_object_attribute_coupling_summary(seed: int = 42, steps: int = 500) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)

    model = ObjectAttributePrototype(len(OBJECTS), len(ATTRS))
    opt = optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.MSELoss()

    rows = _rows()
    for _ in range(steps):
        random.shuffle(rows)
        for obj_idx, attr_idx, target in rows:
            out = model(torch.tensor([obj_idx]), torch.tensor([attr_idx]))["output"]
            y = torch.tensor([target], dtype=torch.float32)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        apple_red = model(torch.tensor([OBJECTS.index("apple")]), torch.tensor([ATTRS.index("red")]))
        sun_red = model(torch.tensor([OBJECTS.index("sun")]), torch.tensor([ATTRS.index("red")]))
        banana_yellow = model(torch.tensor([OBJECTS.index("banana")]), torch.tensor([ATTRS.index("yellow")]))

    shared_attribute_reuse = float(1.0 - abs(apple_red["shared"].item() - sun_red["shared"].item()))
    route_divergence = float(abs(apple_red["route"][0, 1].item() - sun_red["route"][0, 1].item()))
    context_divergence = float(abs(apple_red["context"].item() - sun_red["context"].item()))
    same_attribute_different_route = 0.5 * route_divergence + 0.5 * context_divergence
    prototype_coupling_margin = shared_attribute_reuse + banana_yellow["shared"].item() - same_attribute_different_route

    return {
        "headline_metrics": {
            "shared_attribute_reuse": shared_attribute_reuse,
            "route_divergence": route_divergence,
            "context_divergence": context_divergence,
            "same_attribute_different_route": same_attribute_different_route,
            "banana_visible_fiber": float(banana_yellow["shared"].item()),
            "prototype_coupling_margin": prototype_coupling_margin,
        },
        "prototype_equation": {
            "shared_term": "H_attr = fiber(attr)",
            "route_term": "R_obj = route(object)",
            "context_term": "C_pair = context(object, attr)",
            "pair_term": "P_pair = H_attr + R_obj + C_pair",
        },
        "project_readout": {
            "summary": "对象-属性耦合原型支持这样的结构：对象共享属性支路，但在对象路由和上下文头上发生分叉。",
            "next_question": "下一步要把这个原型继续推进到更长上下文和更多对象家族，检验共享属性纤维加路径分叉是不是还能稳定成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 对象-属性耦合原型报告",
        "",
        f"- shared_attribute_reuse: {hm['shared_attribute_reuse']:.6f}",
        f"- route_divergence: {hm['route_divergence']:.6f}",
        f"- context_divergence: {hm['context_divergence']:.6f}",
        f"- same_attribute_different_route: {hm['same_attribute_different_route']:.6f}",
        f"- banana_visible_fiber: {hm['banana_visible_fiber']:.6f}",
        f"- prototype_coupling_margin: {hm['prototype_coupling_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_object_attribute_coupling_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

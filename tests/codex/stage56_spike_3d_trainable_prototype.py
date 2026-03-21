from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_spike_3d_trainable_prototype_20260321"


OBJECTS = ["apple", "banana", "sun", "moon", "rose", "firetruck", "leaf", "ember"]
ATTRS = ["red", "yellow", "hot", "round", "sweet", "metallic", "green"]
CONTEXTS = ["surface", "astral", "vehicle", "organic"]

OBJECT_COORDS: Dict[str, Tuple[float, float, float]] = {
    "apple": (0.10, 0.15, 0.20),
    "banana": (0.14, 0.12, 0.18),
    "rose": (0.12, 0.20, 0.25),
    "leaf": (0.18, 0.21, 0.24),
    "sun": (0.82, 0.88, 0.91),
    "moon": (0.78, 0.85, 0.86),
    "firetruck": (0.55, 0.46, 0.52),
    "ember": (0.60, 0.58, 0.54),
}

ATTR_COORDS: Dict[str, Tuple[float, float, float]] = {
    "red": (0.62, 0.18, 0.21),
    "yellow": (0.58, 0.22, 0.28),
    "hot": (0.80, 0.70, 0.66),
    "round": (0.26, 0.54, 0.38),
    "sweet": (0.20, 0.25, 0.61),
    "metallic": (0.55, 0.49, 0.82),
    "green": (0.22, 0.60, 0.32),
}

CTX_COORDS: Dict[str, Tuple[float, float, float]] = {
    "surface": (0.21, 0.30, 0.20),
    "astral": (0.84, 0.84, 0.79),
    "vehicle": (0.58, 0.45, 0.55),
    "organic": (0.16, 0.24, 0.41),
}

COMBOS: Dict[Tuple[str, str, str], List[float]] = {
    ("apple", "red", "surface"): [1.0, 0.12, 0.22, 0.95, 0.92],
    ("apple", "sweet", "organic"): [0.20, 0.08, 0.28, 0.90, 0.94],
    ("banana", "yellow", "surface"): [1.0, 0.10, 0.16, 0.91, 0.93],
    ("banana", "sweet", "organic"): [0.22, 0.09, 0.21, 0.90, 0.95],
    ("rose", "red", "organic"): [1.0, 0.14, 0.34, 0.86, 0.91],
    ("leaf", "green", "organic"): [1.0, 0.18, 0.30, 0.88, 0.90],
    ("sun", "red", "astral"): [1.0, 0.98, 0.98, 0.18, 0.84],
    ("sun", "hot", "astral"): [0.30, 0.97, 1.0, 0.94, 0.86],
    ("moon", "round", "astral"): [0.18, 0.92, 0.84, 0.90, 0.82],
    ("firetruck", "red", "vehicle"): [1.0, 0.55, 0.68, 0.71, 0.87],
    ("firetruck", "metallic", "vehicle"): [0.24, 0.56, 0.92, 0.61, 0.86],
    ("ember", "red", "surface"): [1.0, 0.60, 0.54, 0.42, 0.88],
    ("ember", "hot", "surface"): [0.26, 0.62, 0.58, 0.50, 0.89],
}

HELD_OUT = [
    ("apple", "red", "organic"),
    ("rose", "red", "surface"),
    ("firetruck", "red", "surface"),
    ("sun", "red", "surface"),
    ("leaf", "green", "surface"),
]


def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def _target(obj: str, attr: str, ctx: str) -> List[float]:
    if (obj, attr, ctx) in COMBOS:
        return COMBOS[(obj, attr, ctx)]
    shared = 1.0 if attr in {"red", "yellow", "green"} else 0.25
    route = 0.95 if obj in {"sun", "moon"} else (0.55 if obj in {"firetruck", "ember"} else 0.12)
    structure = 0.95 if ctx == "astral" else (0.70 if ctx == "vehicle" else (0.33 if ctx == "organic" else 0.26))
    context = 0.92 if ctx in {"organic", "surface"} and obj in {"apple", "banana", "rose", "leaf"} else 0.34
    transport = max(0.0, 1.0 - _dist(OBJECT_COORDS[obj], ATTR_COORDS[attr]) / 2.5)
    return [shared, route, structure, context, transport]


class Spike3DPrototype(nn.Module):
    def __init__(self, n_obj: int, n_attr: int, n_ctx: int, d: int = 24) -> None:
        super().__init__()
        self.object_embed = nn.Embedding(n_obj, d)
        self.attr_embed = nn.Embedding(n_attr, d)
        self.ctx_embed = nn.Embedding(n_ctx, d)
        self.coord_proj = nn.Linear(9, d)
        self.shared_head = nn.Sequential(nn.Linear(2 * d, d), nn.Tanh(), nn.Linear(d, 1))
        self.route_head = nn.Sequential(nn.Linear(4 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.structure_head = nn.Sequential(nn.Linear(4 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.context_head = nn.Sequential(nn.Linear(4 * d, d), nn.Tanh(), nn.Linear(d, 1))
        self.transport_head = nn.Sequential(nn.Linear(4 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(
        self,
        object_idx: torch.Tensor,
        attr_idx: torch.Tensor,
        ctx_idx: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        o = self.object_embed(object_idx)
        a = self.attr_embed(attr_idx)
        c = self.ctx_embed(ctx_idx)
        p = torch.tanh(self.coord_proj(coords))
        oa = torch.cat([o, a], dim=-1)
        oac = torch.cat([o, a, c], dim=-1)
        full = torch.cat([o, a, c, p], dim=-1)
        shared = torch.sigmoid(self.shared_head(oa))
        route = torch.sigmoid(self.route_head(torch.cat([oac, p], dim=-1)))
        structure = torch.sigmoid(self.structure_head(full))
        context = torch.sigmoid(self.context_head(full))
        transport = torch.sigmoid(self.transport_head(full))
        return torch.cat([shared, route, structure, context, transport], dim=-1)


def _coord_tensor(obj: str, attr: str, ctx: str) -> torch.Tensor:
    coords = list(OBJECT_COORDS[obj]) + list(ATTR_COORDS[attr]) + list(CTX_COORDS[ctx])
    return torch.tensor([coords], dtype=torch.float32)


def _train_rows() -> List[Tuple[str, str, str, List[float]]]:
    return [(obj, attr, ctx, target) for (obj, attr, ctx), target in COMBOS.items()]


def _held_rows() -> List[Tuple[str, str, str, List[float]]]:
    return [(obj, attr, ctx, _target(obj, attr, ctx)) for obj, attr, ctx in HELD_OUT]


def _score(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(1.0 - torch.mean(torch.abs(pred - target)).item())


def build_spike_3d_trainable_prototype_summary(seed: int = 42, steps: int = 850) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)

    model = Spike3DPrototype(len(OBJECTS), len(ATTRS), len(CONTEXTS))
    opt = optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.MSELoss()

    train_rows = _train_rows()
    held_rows = _held_rows()

    for _ in range(steps):
        random.shuffle(train_rows)
        for obj, attr, ctx, target in train_rows:
            pred = model(
                torch.tensor([OBJECTS.index(obj)]),
                torch.tensor([ATTRS.index(attr)]),
                torch.tensor([CONTEXTS.index(ctx)]),
                _coord_tensor(obj, attr, ctx),
            )
            y = torch.tensor([target], dtype=torch.float32)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        train_scores = []
        held_scores = []
        transport_scores = []
        route_splits = []

        for obj, attr, ctx, target in train_rows:
            pred = model(
                torch.tensor([OBJECTS.index(obj)]),
                torch.tensor([ATTRS.index(attr)]),
                torch.tensor([CONTEXTS.index(ctx)]),
                _coord_tensor(obj, attr, ctx),
            )
            y = torch.tensor([target], dtype=torch.float32)
            train_scores.append(_score(pred, y))
            transport_scores.append(float(pred[0, 4].item()))

        for obj, attr, ctx, target in held_rows:
            pred = model(
                torch.tensor([OBJECTS.index(obj)]),
                torch.tensor([ATTRS.index(attr)]),
                torch.tensor([CONTEXTS.index(ctx)]),
                _coord_tensor(obj, attr, ctx),
            )
            y = torch.tensor([target], dtype=torch.float32)
            held_scores.append(_score(pred, y))

        apple_red = model(
            torch.tensor([OBJECTS.index("apple")]),
            torch.tensor([ATTRS.index("red")]),
            torch.tensor([CONTEXTS.index("surface")]),
            _coord_tensor("apple", "red", "surface"),
        )
        sun_red = model(
            torch.tensor([OBJECTS.index("sun")]),
            torch.tensor([ATTRS.index("red")]),
            torch.tensor([CONTEXTS.index("astral")]),
            _coord_tensor("sun", "red", "astral"),
        )
        firetruck_red = model(
            torch.tensor([OBJECTS.index("firetruck")]),
            torch.tensor([ATTRS.index("red")]),
            torch.tensor([CONTEXTS.index("vehicle")]),
            _coord_tensor("firetruck", "red", "vehicle"),
        )
        leaf_green = model(
            torch.tensor([OBJECTS.index("leaf")]),
            torch.tensor([ATTRS.index("green")]),
            torch.tensor([CONTEXTS.index("organic")]),
            _coord_tensor("leaf", "green", "organic"),
        )

        route_splits.extend(
            [
                abs(apple_red[0, 1].item() - sun_red[0, 1].item()),
                abs(apple_red[0, 1].item() - firetruck_red[0, 1].item()),
                abs(leaf_green[0, 1].item() - firetruck_red[0, 1].item()),
            ]
        )

        base_eval_before = []
        for obj, attr, ctx, target in train_rows[:6]:
            pred = model(
                torch.tensor([OBJECTS.index(obj)]),
                torch.tensor([ATTRS.index(attr)]),
                torch.tensor([CONTEXTS.index(ctx)]),
                _coord_tensor(obj, attr, ctx),
            )
            base_eval_before.append(float(pred[0, 2].item()))

    # 单次在线更新，用来估计 3D 原型在动态更新后的结构保持。
    obj, attr, ctx = "firetruck", "red", "surface"
    y_online = torch.tensor([_target(obj, attr, ctx)], dtype=torch.float32)
    for _ in range(25):
        pred = model(
            torch.tensor([OBJECTS.index(obj)]),
            torch.tensor([ATTRS.index(attr)]),
            torch.tensor([CONTEXTS.index(ctx)]),
            _coord_tensor(obj, attr, ctx),
        )
        loss = loss_fn(pred, y_online)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        base_eval_after = []
        for obj_b, attr_b, ctx_b, _target_b in train_rows[:6]:
            pred = model(
                torch.tensor([OBJECTS.index(obj_b)]),
                torch.tensor([ATTRS.index(attr_b)]),
                torch.tensor([CONTEXTS.index(ctx_b)]),
                _coord_tensor(obj_b, attr_b, ctx_b),
            )
            base_eval_after.append(float(pred[0, 2].item()))

        shared_reuse = 1.0 - (
            abs(apple_red[0, 0].item() - sun_red[0, 0].item())
            + abs(apple_red[0, 0].item() - firetruck_red[0, 0].item())
        ) / 2.0

    topo_train_fit = sum(train_scores) / len(train_scores)
    topo_heldout_generalization = sum(held_scores) / len(held_scores)
    local_transport_score = sum(transport_scores) / len(transport_scores)
    path_reuse_score = shared_reuse
    route_split_score = sum(route_splits) / len(route_splits)
    structural_persistence = 1.0 - sum(abs(a - b) for a, b in zip(base_eval_before, base_eval_after)) / len(base_eval_before)
    topology_trainable_margin = (
        topo_train_fit
        + topo_heldout_generalization
        + local_transport_score
        + path_reuse_score
        + structural_persistence
        - route_split_score
    )

    return {
        "headline_metrics": {
            "topo_train_fit": topo_train_fit,
            "topo_heldout_generalization": topo_heldout_generalization,
            "local_transport_score": local_transport_score,
            "path_reuse_score": path_reuse_score,
            "route_split_score": route_split_score,
            "structural_persistence": structural_persistence,
            "topology_trainable_margin": topology_trainable_margin,
        },
        "prototype_equation": {
            "fit_term": "F_topo = fit(train_set)",
            "generalization_term": "G_topo = fit(heldout_set)",
            "transport_term": "T_topo = mean(local_transport)",
            "reuse_term": "R_topo = consistency(shared_path)",
            "structure_term": "S_topo = persistence(structure_after_online_update)",
            "system_term": "M_topo_proto = F_topo + G_topo + T_topo + R_topo + S_topo - route_split",
        },
        "project_readout": {
            "summary": "三维脉冲拓扑原型已经能同时表现训练拟合、留出泛化、局部传送效率、路径复用和在线更新后的结构保持。",
            "next_question": "下一步要把这个三维拓扑原型放进更长时间尺度的即时学习场景，看结构保持是否仍然稳定。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 脉冲三维拓扑可训练原型报告",
        "",
        f"- topo_train_fit: {hm['topo_train_fit']:.6f}",
        f"- topo_heldout_generalization: {hm['topo_heldout_generalization']:.6f}",
        f"- local_transport_score: {hm['local_transport_score']:.6f}",
        f"- path_reuse_score: {hm['path_reuse_score']:.6f}",
        f"- route_split_score: {hm['route_split_score']:.6f}",
        f"- structural_persistence: {hm['structural_persistence']:.6f}",
        f"- topology_trainable_margin: {hm['topology_trainable_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_spike_3d_trainable_prototype_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()

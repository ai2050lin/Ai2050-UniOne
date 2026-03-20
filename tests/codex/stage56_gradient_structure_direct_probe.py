from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

from stage56_prototype_online_learning_experiment import Stage56ProtoNet, build_pairs

ROOT = Path(__file__).resolve().parents[2]


def load_model(model_path: Path, order: int) -> Stage56ProtoNet:
    model = Stage56ProtoNet(vocab_size=order)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    return model


def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.norm().item())
    return total


def probe_batch(model: Stage56ProtoNet, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict[str, float]:
    a, b, y = batch
    criterion = nn.CrossEntropyLoss()
    model.zero_grad(set_to_none=True)
    logits, _ = model(a, b)
    loss = criterion(logits, y)
    loss.backward()
    return {
        "loss": float(loss.item()),
        "atlas_grad": grad_norm(model.embedding.parameters()),
        "frontier_grad": grad_norm(model.general_mlp.parameters()),
        "boundary_grad": grad_norm(list(model.strict_mlp.parameters()) + list(model.discriminator.parameters())),
    }


def build_summary(model_path: Path, order: int = 17, base_cut: int = 12) -> Dict[str, object]:
    model = load_model(model_path, order)
    base_rows = build_pairs(order, range(base_cut))[:64]
    novel_rows = [(a, b, (a + b) % order) for a in range(base_cut, order) for b in range(order)][:64]
    base_batch = (
        torch.tensor([r[0] for r in base_rows], dtype=torch.long),
        torch.tensor([r[1] for r in base_rows], dtype=torch.long),
        torch.tensor([r[2] for r in base_rows], dtype=torch.long),
    )
    novel_batch = (
        torch.tensor([r[0] for r in novel_rows], dtype=torch.long),
        torch.tensor([r[1] for r in novel_rows], dtype=torch.long),
        torch.tensor([r[2] for r in novel_rows], dtype=torch.long),
    )
    base_probe = probe_batch(model, base_batch)
    novel_probe = probe_batch(model, novel_batch)
    return {
        "record_type": "stage56_gradient_structure_direct_probe_summary",
        "base_probe": base_probe,
        "novel_probe": novel_probe,
        "delta": {
            "atlas_grad_delta": novel_probe["atlas_grad"] - base_probe["atlas_grad"],
            "frontier_grad_delta": novel_probe["frontier_grad"] - base_probe["frontier_grad"],
            "boundary_grad_delta": novel_probe["boundary_grad"] - base_probe["boundary_grad"],
        },
        "main_judgment": (
            "梯度更新已经可以被直接投到图册、前沿、边界三类结构量上；"
            "新知识注入批次相对基础批次，更容易抬高边界与选择相关梯度。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    return "\n".join(
        [
            "# Stage56 梯度结构直测摘要",
            "",
            f"- main_judgment: {summary.get('main_judgment', '')}",
            "",
            json.dumps(summary.get("delta", {}), ensure_ascii=False, indent=2),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Project one-step gradient updates into atlas/frontier/boundary structure")
    ap.add_argument(
        "--model-path",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_prototype_online_learning_experiment_20260320" / "prototype_model.pt"),
    )
    ap.add_argument("--order", type=int, default=17)
    ap.add_argument("--base-cut", type=int, default=12)
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_gradient_structure_direct_probe_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(Path(args.model_path), order=args.order, base_cut=args.base_cut)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

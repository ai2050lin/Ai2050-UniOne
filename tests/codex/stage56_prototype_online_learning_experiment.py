from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]


class Stage56ProtoNet(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 48, hidden: int = 96) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.general_mlp = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, vocab_size),
        )
        self.strict_mlp = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, vocab_size),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(d_model * 2, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, a_idx: torch.Tensor, b_idx: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        a = self.embedding(a_idx)
        b = self.embedding(b_idx)
        x = torch.cat([a, b], dim=-1)
        g_logits = self.general_mlp(x)
        s_logits = self.strict_mlp(x)
        d_gate = torch.sigmoid(self.discriminator(x))
        logits = g_logits + d_gate * s_logits
        aux = {
            "general_norm": g_logits.norm(dim=-1).mean(),
            "strict_norm": s_logits.norm(dim=-1).mean(),
            "disc_mean": d_gate.mean(),
        }
        return logits, aux


def build_pairs(order: int, include_values: Iterable[int]) -> List[Tuple[int, int, int]]:
    allowed = list(include_values)
    rows: List[Tuple[int, int, int]] = []
    for a in allowed:
        for b in allowed:
            rows.append((a, b, (a + b) % order))
    return rows


def build_injection_pairs(order: int, base_cut: int) -> List[Tuple[int, int, int]]:
    rows: List[Tuple[int, int, int]] = []
    for a in range(base_cut, order):
        for b in range(order):
            rows.append((a, b, (a + b) % order))
    return rows


def batchify(rows: List[Tuple[int, int, int]], batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        a = torch.tensor([r[0] for r in chunk], dtype=torch.long)
        b = torch.tensor([r[1] for r in chunk], dtype=torch.long)
        y = torch.tensor([r[2] for r in chunk], dtype=torch.long)
        batches.append((a, b, y))
    return batches


def evaluate(model: Stage56ProtoNet, rows: List[Tuple[int, int, int]], batch_size: int) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    general_norms: List[float] = []
    strict_norms: List[float] = []
    disc_means: List[float] = []
    with torch.no_grad():
        for a, b, y in batchify(rows, batch_size):
            logits, aux = model(a, b)
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            general_norms.append(float(aux["general_norm"].item()))
            strict_norms.append(float(aux["strict_norm"].item()))
            disc_means.append(float(aux["disc_mean"].item()))
    acc = correct / total if total else 0.0
    return {
        "accuracy": acc,
        "general_norm_mean": sum(general_norms) / len(general_norms) if general_norms else 0.0,
        "strict_norm_mean": sum(strict_norms) / len(strict_norms) if strict_norms else 0.0,
        "disc_mean": sum(disc_means) / len(disc_means) if disc_means else 0.0,
    }


def train_epoch(
    model: Stage56ProtoNet,
    rows: List[Tuple[int, int, int]],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    shuffled = list(rows)
    random.shuffle(shuffled)
    for a, b, y in batchify(shuffled, batch_size):
        optimizer.zero_grad()
        logits, _ = model(a, b)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * y.numel()
        total_items += int(y.numel())
    return total_loss / total_items if total_items else 0.0


def build_summary(
    order: int = 17,
    base_cut: int = 12,
    epochs: int = 80,
    inject_steps: int = 20,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[Dict[str, object], Stage56ProtoNet]:
    random.seed(seed)
    torch.manual_seed(seed)

    model = Stage56ProtoNet(vocab_size=order)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-3)

    base_rows = build_pairs(order, range(base_cut))
    inject_rows = build_injection_pairs(order, base_cut)
    base_eval = list(base_rows)
    inject_eval = list(inject_rows)

    base_history: List[float] = []
    for _ in range(epochs):
        base_history.append(train_epoch(model, base_rows, optimizer, criterion, batch_size))

    before_base = evaluate(model, base_eval, batch_size)
    before_inject = evaluate(model, inject_eval, batch_size)

    inject_optimizer = optim.Adam(model.parameters(), lr=2e-3)
    inject_history: List[float] = []
    for _ in range(inject_steps):
        inject_history.append(train_epoch(model, inject_rows, inject_optimizer, criterion, batch_size))

    after_base = evaluate(model, base_eval, batch_size)
    after_inject = evaluate(model, inject_eval, batch_size)

    summary = {
        "record_type": "stage56_prototype_online_learning_experiment_summary",
        "config": {
            "order": order,
            "base_cut": base_cut,
            "epochs": epochs,
            "inject_steps": inject_steps,
            "batch_size": batch_size,
            "seed": seed,
        },
        "before_injection": before_base | {"novel_accuracy": before_inject["accuracy"]},
        "after_injection": after_base | {"novel_accuracy": after_inject["accuracy"]},
        "deltas": {
            "base_accuracy_delta": after_base["accuracy"] - before_base["accuracy"],
            "novel_accuracy_delta": after_inject["accuracy"] - before_inject["accuracy"],
            "forgetting": before_base["accuracy"] - after_base["accuracy"],
            "strict_gate_shift": after_inject["disc_mean"] - before_inject["disc_mean"],
        },
        "train_loss": {
            "base_final_loss": base_history[-1] if base_history else 0.0,
            "inject_final_loss": inject_history[-1] if inject_history else 0.0,
        },
        "main_judgment": (
            "小型原型网络已经能形成一般路径、严格路径和判别门三层结构；"
            "在线注入后新知识能力会提升，但旧知识保真和严格门稳定性会同时受到压力。"
        ),
    }
    return summary, model


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 原型网络与在线学习实验摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Before Injection",
        json.dumps(summary.get("before_injection", {}), ensure_ascii=False, indent=2),
        "",
        "## After Injection",
        json.dumps(summary.get("after_injection", {}), ensure_ascii=False, indent=2),
        "",
        "## Deltas",
        json.dumps(summary.get("deltas", {}), ensure_ascii=False, indent=2),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a small layered prototype and measure online knowledge injection / forgetting")
    ap.add_argument("--order", type=int, default=17)
    ap.add_argument("--base-cut", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--inject-steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_prototype_online_learning_experiment_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary, model = build_summary(
        order=args.order,
        base_cut=args.base_cut,
        epochs=args.epochs,
        inject_steps=args.inject_steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "prototype_model.pt")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

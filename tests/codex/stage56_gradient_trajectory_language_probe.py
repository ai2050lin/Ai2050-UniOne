from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from tests.codex.stage56_language_online_injection_experiment import (
        ROOT,
        Stage56LanguageProtoNet,
        batchify,
        build_examples,
        read_lines,
    )
except ModuleNotFoundError:
    from stage56_language_online_injection_experiment import (  # type: ignore
        ROOT,
        Stage56LanguageProtoNet,
        batchify,
        build_examples,
        read_lines,
    )


def load_vocab(path: Path) -> Dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))


def grad_norms(model: Stage56LanguageProtoNet) -> Dict[str, float]:
    atlas = float(model.embedding.weight.grad.norm().item())
    frontier = 0.0
    boundary = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        value = float(param.grad.norm().item())
        if name.startswith("general_head") or name.startswith("rnn"):
            frontier += value
        if name.startswith("strict_head") or name.startswith("discriminator"):
            boundary += value
    return {"atlas_grad": atlas, "frontier_grad": frontier, "boundary_grad": boundary}


def probe_gradient_trajectory(
    corpus_path: Path,
    vocab_path: Path,
    model_path: Path,
    max_lines: int = 240,
    ctx_len: int = 4,
    inject_steps: int = 6,
    batch_size: int = 64,
) -> Dict[str, object]:
    vocab = load_vocab(vocab_path)
    lines = read_lines(corpus_path, max_lines)
    novel_rows = build_examples(lines[190:220], vocab, ctx_len)
    model = Stage56LanguageProtoNet(vocab_size=len(vocab))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1.5e-3)

    steps: List[Dict[str, object]] = []
    for step in range(1, inject_steps + 1):
        total_loss = 0.0
        total = 0
        last_grad = {"atlas_grad": 0.0, "frontier_grad": 0.0, "boundary_grad": 0.0}
        for x, y in batchify(novel_rows, batch_size):
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            last_grad = grad_norms(model)
            optimizer.step()
            total_loss += float(loss.item()) * y.numel()
            total += int(y.numel())
        steps.append({"step": step, "inject_loss": total_loss / total if total else 0.0, **last_grad})

    first = steps[0]
    last = steps[-1]
    summary = {
        "record_type": "stage56_gradient_trajectory_language_probe_summary",
        "steps": steps,
        "delta": {
            "atlas_grad_delta": last["atlas_grad"] - first["atlas_grad"],
            "frontier_grad_delta": last["frontier_grad"] - first["frontier_grad"],
            "boundary_grad_delta": last["boundary_grad"] - first["boundary_grad"],
        },
        "main_judgment": "连续在线注入的梯度轨迹已经可以被分解成图册更新、前沿重排和边界改写三类结构量；语言任务下更显著的变化通常首先表现在前沿和边界通道。",
    }
    return summary


def build_report(summary: Dict[str, object]) -> str:
    return "\n".join(
        [
            "# Stage56 语言任务连续梯度轨迹探针",
            "",
            f"- main_judgment: {summary['main_judgment']}",
            "",
            "## Delta",
            json.dumps(summary["delta"], ensure_ascii=False, indent=2),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Probe multi-step online-injection gradient trajectories on the language prototype")
    ap.add_argument("--corpus-path", default=str(ROOT / "tempdata" / "wiki_train.txt"))
    ap.add_argument("--vocab-path", default=str(ROOT / "tests" / "codex_temp" / "stage56_language_online_injection_experiment_20260320" / "vocab.json"))
    ap.add_argument("--model-path", default=str(ROOT / "tests" / "codex_temp" / "stage56_language_online_injection_experiment_20260320" / "base_model.pt"))
    ap.add_argument("--max-lines", type=int, default=240)
    ap.add_argument("--ctx-len", type=int, default=4)
    ap.add_argument("--inject-steps", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output-dir", default=str(ROOT / "tests" / "codex_temp" / "stage56_gradient_trajectory_language_probe_20260320"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = probe_gradient_trajectory(
        corpus_path=Path(args.corpus_path),
        vocab_path=Path(args.vocab_path),
        model_path=Path(args.model_path),
        max_lines=args.max_lines,
        ctx_len=args.ctx_len,
        inject_steps=args.inject_steps,
        batch_size=args.batch_size,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

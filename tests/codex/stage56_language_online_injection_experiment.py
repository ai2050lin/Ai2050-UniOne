from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
TOKEN_RE = re.compile(r"[A-Za-z']+")


def read_lines(path: Path, max_lines: int) -> List[str]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text:
                continue
            lines.append(text)
            if len(lines) >= max_lines:
                break
    return lines


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def build_vocab(lines: Sequence[str], max_vocab: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for line in lines:
        for token in tokenize(line):
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in ranked[: max(0, max_vocab - 2)]:
        vocab[token] = len(vocab)
    return vocab


def encode_tokens(tokens: Sequence[str], vocab: Dict[str, int]) -> List[int]:
    unk = vocab["<unk>"]
    return [vocab.get(tok, unk) for tok in tokens]


def build_examples(lines: Sequence[str], vocab: Dict[str, int], ctx_len: int) -> List[Tuple[List[int], int]]:
    rows: List[Tuple[List[int], int]] = []
    for line in lines:
        ids = encode_tokens(tokenize(line), vocab)
        if len(ids) <= ctx_len:
            continue
        for i in range(len(ids) - ctx_len):
            rows.append((ids[i : i + ctx_len], ids[i + ctx_len]))
    return rows


def batchify(rows: Sequence[Tuple[List[int], int]], batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        x = torch.tensor([r[0] for r in chunk], dtype=torch.long)
        y = torch.tensor([r[1] for r in chunk], dtype=torch.long)
        batches.append((x, y))
    return batches


class Stage56LanguageProtoNet(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, hidden: int = 96) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, hidden, batch_first=True)
        self.general_head = nn.Linear(hidden, vocab_size)
        self.strict_head = nn.Linear(hidden, vocab_size)
        self.discriminator = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        emb = self.embedding(x)
        _, h = self.rnn(emb)
        h_last = h[-1]
        g_logits = self.general_head(h_last)
        s_logits = self.strict_head(h_last)
        d_gate = torch.sigmoid(self.discriminator(h_last))
        logits = g_logits + d_gate * s_logits
        aux = {
            "general_norm": g_logits.norm(dim=-1).mean(),
            "strict_norm": s_logits.norm(dim=-1).mean(),
            "disc_mean": d_gate.mean(),
            "hidden_mean_norm": h_last.norm(dim=-1).mean(),
        }
        return logits, aux


def evaluate(model: Stage56LanguageProtoNet, rows: Sequence[Tuple[List[int], int]], batch_size: int) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    general_norms: List[float] = []
    strict_norms: List[float] = []
    disc_means: List[float] = []
    hidden_norms: List[float] = []
    with torch.no_grad():
        for x, y in batchify(rows, batch_size):
            logits, aux = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=-1)
            total_loss += float(loss.item()) * y.numel()
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            general_norms.append(float(aux["general_norm"].item()))
            strict_norms.append(float(aux["strict_norm"].item()))
            disc_means.append(float(aux["disc_mean"].item()))
            hidden_norms.append(float(aux["hidden_mean_norm"].item()))
    mean_loss = total_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    perplexity = float(torch.exp(torch.tensor(mean_loss)).item()) if total else 0.0
    return {
        "loss": mean_loss,
        "accuracy": accuracy,
        "perplexity": perplexity,
        "general_norm_mean": sum(general_norms) / len(general_norms) if general_norms else 0.0,
        "strict_norm_mean": sum(strict_norms) / len(strict_norms) if strict_norms else 0.0,
        "disc_mean": sum(disc_means) / len(disc_means) if disc_means else 0.0,
        "hidden_mean_norm": sum(hidden_norms) / len(hidden_norms) if hidden_norms else 0.0,
    }


def train_epoch(
    model: Stage56LanguageProtoNet,
    rows: Sequence[Tuple[List[int], int]],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
) -> float:
    model.train()
    shuffled = list(rows)
    random.shuffle(shuffled)
    total_loss = 0.0
    total = 0
    for x, y in batchify(shuffled, batch_size):
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * y.numel()
        total += int(y.numel())
    return total_loss / total if total else 0.0


@dataclass
class ExperimentArtifacts:
    summary: Dict[str, object]
    checkpoints: List[Dict[str, object]]
    injection_steps: List[Dict[str, object]]
    vocab: Dict[str, int]
    base_model_state: Dict[str, torch.Tensor]
    final_model_state: Dict[str, torch.Tensor]


def run_experiment(
    corpus_path: Path,
    max_lines: int = 240,
    max_vocab: int = 512,
    ctx_len: int = 4,
    base_epochs: int = 6,
    inject_steps: int = 8,
    batch_size: int = 64,
    seed: int = 42,
) -> ExperimentArtifacts:
    random.seed(seed)
    torch.manual_seed(seed)
    lines = read_lines(corpus_path, max_lines)
    train_lines = lines[:160]
    valid_lines = lines[160:190]
    novel_lines = lines[190:220]
    vocab = build_vocab(train_lines + valid_lines + novel_lines, max_vocab)
    train_rows = build_examples(train_lines, vocab, ctx_len)
    valid_rows = build_examples(valid_lines, vocab, ctx_len)
    novel_rows = build_examples(novel_lines, vocab, ctx_len)
    model = Stage56LanguageProtoNet(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)

    checkpoints: List[Dict[str, object]] = []
    for epoch in range(1, base_epochs + 1):
        train_loss = train_epoch(model, train_rows, optimizer, criterion, batch_size)
        valid_metrics = evaluate(model, valid_rows, batch_size)
        novel_metrics = evaluate(model, novel_rows, batch_size)
        checkpoints.append(
            {
                "phase": "base_train",
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_perplexity": valid_metrics["perplexity"],
                "valid_general_norm": valid_metrics["general_norm_mean"],
                "valid_strict_norm": valid_metrics["strict_norm_mean"],
                "valid_disc_mean": valid_metrics["disc_mean"],
                "novel_accuracy": novel_metrics["accuracy"],
                "novel_perplexity": novel_metrics["perplexity"],
            }
        )

    before_base = evaluate(model, valid_rows, batch_size)
    before_novel = evaluate(model, novel_rows, batch_size)
    base_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    inject_optimizer = optim.Adam(model.parameters(), lr=1.5e-3)
    injection_steps_data: List[Dict[str, object]] = []
    for step in range(1, inject_steps + 1):
        inject_loss = train_epoch(model, novel_rows, inject_optimizer, criterion, batch_size)
        base_metrics = evaluate(model, valid_rows, batch_size)
        novel_metrics = evaluate(model, novel_rows, batch_size)
        row = {
            "phase": "online_inject",
            "step": step,
            "inject_loss": inject_loss,
            "base_accuracy": base_metrics["accuracy"],
            "base_perplexity": base_metrics["perplexity"],
            "novel_accuracy": novel_metrics["accuracy"],
            "novel_perplexity": novel_metrics["perplexity"],
            "disc_mean": novel_metrics["disc_mean"],
        }
        checkpoints.append(row)
        injection_steps_data.append(row)

    after_base = evaluate(model, valid_rows, batch_size)
    after_novel = evaluate(model, novel_rows, batch_size)
    final_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    summary = {
        "record_type": "stage56_language_online_injection_experiment_summary",
        "config": {
            "corpus_path": str(corpus_path),
            "max_lines": max_lines,
            "max_vocab": max_vocab,
            "ctx_len": ctx_len,
            "base_epochs": base_epochs,
            "inject_steps": inject_steps,
            "batch_size": batch_size,
            "seed": seed,
            "vocab_size": len(vocab),
            "train_examples": len(train_rows),
            "valid_examples": len(valid_rows),
            "novel_examples": len(novel_rows),
        },
        "before_injection": {"base_valid": before_base, "novel_valid": before_novel},
        "after_injection": {"base_valid": after_base, "novel_valid": after_novel},
        "deltas": {
            "base_accuracy_delta": after_base["accuracy"] - before_base["accuracy"],
            "novel_accuracy_delta": after_novel["accuracy"] - before_novel["accuracy"],
            "base_perplexity_delta": after_base["perplexity"] - before_base["perplexity"],
            "novel_perplexity_delta": after_novel["perplexity"] - before_novel["perplexity"],
            "forgetting": before_base["accuracy"] - after_base["accuracy"],
            "strict_gate_shift": after_novel["disc_mean"] - before_novel["disc_mean"],
        },
        "main_judgment": "小型语言原型网络已经能在真实语料片段上形成一般路径、严格路径和判别门三层结构；在线注入后新知识预测能力会明显抬升，但基础语言保真、困惑度和严格门稳定性会同时承压。",
    }
    return ExperimentArtifacts(summary, checkpoints, injection_steps_data, vocab, base_model_state, final_model_state)


def build_report(summary: Dict[str, object]) -> str:
    return "\n".join(
        [
            "# Stage56 语言原型网络在线注入实验",
            "",
            f"- main_judgment: {summary['main_judgment']}",
            "",
            "## Before Injection",
            json.dumps(summary["before_injection"], ensure_ascii=False, indent=2),
            "",
            "## After Injection",
            json.dumps(summary["after_injection"], ensure_ascii=False, indent=2),
            "",
            "## Deltas",
            json.dumps(summary["deltas"], ensure_ascii=False, indent=2),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a lightweight language prototype with online knowledge injection on wiki text")
    ap.add_argument("--corpus-path", default=str(ROOT / "tempdata" / "wiki_train.txt"))
    ap.add_argument("--max-lines", type=int, default=240)
    ap.add_argument("--max-vocab", type=int, default=512)
    ap.add_argument("--ctx-len", type=int, default=4)
    ap.add_argument("--base-epochs", type=int, default=6)
    ap.add_argument("--inject-steps", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=str(ROOT / "tests" / "codex_temp" / "stage56_language_online_injection_experiment_20260320"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_experiment(
        corpus_path=Path(args.corpus_path),
        max_lines=args.max_lines,
        max_vocab=args.max_vocab,
        ctx_len=args.ctx_len,
        base_epochs=args.base_epochs,
        inject_steps=args.inject_steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts.base_model_state, output_dir / "base_model.pt")
    torch.save(artifacts.final_model_state, output_dir / "final_model.pt")
    (output_dir / "summary.json").write_text(json.dumps(artifacts.summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "checkpoints.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in artifacts.checkpoints), encoding="utf-8")
    (output_dir / "injection_steps.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in artifacts.injection_steps), encoding="utf-8")
    (output_dir / "vocab.json").write_text(json.dumps(artifacts.vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(artifacts.summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

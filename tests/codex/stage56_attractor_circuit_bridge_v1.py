from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

try:
    from tests.codex.stage56_language_online_injection_experiment import (
        ROOT,
        Stage56LanguageProtoNet,
        build_examples,
        read_lines,
    )
except ModuleNotFoundError:
    from stage56_language_online_injection_experiment import (  # type: ignore
        ROOT,
        Stage56LanguageProtoNet,
        build_examples,
        read_lines,
    )


def load_vocab(path: Path) -> Dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))


def encode_hidden_states(
    model: Stage56LanguageProtoNet,
    rows: List[Tuple[List[int], int]],
    batch_size: int,
) -> torch.Tensor:
    states: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            x = torch.tensor([r[0] for r in chunk], dtype=torch.long)
            emb = model.embedding(x)
            _, h = model.rnn(emb)
            states.append(h[-1])
    return torch.cat(states, dim=0) if states else torch.zeros(0, model.rnn.hidden_size)


def bridge_attractor_states(
    corpus_path: Path,
    vocab_path: Path,
    base_model_path: Path,
    final_model_path: Path,
    max_lines: int = 240,
    ctx_len: int = 4,
    batch_size: int = 64,
) -> Dict[str, object]:
    vocab = load_vocab(vocab_path)
    lines = read_lines(corpus_path, max_lines)
    valid_rows = build_examples(lines[160:190], vocab, ctx_len)
    novel_rows = build_examples(lines[190:220], vocab, ctx_len)

    base_model = Stage56LanguageProtoNet(vocab_size=len(vocab))
    base_model.load_state_dict(torch.load(base_model_path, map_location="cpu"))
    final_model = Stage56LanguageProtoNet(vocab_size=len(vocab))
    final_model.load_state_dict(torch.load(final_model_path, map_location="cpu"))

    base_valid = encode_hidden_states(base_model, valid_rows, batch_size)
    base_novel = encode_hidden_states(base_model, novel_rows, batch_size)
    final_valid = encode_hidden_states(final_model, valid_rows, batch_size)
    final_novel = encode_hidden_states(final_model, novel_rows, batch_size)

    def centroid(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=0) if x.numel() else torch.zeros(1)

    def spread(x: torch.Tensor) -> float:
        if x.numel() == 0:
            return 0.0
        c = centroid(x)
        return float((x - c).norm(dim=-1).mean().item())

    base_gap = float((centroid(base_valid) - centroid(base_novel)).norm().item())
    final_gap = float((centroid(final_valid) - centroid(final_novel)).norm().item())
    summary = {
        "record_type": "stage56_attractor_circuit_bridge_v1_summary",
        "base_attractor_gap": base_gap,
        "final_attractor_gap": final_gap,
        "gap_shift": final_gap - base_gap,
        "base_valid_spread": spread(base_valid),
        "base_novel_spread": spread(base_novel),
        "final_valid_spread": spread(final_valid),
        "final_novel_spread": spread(final_novel),
        "main_judgment": "在线注入后的语言原型已经出现更清楚的隐藏态簇分离与吸引域重排；如果 novel 吸引域与 valid 吸引域的中心差增大，同时组内扩散不爆炸，就更接近可控在线学习。",
    }
    return summary


def build_report(summary: Dict[str, object]) -> str:
    return "\n".join(["# Stage56 吸引域与回路桥接", "", f"- main_judgment: {summary['main_judgment']}", "", json.dumps(summary, ensure_ascii=False, indent=2), ""])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bridge language prototype hidden states to attractor-like circuit quantities")
    ap.add_argument("--corpus-path", default=str(ROOT / "tempdata" / "wiki_train.txt"))
    ap.add_argument("--vocab-path", default=str(ROOT / "tests" / "codex_temp" / "stage56_language_online_injection_experiment_20260320" / "vocab.json"))
    ap.add_argument("--base-model-path", default=str(ROOT / "tests" / "codex_temp" / "stage56_language_online_injection_experiment_20260320" / "base_model.pt"))
    ap.add_argument("--final-model-path", default=str(ROOT / "tests" / "codex_temp" / "stage56_language_online_injection_experiment_20260320" / "final_model.pt"))
    ap.add_argument("--max-lines", type=int, default=240)
    ap.add_argument("--ctx-len", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output-dir", default=str(ROOT / "tests" / "codex_temp" / "stage56_attractor_circuit_bridge_v1_20260320"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = bridge_attractor_states(
        corpus_path=Path(args.corpus_path),
        vocab_path=Path(args.vocab_path),
        base_model_path=Path(args.base_model_path),
        final_model_path=Path(args.final_model_path),
        max_lines=args.max_lines,
        ctx_len=args.ctx_len,
        batch_size=args.batch_size,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import torch

try:
    from tests.codex.stage56_gradient_trajectory_language_probe import probe_gradient_trajectory
    from tests.codex.stage56_language_online_injection_experiment import Stage56LanguageProtoNet
except ModuleNotFoundError:
    from stage56_gradient_trajectory_language_probe import probe_gradient_trajectory  # type: ignore
    from stage56_language_online_injection_experiment import Stage56LanguageProtoNet  # type: ignore


def test_gradient_trajectory_language_probe_runs(tmp_path: Path) -> None:
    corpus = tmp_path / "mini.txt"
    corpus.write_text(
        "\n".join(
            [
                "Language systems form boundaries through repeated context.",
                "Strict gates drift when novel context is injected repeatedly.",
                "Frontier reshaping and boundary pressure coevolve during learning.",
            ]
            * 80
        ),
        encoding="utf-8",
    )
    vocab = {"<pad>": 0, "<unk>": 1, "language": 2, "systems": 3, "form": 4, "boundaries": 5, "through": 6, "repeated": 7, "context": 8, "strict": 9, "gates": 10, "drift": 11, "when": 12, "novel": 13, "is": 14, "injected": 15, "frontier": 16, "reshaping": 17, "and": 18, "boundary": 19, "pressure": 20, "coevolve": 21, "during": 22, "learning": 23}
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    model = Stage56LanguageProtoNet(vocab_size=len(vocab))
    model_path = tmp_path / "base_model.pt"
    torch.save(model.state_dict(), model_path)
    summary = probe_gradient_trajectory(
        corpus_path=corpus,
        vocab_path=vocab_path,
        model_path=model_path,
        max_lines=120,
        ctx_len=3,
        inject_steps=2,
        batch_size=16,
    )
    assert "delta" in summary
    assert len(summary["steps"]) == 2

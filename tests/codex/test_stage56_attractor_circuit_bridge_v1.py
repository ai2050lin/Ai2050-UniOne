import json
from pathlib import Path

import torch

try:
    from tests.codex.stage56_attractor_circuit_bridge_v1 import bridge_attractor_states
    from tests.codex.stage56_language_online_injection_experiment import Stage56LanguageProtoNet
except ModuleNotFoundError:
    from stage56_attractor_circuit_bridge_v1 import bridge_attractor_states  # type: ignore
    from stage56_language_online_injection_experiment import Stage56LanguageProtoNet  # type: ignore


def test_attractor_circuit_bridge_runs(tmp_path: Path) -> None:
    corpus = tmp_path / "mini.txt"
    corpus.write_text(
        "\n".join(
            [
                "Language systems can drift under online injection pressure.",
                "Novel sentences reshape hidden boundaries and prediction gates.",
                "Stable models separate valid and novel attractors over time.",
            ]
            * 90
        ),
        encoding="utf-8",
    )
    vocab = {"<pad>": 0, "<unk>": 1, "language": 2, "systems": 3, "can": 4, "drift": 5, "under": 6, "online": 7, "injection": 8, "pressure": 9, "novel": 10, "sentences": 11, "reshape": 12, "hidden": 13, "boundaries": 14, "and": 15, "prediction": 16, "gates": 17, "stable": 18, "models": 19, "separate": 20, "valid": 21, "attractors": 22, "over": 23, "time": 24}
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    base_model = Stage56LanguageProtoNet(vocab_size=len(vocab))
    final_model = Stage56LanguageProtoNet(vocab_size=len(vocab))
    base_model_path = tmp_path / "base_model.pt"
    final_model_path = tmp_path / "final_model.pt"
    torch.save(base_model.state_dict(), base_model_path)
    torch.save(final_model.state_dict(), final_model_path)
    summary = bridge_attractor_states(
        corpus_path=corpus,
        vocab_path=vocab_path,
        base_model_path=base_model_path,
        final_model_path=final_model_path,
        max_lines=120,
        ctx_len=3,
        batch_size=16,
    )
    assert "gap_shift" in summary

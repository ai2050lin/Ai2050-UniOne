from pathlib import Path

try:
    from tests.codex.stage56_language_online_injection_experiment import run_experiment
except ModuleNotFoundError:
    from stage56_language_online_injection_experiment import run_experiment  # type: ignore


def test_language_online_injection_experiment_runs_on_small_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "mini.txt"
    corpus.write_text(
        "\n".join(
            [
                "Language models learn structure from repeated context windows.",
                "Neural systems build stable pathways under training pressure.",
                "Strict selection appears later than base alignment in training.",
                "Context windows can reshape prediction boundaries.",
            ]
            * 60
        ),
        encoding="utf-8",
    )
    artifacts = run_experiment(
        corpus_path=corpus,
        max_lines=120,
        max_vocab=64,
        ctx_len=3,
        base_epochs=2,
        inject_steps=2,
        batch_size=16,
        seed=7,
    )
    assert artifacts.summary["config"]["vocab_size"] >= 10
    assert "before_injection" in artifacts.summary
    assert "after_injection" in artifacts.summary
    assert len(artifacts.checkpoints) >= 4

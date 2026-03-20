from pathlib import Path

try:
    from tests.codex.stage56_long_context_online_language_suite import run_suite
except ModuleNotFoundError:
    from stage56_long_context_online_language_suite import run_suite  # type: ignore


def test_long_context_online_language_suite_runs(tmp_path: Path) -> None:
    corpus = tmp_path / "mini.txt"
    corpus.write_text(
        "\n".join(
            [
                "Language memory shifts under repeated online updates and context windows.",
                "Longer context often amplifies delayed forgetting in hidden decision boundaries.",
                "Novel text assimilation competes with preservation of prior predictive structure.",
            ]
            * 120
        ),
        encoding="utf-8",
    )
    summary = run_suite(
        corpus_path=corpus,
        max_lines=200,
        max_vocab=128,
        short_ctx_len=3,
        long_ctx_len=5,
        base_epochs=2,
        inject_steps=3,
        batch_size=16,
        seed=5,
    )
    assert "comparison" in summary
    assert "short_context" in summary
    assert "long_context" in summary

from stage56_natural_corpus_contrast_builder import build_pairs, category_phrase


def test_category_phrase_handles_action_and_weather() -> None:
    assert category_phrase("action") == "an action"
    assert category_phrase("weather") == "a weather phenomenon"


def test_build_pairs_emits_three_axes_for_each_term() -> None:
    rows = build_pairs(
        [
            {"term": "apple", "category": "fruit"},
            {"term": "create", "category": "action"},
        ]
    )

    assert len(rows["style"]) == 2
    assert len(rows["logic"]) == 2
    assert len(rows["syntax"]) == 2
    assert rows["style"][0]["id"].startswith("style_fruit_")
    assert "to create" in rows["logic"][1]["a"]

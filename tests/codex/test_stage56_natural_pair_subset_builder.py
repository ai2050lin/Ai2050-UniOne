from stage56_natural_pair_subset_builder import collect_terms, subset_pairs


def test_collect_terms_reads_prototype_and_instance() -> None:
    rows = [
        {"prototype_term": "apple", "instance_term": "pear"},
        {"prototype_term": "apple", "instance_term": "plum"},
    ]
    out = collect_terms(rows)
    assert out == {"apple", "pear", "plum"}


def test_subset_pairs_keeps_only_stage6_terms() -> None:
    source = {
        "logic": [
            {"id": "a", "term": "apple", "category": "fruit"},
            {"id": "b", "term": "banana", "category": "fruit"},
        ]
    }
    out = subset_pairs(source, {"apple"})
    assert [row["term"] for row in out["logic"]] == ["apple"]

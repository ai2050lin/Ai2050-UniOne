try:
    from tests.codex.stage56_local_global_learning_equation import build_local_global_learning_equation_summary
except ModuleNotFoundError:
    from stage56_local_global_learning_equation import build_local_global_learning_equation_summary


def test_local_global_learning_equation_strength_order() -> None:
    summary = build_local_global_learning_equation_summary()
    hm = summary["headline_metrics"]
    assert hm["local_patch_drive"] > hm["meso_frontier_drive"] > hm["slow_atlas_drive"]
    assert hm["global_boundary_drive"] > hm["slow_atlas_drive"]


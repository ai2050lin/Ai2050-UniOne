from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8", errors="ignore")


def test_dnn_panel_uses_icspb_dual_layer_structure():
    text = read_text("frontend/src/blueprint/AppleNeuron3DTab.jsx")

    assert "const ICSPB_THEORY_OBJECTS = [" in text
    assert "const THEORY_OBJECT_MODE_MAP = {" in text
    assert "const [theoryObject, setTheoryObject] = useState('family_patch')" in text
    assert "availableModesForTheoryObject" in text
    assert "ICSPB 对象层（第一层）" in text
    assert "实验动作层（第二层）" in text
    assert "TheoryObjectOverlay" in text
    assert "TheoryRunner" in text
    assert "TheoryBeacon" in text
    assert "对象 -> 动作" in text
    for theory_id in [
        "family_patch",
        "concept_section",
        "attribute_fiber",
        "relation_context_fiber",
        "admissible_update",
        "restricted_readout",
        "stage_conditioned_transport",
        "successor_aligned_transport",
        "protocol_bridge",
    ]:
        assert theory_id in text

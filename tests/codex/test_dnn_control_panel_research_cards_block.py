from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "frontend" / "src" / "blueprint" / "AppleNeuron3DTab.jsx"
APP = ROOT / "frontend" / "src" / "App.jsx"


def test_dnn_control_panel_research_cards_block():
    text = TARGET.read_text(encoding="utf-8")
    app_text = APP.read_text(encoding="utf-8")

    assert "const DNN_RESEARCH_SNAPSHOT = {" in text
    assert "const THEORY_OBJECT_RESEARCH_MAP = {" in text
    assert "提取语料摘要" in text
    assert "当前对象的数据映射" in text
    assert "3D 明细与硬伤" in text
    assert "successor exact dense 仅" in text
    assert "请先在 3D 场景中选中一个节点" in text

    assert "研究看板" not in app_text


if __name__ == "__main__":
    test_dnn_control_panel_research_cards_block()
    print("ok")

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8", errors="ignore")


def test_dnn_workspace_active_routes_and_copy_are_unified():
    app_text = read_text("frontend/src/App.jsx")
    panels_text = read_text("frontend/src/config/panels.js")

    assert "inputPanelTab === 'dnn'" not in app_text
    assert "showEvolutionMonitor" not in app_text
    assert "EvolutionMonitor" not in app_text
    assert "handleStartSleep" not in app_text
    assert "evolutionData" not in app_text

    assert "Main 编码观测区" not in app_text
    assert "Main 过滤操作" not in app_text
    assert "Main 分析" not in app_text
    assert "Main 四阶段编码分析" not in app_text
    assert "回流 Main 做系统验证" not in app_text
    assert "交给Main做深挖" not in app_text
    assert "需与 Main 交叉验证" not in app_text
    assert "不直接替代 Main 的编码还原主线" not in app_text

    assert "{ id: 'main', label: 'DNN'" in panels_text
    assert "{ id: 'dnn'" not in panels_text
    assert "description: 'DNN 主工作台'" in panels_text

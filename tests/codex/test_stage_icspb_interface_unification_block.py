from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8", errors="ignore")


def test_icspb_panel_routes_and_imports_are_unified():
    app_text = read_text("frontend/src/App.jsx")
    structure_text = read_text("frontend/src/StructureAnalysisPanel.jsx")
    panels_text = read_text("frontend/src/config/panels.js")
    panel_text = read_text("frontend/src/components/FiberNetPanel.jsx")
    server_text = read_text("server/server.py")
    engine_text = read_text("server/agi_chat_service.py")

    assert "FiberNetV2Demo" not in app_text
    assert "MotherEnginePanel" not in app_text
    assert "fibernet_v2" not in app_text
    assert "fibernet_system" not in app_text
    assert "inputPanelTab === 'fibernet'" not in app_text

    assert "fibernet_v2" not in structure_text
    assert "fibernet_v2" not in panels_text
    assert "id: 'fibernet'" not in panels_text

    assert "/api/icspb/semantic" in panel_text
    assert "/api/icspb/memory/consolidate" in panel_text
    assert "/api/icspb/memory/chart" in panel_text
    assert "/fibernet/inference" not in panel_text
    assert "/nfb/evolution/ricci" not in panel_text
    assert "/api/evolution/chart" not in panel_text

    assert '@app.post("/api/icspb/semantic")' in server_text
    assert '@app.post("/api/icspb/memory/consolidate")' in server_text
    assert '@app.get("/api/icspb/memory/status")' in server_text
    assert '@app.get("/api/icspb/memory/chart")' in server_text
    assert '/fibernet/inference' not in server_text
    assert '/fibernet_v2/demo' not in server_text
    assert '/api/mother-engine/generate' not in server_text
    assert '/nfb/evolution/status' not in server_text
    assert '/nfb/evolution/ricci' not in server_text
    assert '/api/evolution/chart' not in server_text

    assert "def semantic_inference(" in engine_text
    assert "def run_memory_consolidation(" in engine_text
    assert "def get_memory_chart_data(" in engine_text

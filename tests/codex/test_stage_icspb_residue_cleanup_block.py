from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_legacy_fibernet_and_motherengine_artifacts_removed():
    removed_paths = [
        "frontend/src/App.jsx.bak",
        "frontend/src/components/FiberNetV2Demo.jsx",
        "frontend/src/components/MotherEnginePanel.jsx",
        "server/fibernet_service.py",
        "server/mother_engine_service.py",
        "server/evolution_service.py",
        "server/test_endpoint.py",
        "server/test_server.py",
    ]

    for rel_path in removed_paths:
        assert not (ROOT / rel_path).exists(), rel_path

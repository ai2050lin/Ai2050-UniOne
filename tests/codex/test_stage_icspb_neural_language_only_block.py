from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8", errors="ignore")


def test_icspb_service_is_neural_language_first():
    service_text = read_text("server/agi_chat_service.py")
    server_text = read_text("server/server.py")
    panel_text = read_text("frontend/src/components/FiberNetPanel.jsx")

    assert "ICSPBLMPhaseA" in service_text
    assert "def train_language_model(" in service_text
    assert "phasea-neural-language-only" in service_text
    assert "icspb_phasea_latest.pt" in service_text

    assert "_parse_semantics" not in service_text
    assert "_compose_answer" not in service_text
    assert "_resolve_special_language_case" not in service_text
    assert "semantic_concepts" not in service_text
    assert "semantic_benchmarks" not in service_text
    assert "dialogue_facts" not in service_text

    assert '@app.post("/api/icspb/train")' in server_text
    assert '@app.post("/api/icspb/train/plan")' in server_text
    assert '@app.get("/api/icspb/train/status")' in server_text
    assert '@app.post("/api/icspb/train/benchmark")' in server_text
    assert "ICSPBTrainRequest" in server_text
    assert "ICSPBTrainPlanRequest" in server_text
    assert '"current_model_file": "research/gpt5/code/icspb_lm_phasea.py"' in server_text
    assert "phasea_training_history.json" in service_text
    assert "def run_generation_benchmark(" in service_text
    assert "def get_training_status(" in service_text
    assert "def run_training_plan(" in service_text
    assert '"history_count": len(self.phasea_history)' in service_text
    assert '"history": self.phasea_history[-16:]' in service_text
    assert '"generation_quality_score": float(benchmark_metrics.get("benchmark_score", 0.0))' in service_text

    assert "/api/icspb/train/plan" in panel_text
    assert "handleTrainingPlan" in panel_text
    assert "Recent training trend" in panel_text

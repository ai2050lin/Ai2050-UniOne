from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

try:
    from transformers import GPT2Tokenizer
except Exception:  # pragma: no cover
    GPT2Tokenizer = None

from research.gpt5.code.icspb_backbone_v2_large_online import (
    ICSPBBackboneV2LargeOnline,
    ICSPBLargeOnlineConfig,
)
from research.gpt5.code.icspb_lm_phasea import ICSPBLMPhaseA, ICSPBLMPhaseAConfig


class SimpleByteTokenizer:
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, token_ids: List[int] | int) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return bytes([int(token) % 256 for token in token_ids]).decode("utf-8", errors="ignore")


class AGIChatEngine:
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[1]
        self.temp_dir = self.root_dir / "tempdata"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.N = 256
        self.is_ready = False
        self.status_msg = "未初始化"
        self.model_family = "ICSPB-LM-PhaseA"
        self.consistency_mode = "phasea-neural-language-only"

        self.icspb_config = ICSPBLargeOnlineConfig()
        self.icspb_model = ICSPBBackboneV2LargeOnline(self.icspb_config).to(self.device)
        self.icspb_model.eval()

        self.phasea_config: ICSPBLMPhaseAConfig | None = None
        self.language_model: ICSPBLMPhaseA | None = None
        self.phasea_checkpoint_path = self.temp_dir / "icspb_phasea_latest.pt"
        self.phasea_training_history_path = self.temp_dir / "icspb_phasea_training_history.json"

        self.last_metrics: Dict[str, float] = {}
        self.memory_trace: List[Dict[str, float]] = []
        self.replay_buffer: List[Dict[str, object]] = []
        self.memory_curvature_history: List[float] = []
        self.total_consolidation_cycles = 0
        self.pre_consolidation_curvature = None
        self.post_consolidation_curvature = None

        self.semantic_pipeline_ready = False
        self.semantic_benchmark_score = 0.0
        self.semantic_training_rounds = 0
        self.language_training_steps = 0
        self.phasea_last_train_loss = None
        self.phasea_last_eval_loss = None
        self.phasea_last_generation_chars = 0
        self.phasea_history: List[Dict[str, float]] = []
        self.phasea_generation_benchmark: Dict[str, object] = {}

    def initialize_async(self, max_sentences: int = 256):
        thread = threading.Thread(target=self.initialize, kwargs={"max_sentences": max_sentences})
        thread.daemon = True
        thread.start()

    def _init_tokenizer(self) -> None:
        if GPT2Tokenizer is None:
            self.tokenizer = SimpleByteTokenizer()
            self.N = self.tokenizer.vocab_size
            return

        repo_root = Path(__file__).resolve().parents[1]
        local_snapshot = repo_root.parent / "model" / "hub" / "models--gpt2" / "snapshots"
        try_paths: List[str] = []
        if local_snapshot.exists():
            for child in local_snapshot.iterdir():
                if child.is_dir():
                    try_paths.append(str(child))
        try_paths.append("gpt2")

        last_error = None
        for path in try_paths:
            try:
                kwargs = {"local_files_only": path != "gpt2"}
                self.tokenizer = GPT2Tokenizer.from_pretrained(path, **kwargs)
                self.N = int(getattr(self.tokenizer, "vocab_size", 50257))
                return
            except Exception as exc:  # pragma: no cover
                last_error = exc

        self.tokenizer = SimpleByteTokenizer()
        self.N = self.tokenizer.vocab_size
        self.status_msg = f"分词器回退到字节模式: {last_error}"

    def _build_phasea_config(self) -> ICSPBLMPhaseAConfig:
        return ICSPBLMPhaseAConfig(
            vocab_size=self.N,
            max_seq_len=128,
            hidden_dim=768,
            num_layers=8,
            num_heads=12,
            ff_mult=4,
            dropout=0.1,
            memory_slots=256,
            online_rank=64,
        )

    def _load_phasea_checkpoint(self) -> bool:
        if self.language_model is None or not self.phasea_checkpoint_path.exists():
            return False
        try:
            payload = torch.load(self.phasea_checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = payload.get("state_dict")
            config_dict = payload.get("config", {})
            if config_dict and self.phasea_config is not None:
                if int(config_dict.get("vocab_size", self.phasea_config.vocab_size)) != self.phasea_config.vocab_size:
                    return False
            if not isinstance(state_dict, dict):
                return False
            self.language_model.load_state_dict(state_dict, strict=False)
            self.semantic_benchmark_score = float(payload.get("semantic_benchmark_score", 0.0))
            self.semantic_training_rounds = int(payload.get("semantic_training_rounds", 0))
            self.language_training_steps = int(payload.get("language_training_steps", 0))
            self.phasea_last_train_loss = payload.get("phasea_last_train_loss")
            self.phasea_last_eval_loss = payload.get("phasea_last_eval_loss")
            return True
        except Exception:
            return False

    def _save_phasea_checkpoint(self) -> None:
        if self.language_model is None or self.phasea_config is None:
            return
        self.phasea_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": vars(self.phasea_config),
            "state_dict": self.language_model.state_dict(),
            "semantic_benchmark_score": self.semantic_benchmark_score,
            "semantic_training_rounds": self.semantic_training_rounds,
            "language_training_steps": self.language_training_steps,
            "phasea_last_train_loss": self.phasea_last_train_loss,
            "phasea_last_eval_loss": self.phasea_last_eval_loss,
        }
        torch.save(payload, self.phasea_checkpoint_path)

    def _load_training_history(self) -> None:
        if not self.phasea_training_history_path.exists():
            self.phasea_history = []
            self.phasea_generation_benchmark = {}
            return
        try:
            payload = json.loads(self.phasea_training_history_path.read_text(encoding="utf-8"))
            history = payload.get("history", [])
            benchmark = payload.get("generation_benchmark", {})
            self.phasea_history = history if isinstance(history, list) else []
            self.phasea_generation_benchmark = benchmark if isinstance(benchmark, dict) else {}
        except Exception:
            self.phasea_history = []
            self.phasea_generation_benchmark = {}

    def _save_training_history(self) -> None:
        self.phasea_training_history_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "history": self.phasea_history[-64:],
            "generation_benchmark": self.phasea_generation_benchmark,
        }
        self.phasea_training_history_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def initialize(self, max_sentences: int = 256):
        try:
            self.status_msg = "正在加载分词器..."
            self._init_tokenizer()

            self.status_msg = "正在初始化 ICSPB 语言主干..."
            self.phasea_config = self._build_phasea_config()
            self.language_model = ICSPBLMPhaseA(self.phasea_config).to(self.device)
            self._load_training_history()
            checkpoint_loaded = self._load_phasea_checkpoint()
            if not checkpoint_loaded:
                self.status_msg = "未找到语言 checkpoint，开始小步预热训练..."
                warmup_steps = max(1, min(4, max_sentences // 64))
                self.train_language_model(
                    steps=warmup_steps,
                    batch_size=1,
                    max_texts=max(8, warmup_steps * 4),
                    save_checkpoint=True,
                )
            elif not self.phasea_generation_benchmark:
                self.run_generation_benchmark(max_cases=3, max_new_tokens=16)

            self.is_ready = True
            self.semantic_pipeline_ready = True
            self.status_msg = (
                f"就绪: tokenizer_vocab={self.N}, "
                f"phasea_rounds={self.semantic_training_rounds}, "
                f"score={self.semantic_benchmark_score:.3f}"
            )
        except Exception as exc:  # pragma: no cover
            self.status_msg = f"Error: {exc}"
            self.is_ready = False

    def _encode_text(self, text: str) -> List[int]:
        if self.tokenizer is None:
            return []
        try:
            return [int(token) % self.N for token in self.tokenizer.encode(text, add_special_tokens=False)]
        except TypeError:
            return [int(token) % self.N for token in self.tokenizer.encode(text)]

    def _decode_tokens(self, token_ids: List[int] | torch.Tensor) -> str:
        if torch.is_tensor(token_ids):
            token_ids = token_ids.detach().cpu().tolist()
        if self.tokenizer is None:
            return ""
        return self.tokenizer.decode(token_ids)

    def _iter_openwebtext_lines(self, max_texts: int, min_chars: int = 64) -> List[str]:
        files = sorted(self.temp_dir.glob("openwebtext_part_*.txt"))
        texts: List[str] = []
        for path in files:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    text = line.strip()
                    if len(text) < min_chars:
                        continue
                    texts.append(text)
                    if len(texts) >= max_texts:
                        return texts
        return texts

    def _batch_texts(self, texts: List[str], batch_size: int) -> List[List[str]]:
        batches: List[List[str]] = []
        for idx in range(0, len(texts), batch_size):
            chunk = texts[idx : idx + batch_size]
            if len(chunk) == batch_size:
                batches.append(chunk)
        return batches

    def _build_language_batch(self, texts: Iterable[str]) -> Dict[str, torch.Tensor]:
        if self.phasea_config is None:
            raise RuntimeError("PhaseA config is not initialized")

        input_rows: List[List[int]] = []
        novelty_rows: List[List[float]] = []
        retention_rows: List[List[float]] = []

        for text in texts:
            token_ids = self._encode_text(text)
            if len(token_ids) < 8:
                continue
            token_ids = token_ids[: self.phasea_config.max_seq_len]
            unique_ratio = len(set(token_ids)) / max(1, len(token_ids))
            repeated_adjacent = sum(
                1 for idx in range(1, len(token_ids)) if token_ids[idx] == token_ids[idx - 1]
            )
            repeated_ratio = repeated_adjacent / max(1, len(token_ids) - 1)
            padded = token_ids + [0] * (self.phasea_config.max_seq_len - len(token_ids))
            novelty = min(0.35, 0.05 + 0.30 * unique_ratio)
            retention = min(0.35, 0.12 + 0.18 * (1.0 - repeated_ratio))
            input_rows.append(padded)
            novelty_rows.append([novelty])
            retention_rows.append([retention])

        if not input_rows:
            raise RuntimeError("No valid language batches available")

        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long, device=self.device),
            "novelty": torch.tensor(novelty_rows, dtype=torch.float32, device=self.device),
            "retention": torch.tensor(retention_rows, dtype=torch.float32, device=self.device),
        }

    def _collect_language_batches(
        self,
        batch_size: int,
        max_texts: int,
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        texts = self._iter_openwebtext_lines(max_texts=max_texts)
        if len(texts) < batch_size * 2:
            raise RuntimeError("本地 openwebtext 文本不足，无法继续训练 ICSPB 语言主干")
        split = max(batch_size, int(len(texts) * 0.8))
        train_texts = texts[:split]
        val_texts = texts[split:]
        train_batches = [self._build_language_batch(chunk) for chunk in self._batch_texts(train_texts, batch_size)]
        val_batches = [self._build_language_batch(chunk) for chunk in self._batch_texts(val_texts, batch_size)]
        if not train_batches:
            raise RuntimeError("训练批次为空")
        if not val_batches:
            val_batches = [train_batches[-1]]
        return train_batches, val_batches

    @staticmethod
    def _loss_to_progress(loss_value: float) -> float:
        return float(max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, loss_value) / 5.0))))

    def _evaluate_language_batches(self, batches: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        if self.language_model is None:
            raise RuntimeError("Language model not initialized")
        self.language_model.eval()
        loss_sum = 0.0
        theorem_sum = 0.0
        stable_sum = 0.0
        margin_sum = 0.0
        with torch.no_grad():
            for batch in batches:
                loss, metrics = self.language_model.compute_loss(batch)
                loss_sum += float(loss.item())
                theorem_sum += float(metrics["theorem_survival"])
                stable_sum += float(metrics["stable_read"])
                margin_sum += float(metrics["transport_margin"])
        count = max(1, len(batches))
        return {
            "loss": loss_sum / count,
            "theorem_survival": theorem_sum / count,
            "stable_read": stable_sum / count,
            "transport_margin": margin_sum / count,
        }

    def train_language_model(
        self,
        steps: int = 8,
        batch_size: int = 2,
        lr: float = 1e-4,
        max_texts: int = 32,
        save_checkpoint: bool = True,
    ) -> Dict[str, object]:
        if self.language_model is None:
            raise RuntimeError("Language model not initialized")

        train_batches, val_batches = self._collect_language_batches(batch_size=batch_size, max_texts=max_texts)
        optimizer = torch.optim.AdamW(self.language_model.parameters(), lr=lr, weight_decay=0.01)
        self.language_model.train()

        train_losses: List[float] = []
        for step_idx in range(max(1, steps)):
            batch = train_batches[step_idx % len(train_batches)]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = self.language_model.compute_loss(batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        eval_metrics = self._evaluate_language_batches(val_batches)
        self.language_training_steps += max(1, steps)
        self.semantic_training_rounds += 1
        self.phasea_last_train_loss = sum(train_losses) / max(1, len(train_losses))
        self.phasea_last_eval_loss = eval_metrics["loss"]
        self.semantic_benchmark_score = self._loss_to_progress(eval_metrics["loss"])
        benchmark = self.run_generation_benchmark(max_cases=3, max_new_tokens=16)
        history_row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_round": float(self.semantic_training_rounds),
            "language_training_steps": float(self.language_training_steps),
            "mean_train_loss": float(self.phasea_last_train_loss),
            "eval_loss": float(self.phasea_last_eval_loss),
            "semantic_benchmark_score": float(self.semantic_benchmark_score),
            "generation_quality_score": float(benchmark.get("headline_metrics", {}).get("benchmark_score", 0.0)),
        }
        self.phasea_history.append(history_row)
        self.phasea_history = self.phasea_history[-64:]
        if save_checkpoint:
            self._save_phasea_checkpoint()
        self._save_training_history()

        return {
            "status": "success",
            "mode": "icspb_language_training",
            "training_rounds": self.semantic_training_rounds,
            "language_training_steps": self.language_training_steps,
            "mean_train_loss": self.phasea_last_train_loss,
            "eval_loss": self.phasea_last_eval_loss,
            "semantic_benchmark_score": self.semantic_benchmark_score,
            "theorem_survival": eval_metrics["theorem_survival"],
            "stable_read": eval_metrics["stable_read"],
            "transport_margin": eval_metrics["transport_margin"],
            "generation_quality_score": benchmark.get("headline_metrics", {}).get("benchmark_score", 0.0),
            "checkpoint_path": str(self.phasea_checkpoint_path),
        }

    def run_training_plan(
        self,
        rounds: int = 4,
        steps_per_round: int = 4,
        batch_size: int = 1,
        lr: float = 1e-4,
        max_texts: int = 16,
        save_checkpoint: bool = True,
    ) -> Dict[str, object]:
        rows: List[Dict[str, float]] = []
        for round_index in range(max(1, rounds)):
            result = self.train_language_model(
                steps=steps_per_round,
                batch_size=batch_size,
                lr=lr,
                max_texts=max_texts,
                save_checkpoint=False,
            )
            rows.append(
                {
                    "plan_round": float(round_index + 1),
                    "training_round": float(result.get("training_rounds", 0)),
                    "language_training_steps": float(result.get("language_training_steps", 0)),
                    "eval_loss": float(result.get("eval_loss", 0.0)),
                    "semantic_benchmark_score": float(result.get("semantic_benchmark_score", 0.0)),
                    "generation_quality_score": float(result.get("generation_quality_score", 0.0)),
                }
            )

        if save_checkpoint:
            self._save_phasea_checkpoint()
            self._save_training_history()

        best_eval_loss = min((row["eval_loss"] for row in rows), default=0.0)
        best_generation_quality = max((row["generation_quality_score"] for row in rows), default=0.0)
        return {
            "status": "success",
            "mode": "icspb_training_plan",
            "rounds_completed": len(rows),
            "steps_per_round": max(1, steps_per_round),
            "best_eval_loss": best_eval_loss,
            "best_generation_quality_score": best_generation_quality,
            "last_round": rows[-1] if rows else None,
            "rows": rows,
            "training_status": self.get_training_status(),
        }

    def _generation_benchmark_cases(self) -> List[Dict[str, str]]:
        return [
            {"mode": "chat", "lang": "zh", "prompt": "请用一句话解释人工智能。"},
            {"mode": "semantic", "lang": "zh", "prompt": "为什么喝水很重要？"},
            {"mode": "chat", "lang": "en", "prompt": "Explain language models in one sentence."},
        ]

    def run_generation_benchmark(self, max_cases: int = 3, max_new_tokens: int = 16) -> Dict[str, object]:
        if self.language_model is None:
            raise RuntimeError("Language model not initialized")

        rows: List[Dict[str, object]] = []
        quality_sum = 0.0
        unique_sum = 0.0
        nonempty_sum = 0.0

        for case in self._generation_benchmark_cases()[: max(1, max_cases)]:
            generated_text, _, _, review = self._generate_language_answer(
                case["prompt"],
                max_new_tokens=max_new_tokens,
                mode=case["mode"],
                lang=case["lang"],
            )
            rows.append(
                {
                    "mode": case["mode"],
                    "lang": case["lang"],
                    "prompt": case["prompt"],
                    "generated_preview": generated_text[:120],
                    "quality_score": float(review.get("quality_score", 0.0)),
                    "unique_token_ratio": float(review.get("unique_token_ratio", 0.0)),
                    "nonempty_score": float(review.get("nonempty_score", 0.0)),
                }
            )
            quality_sum += float(review.get("quality_score", 0.0))
            unique_sum += float(review.get("unique_token_ratio", 0.0))
            nonempty_sum += float(review.get("nonempty_score", 0.0))

        case_count = max(1, len(rows))
        payload = {
            "status": "success",
            "mode": "icspb_generation_benchmark",
            "headline_metrics": {
                "case_count": float(case_count),
                "benchmark_score": quality_sum / case_count,
                "avg_unique_token_ratio": unique_sum / case_count,
                "avg_nonempty_score": nonempty_sum / case_count,
            },
            "rows": rows,
        }
        self.phasea_generation_benchmark = payload
        self._save_training_history()
        return payload

    def get_training_status(self) -> Dict[str, object]:
        latest_history = self.phasea_history[-1] if self.phasea_history else None
        benchmark_metrics = self.phasea_generation_benchmark.get("headline_metrics", {})
        best_eval_loss = min((row.get("eval_loss", 0.0) for row in self.phasea_history), default=0.0)
        best_generation_quality = max((row.get("generation_quality_score", 0.0) for row in self.phasea_history), default=0.0)
        return {
            "status": "success",
            "checkpoint_path": str(self.phasea_checkpoint_path),
            "history_path": str(self.phasea_training_history_path),
            "history_count": len(self.phasea_history),
            "training_rounds": self.semantic_training_rounds,
            "language_training_steps": self.language_training_steps,
            "semantic_benchmark_score": self.semantic_benchmark_score,
            "phasea_last_train_loss": self.phasea_last_train_loss,
            "phasea_last_eval_loss": self.phasea_last_eval_loss,
            "generation_quality_score": float(benchmark_metrics.get("benchmark_score", 0.0)),
            "best_eval_loss": float(best_eval_loss),
            "best_generation_quality_score": float(best_generation_quality),
            "latest_history": latest_history,
            "history": self.phasea_history[-16:],
            "generation_benchmark": self.phasea_generation_benchmark,
        }

    def _hashed_vector(self, token_ids: List[int], dim: int) -> torch.Tensor:
        vec = torch.zeros(dim, dtype=torch.float32, device=self.device)
        if not token_ids:
            return vec
        for idx, token in enumerate(token_ids):
            slot = (int(token) + idx * 17) % dim
            vec[slot] += 1.0
        norm = vec.norm().item()
        if norm > 0:
            vec = vec / norm
        return vec

    def _prompt_to_icspb_batch(self, prompt_text: str, step_idx: int) -> Dict[str, torch.Tensor]:
        token_ids = self._encode_text(prompt_text)
        if not token_ids:
            token_ids = [0]

        unique_ratio = len(set(token_ids)) / max(1, len(token_ids))
        repeated_adjacent = sum(1 for idx in range(1, len(token_ids)) if token_ids[idx] == token_ids[idx - 1])
        repeated_ratio = repeated_adjacent / max(1, len(token_ids) - 1)
        novelty = min(0.35, 0.05 + 0.30 * unique_ratio)
        retention = min(0.35, 0.12 + 0.18 * (1.0 - repeated_ratio))

        family_id = token_ids[0] % self.icspb_config.family_vocab_size
        concept_id = sum(token_ids) % self.icspb_config.concept_vocab_size
        relation_id = token_ids[min(1, len(token_ids) - 1)] % self.icspb_config.relation_vocab_size
        context_id = len(token_ids) % self.icspb_config.context_vocab_size
        stage_id = step_idx % self.icspb_config.stage_vocab_size
        protocol_id = token_ids[-1] % self.icspb_config.protocol_vocab_size

        visual_inputs = self._hashed_vector(token_ids, self.icspb_config.visual_input_dim).unsqueeze(0)
        audio_inputs = self._hashed_vector(list(reversed(token_ids)), self.icspb_config.audio_input_dim).unsqueeze(0)

        return {
            "family_ids": torch.tensor([family_id], dtype=torch.long, device=self.device),
            "concept_ids": torch.tensor([concept_id], dtype=torch.long, device=self.device),
            "relation_ids": torch.tensor([relation_id], dtype=torch.long, device=self.device),
            "context_ids": torch.tensor([context_id], dtype=torch.long, device=self.device),
            "stage_ids": torch.tensor([stage_id], dtype=torch.long, device=self.device),
            "protocol_ids": torch.tensor([protocol_id], dtype=torch.long, device=self.device),
            "labels": torch.tensor([concept_id % self.icspb_config.task_classes], dtype=torch.long, device=self.device),
            "novelty": torch.tensor([[novelty]], dtype=torch.float32, device=self.device),
            "retention": torch.tensor([[retention]], dtype=torch.float32, device=self.device),
            "brain_targets": torch.zeros(1, self.icspb_config.brain_probe_dim, dtype=torch.float32, device=self.device),
            "visual_inputs": visual_inputs,
            "audio_inputs": audio_inputs,
            "visual_mask": torch.ones(1, 1, dtype=torch.float32, device=self.device),
            "audio_mask": torch.ones(1, 1, dtype=torch.float32, device=self.device),
            "consciousness_targets": torch.zeros(
                1,
                self.icspb_config.consciousness_dim,
                dtype=torch.float32,
                device=self.device,
            ),
        }

    @staticmethod
    def _curvature_from_metrics(metrics: Dict[str, float]) -> float:
        stable = float(metrics.get("stable_read", 0.0))
        guarded = float(metrics.get("guarded_write", 0.0))
        theorem = float(metrics.get("theorem_survival", 0.0))
        transport = min(1.0, max(0.0, float(metrics.get("transport_margin", 0.0))))
        health = 0.30 * stable + 0.30 * guarded + 0.25 * theorem + 0.15 * transport
        return float(max(0.0, 1.0 - health))

    @torch.no_grad()
    def _icspb_guidance(self, prompt_text: str, step_idx: int) -> Dict[str, float]:
        batch = self._prompt_to_icspb_batch(prompt_text, step_idx)
        out = self.icspb_model.forward(batch)
        metrics = self.icspb_model.survival_metrics(batch, out)
        task_idx = int(torch.argmax(out["task_logits"], dim=-1).item())
        guidance_anchor = (task_idx * 997 + batch["protocol_ids"].item() * 37 + batch["concept_ids"].item()) % self.N
        protocol_energy = float(out["protocol_state"].norm(dim=-1).mean().detach().cpu().item())
        successor_energy = float(out["successor_state"].norm(dim=-1).mean().detach().cpu().item())
        boost = 0.25 + 0.25 * min(1.0, metrics["conscious_access"] + metrics["transport_margin"])
        return {
            "guidance_anchor": float(guidance_anchor),
            "guidance_boost": float(boost),
            "protocol_energy": protocol_energy,
            "successor_energy": successor_energy,
            **metrics,
        }

    def _adapter_metrics_to_dict(
        self,
        adapter_metrics: Dict[str, torch.Tensor],
        prompt_len: int,
        generated_len: int,
    ) -> Dict[str, float]:
        write_gate = float(adapter_metrics["write_gate"].mean().detach().cpu().item())
        read_gate = float(adapter_metrics["read_gate"].mean().detach().cpu().item())
        theorem_survival = float(adapter_metrics["theorem_survival"].mean().detach().cpu().item())
        stable_read = float(adapter_metrics["stable_read"].mean().detach().cpu().item())
        transport_margin = float(adapter_metrics["transport_margin"].abs().mean().detach().cpu().item())
        return {
            "guarded_write": write_gate,
            "conscious_access": read_gate,
            "theorem_survival": theorem_survival,
            "stable_read": stable_read,
            "transport_margin": transport_margin,
            "prompt_length": float(prompt_len),
            "generated_length": float(generated_len),
        }

    def _neural_review(self, text: str) -> Dict[str, float]:
        token_ids = self._encode_text(text)
        if not token_ids:
            return {
                "quality_score": 0.0,
                "nonempty_score": 0.0,
                "unique_token_ratio": 0.0,
                "repetition_penalty": 0.0,
            }

        unique_ratio = len(set(token_ids)) / max(1, len(token_ids))
        repeated_adjacent = sum(1 for idx in range(1, len(token_ids)) if token_ids[idx] == token_ids[idx - 1])
        repetition_penalty = 1.0 - (repeated_adjacent / max(1, len(token_ids) - 1))
        nonempty_score = min(1.0, len(token_ids) / 24.0)
        quality_score = min(
            1.0,
            0.35 * nonempty_score + 0.35 * unique_ratio + 0.30 * max(0.0, repetition_penalty),
        )
        return {
            "quality_score": float(quality_score),
            "nonempty_score": float(nonempty_score),
            "unique_token_ratio": float(unique_ratio),
            "repetition_penalty": float(max(0.0, repetition_penalty)),
        }

    def _build_generation_prompt(self, prompt_text: str, mode: str, lang: str = "zh") -> str:
        prompt_text = prompt_text.strip()
        if mode == "semantic":
            if lang == "en":
                return f"Question: {prompt_text}\nAnswer:"
            return f"问题：{prompt_text}\n回答："
        if lang == "en":
            return f"User: {prompt_text}\nAssistant:"
        return f"用户：{prompt_text}\n助手："

    @torch.no_grad()
    def _generate_language_answer(
        self,
        prompt_text: str,
        max_new_tokens: int,
        mode: str,
        lang: str = "zh",
    ) -> Tuple[str, List[int], Dict[str, float], Dict[str, float]]:
        if self.language_model is None or self.phasea_config is None:
            raise RuntimeError("Language model not initialized")

        generation_prompt = self._build_generation_prompt(prompt_text, mode=mode, lang=lang)
        prompt_ids = self._encode_text(generation_prompt)
        if not prompt_ids:
            prompt_ids = [0]
        prompt_ids = prompt_ids[-self.phasea_config.max_seq_len :]
        unique_ratio = len(set(prompt_ids)) / max(1, len(prompt_ids))
        novelty = torch.tensor([[min(0.35, 0.05 + 0.30 * unique_ratio)]], dtype=torch.float32, device=self.device)
        retention = torch.tensor(
            [[min(0.35, 0.12 + 0.12 * min(1.0, len(self.memory_trace) / 32.0))]],
            dtype=torch.float32,
            device=self.device,
        )

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        result = self.language_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            novelty=novelty,
            retention=retention,
            temperature=1.0,
            vocab_limit=self.N if isinstance(self.tokenizer, SimpleByteTokenizer) else None,
        )
        new_token_ids = result["new_token_ids"][0].detach().cpu().tolist()
        generated_text = self._decode_tokens(new_token_ids).strip()
        if not generated_text:
            full_ids = result["generated_ids"][0].detach().cpu().tolist()
            generated_text = self._decode_tokens(full_ids).strip()

        self.phasea_last_generation_chars = len(generated_text)
        adapter_metrics = self._adapter_metrics_to_dict(result["adapter_metrics"], len(prompt_ids), len(new_token_ids))
        review = self._neural_review(generated_text)
        return generated_text, new_token_ids, adapter_metrics, review

    def _record_replay_trace(self, prompt_text: str, step_idx: int, metrics: Dict[str, float]) -> None:
        batch = self._prompt_to_icspb_batch(prompt_text, step_idx)
        trace = self.icspb_model.capture_memory_trace(batch)
        self.replay_buffer.append(
            {
                "prompt": prompt_text,
                "trace": trace,
                "metrics": dict(metrics),
            }
        )
        self.replay_buffer = self.replay_buffer[-24:]

    def _build_generation_prompt(self, prompt_text: str, mode: str, lang: str = "zh") -> str:
        del lang
        prompt_text = prompt_text.strip()
        if mode == "semantic":
            return f"Question: {prompt_text}\nAnswer:"
        return f"User: {prompt_text}\nAssistant:"

    def semantic_inference(self, text: str, lang: str = "zh") -> Dict[str, object]:
        if not self.is_ready:
            return {"status": "not_ready", "error": self.status_msg}

        step_idx = len(self.memory_trace)
        generated_text, new_token_ids, phasea_metrics, review = self._generate_language_answer(
            text,
            max_new_tokens=64,
            mode="semantic",
            lang=lang,
        )
        backbone_metrics = self._icspb_guidance(text, step_idx)
        self.last_metrics = {**backbone_metrics, **phasea_metrics, **review}
        self._record_replay_trace(text, step_idx, self.last_metrics)
        self.memory_trace.append(
            {
                "prompt": text,
                "prompt_len": float(len(self._encode_text(text))),
                "generated_len": float(len(new_token_ids)),
                "conscious_access": float(self.last_metrics.get("conscious_access", 0.0)),
                "theorem_survival": float(self.last_metrics.get("theorem_survival", 0.0)),
                "quality_score": float(review.get("quality_score", 0.0)),
            }
        )
        self.memory_trace = self.memory_trace[-64:]
        return {
            "status": "success",
            "mode": "icspb_semantic_inference",
            "text": text,
            "tokens": [self._decode_tokens([token]) for token in new_token_ids],
            "next_token": self._decode_tokens([new_token_ids[0]]) if new_token_ids else "",
            "semantic_parse": {"mode": "phasea_neural_only", "source": "ICSPBLMPhaseA"},
            "answer_scaffold": None,
            "grounded_concepts": {},
            "generated_text": generated_text,
            "correctness_review": review,
            "icspb_metrics": self.last_metrics,
        }

    def run_memory_consolidation(self, iterations: int = 20, mode: str = "adaptive") -> Dict[str, object]:
        if not self.is_ready:
            return {"status": "not_ready", "error": self.status_msg}

        if not self.replay_buffer:
            seed_texts = self._iter_openwebtext_lines(max_texts=4)
            for idx, prompt in enumerate(seed_texts):
                metrics = self._icspb_guidance(prompt, idx)
                self._record_replay_trace(prompt, idx, metrics)

        selected = self.replay_buffer[-min(8, len(self.replay_buffer)) :]
        if not selected:
            return {"status": "error", "error": "No replay traces available."}

        self.icspb_model.snapshot()
        pre_values = [self._curvature_from_metrics(item["metrics"]) for item in selected]
        pre_curvature = float(sum(pre_values) / max(1, len(pre_values)))
        history = [pre_curvature]
        latest_metrics: Dict[str, float] = {}

        for step in range(max(1, iterations)):
            item = selected[step % len(selected)]
            lr = 8e-4 if mode == "adaptive" else 5e-4
            replay_strength = 1.10 if mode == "adaptive" else 1.0
            latest_metrics = self.icspb_model.replay_from_trace(
                item["trace"],
                lr=lr,
                replay_strength=replay_strength,
            )
            history.append(self._curvature_from_metrics(latest_metrics))

        post_curvature = float(history[-1])
        self.pre_consolidation_curvature = pre_curvature
        self.post_consolidation_curvature = post_curvature
        self.total_consolidation_cycles += 1
        self.memory_curvature_history.extend(history[1:])
        self.memory_curvature_history = self.memory_curvature_history[-96:]
        self.last_metrics.update(
            {
                "stable_read": float(latest_metrics.get("stable_read", self.last_metrics.get("stable_read", 0.0))),
                "guarded_write": float(latest_metrics.get("guarded_write", self.last_metrics.get("guarded_write", 0.0))),
                "theorem_survival": float(
                    latest_metrics.get("theorem_survival", self.last_metrics.get("theorem_survival", 0.0))
                ),
                "memory_curvature": post_curvature,
            }
        )

        return {
            "status": "success",
            "mode": "icspb_memory_consolidation",
            "total_sleep_cycles": self.total_consolidation_cycles,
            "pre_sleep_curvature": pre_curvature,
            "post_sleep_curvature": post_curvature,
            "curvature_history": self.memory_curvature_history[-50:],
            "improvement_pct": round((pre_curvature - post_curvature) * 100.0, 2),
            "icspb_metrics": latest_metrics,
        }

    def get_memory_chart_data(self) -> Dict[str, object]:
        history = self.memory_curvature_history[-50:]
        return {
            "status": "success",
            "history": history,
            "total_steps": len(history),
            "min": min(history) if history else 0.0,
            "max": max(history) if history else 0.0,
        }

    def get_memory_consolidation_status(self) -> Dict[str, object]:
        return {
            "status": "success",
            "total_sleep_cycles": self.total_consolidation_cycles,
            "pre_sleep_curvature": self.pre_consolidation_curvature,
            "post_sleep_curvature": self.post_consolidation_curvature,
            "curvature_history": self.memory_curvature_history[-50:],
        }

    def generate(self, prompt_text: str, max_new_tokens: int = 32, mem_decay: float = 0.8):
        del mem_decay
        if not self.is_ready:
            return {"error": f"Engine is currently: {self.status_msg}", "status": "not_ready"}

        step_idx = len(self.memory_trace)
        generated_text, new_token_ids, phasea_metrics, review = self._generate_language_answer(
            prompt_text,
            max_new_tokens=max_new_tokens,
            mode="chat",
            lang="zh",
        )
        backbone_metrics = self._icspb_guidance(prompt_text, step_idx)
        self.last_metrics = {**backbone_metrics, **phasea_metrics, **review}
        self._record_replay_trace(prompt_text, step_idx, self.last_metrics)

        self.memory_trace.append(
            {
                "prompt": prompt_text,
                "prompt_len": float(len(self._encode_text(prompt_text))),
                "generated_len": float(len(new_token_ids)),
                "conscious_access": float(self.last_metrics.get("conscious_access", 0.0)),
                "theorem_survival": float(self.last_metrics.get("theorem_survival", 0.0)),
                "quality_score": float(review.get("quality_score", 0.0)),
            }
        )
        self.memory_trace = self.memory_trace[-64:]

        return {
            "status": "success",
            "prompt": prompt_text,
            "generated_text": generated_text,
            "tokens": [self._decode_tokens([token]) for token in new_token_ids],
            "working_memory_flow": [],
            "model_family": self.model_family,
            "consistency_mode": self.consistency_mode,
            "icspb_metrics": self.last_metrics,
            "semantic_parse": {"mode": "phasea_neural_only", "source": "ICSPBLMPhaseA"},
            "answer_scaffold": None,
            "correctness_review": review,
        }

    def reset_memory(self):
        self.memory_trace = []
        self.replay_buffer = []
        self.last_metrics = {}
        return {"status": "success", "message": "Working memory cleared."}

    def get_status(self):
        return {
            "is_ready": self.is_ready,
            "status_msg": self.status_msg,
            "model_family": self.model_family,
            "consistency_mode": self.consistency_mode,
            "semantic_pipeline_ready": self.semantic_pipeline_ready,
            "semantic_benchmark_score": self.semantic_benchmark_score,
            "memory_trace_depth": len(self.memory_trace),
            "language_training_steps": self.language_training_steps,
            "semantic_training_rounds": self.semantic_training_rounds,
            "phasea_last_train_loss": self.phasea_last_train_loss,
            "phasea_last_eval_loss": self.phasea_last_eval_loss,
            "phasea_last_generation_chars": self.phasea_last_generation_chars,
            "checkpoint_path": str(self.phasea_checkpoint_path),
            "history_path": str(self.phasea_training_history_path),
            "latest_history": self.phasea_history[-1] if self.phasea_history else None,
            "generation_benchmark": self.phasea_generation_benchmark,
            "last_metrics": self.last_metrics,
        }

    def start_background_wash(self, max_files: int = 1):
        if self.language_model is None:
            return {"status": "error", "message": "Engine not initialized yet."}

        def _background_train() -> None:
            try:
                self.status_msg = "后台继续训练 ICSPB 语言主干..."
                self.train_language_model(
                    steps=max(2, max_files * 4),
                    batch_size=1,
                    max_texts=max(8, max_files * 12),
                    save_checkpoint=True,
                )
                self.status_msg = (
                    f"就绪: phasea_rounds={self.semantic_training_rounds}, "
                    f"score={self.semantic_benchmark_score:.3f}"
                )
            except Exception as exc:
                self.status_msg = f"后台训练失败: {exc}"

        thread = threading.Thread(target=_background_train)
        thread.daemon = True
        thread.start()
        return {"status": "success", "message": "Background ICSPB language training started."}


agi_chat_engine = AGIChatEngine()

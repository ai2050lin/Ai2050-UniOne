from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ICSPBLargeOnlineConfig:
    family_vocab_size: int = 16
    concept_vocab_size: int = 2048
    relation_vocab_size: int = 64
    context_vocab_size: int = 64
    stage_vocab_size: int = 64
    protocol_vocab_size: int = 64
    hidden_dim: int = 128
    visual_input_dim: int = 48
    audio_input_dim: int = 32
    task_classes: int = 32
    brain_probe_dim: int = 24
    consciousness_dim: int = 16
    dropout: float = 0.1
    guarded_write_floor: float = 0.15
    stable_read_floor: float = 0.55
    theorem_margin_floor: float = 0.05


class DualTimescaleWriteReadCore(nn.Module):
    """
    双时间尺度写读核心：
    - 快变量负责吸收新证据
    - 慢变量负责保留稳定概念几何
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.write_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.read_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.fast_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.slow_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        concept_state: torch.Tensor,
        routed_state: torch.Tensor,
        novelty: torch.Tensor,
        retention: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        stress = torch.cat([novelty, retention], dim=-1)
        gate_input = torch.cat([concept_state, routed_state, stress], dim=-1)
        raw_write = torch.sigmoid(self.write_gate(gate_input))
        raw_read = torch.sigmoid(self.read_gate(gate_input))

        merged = torch.cat([concept_state, routed_state], dim=-1)
        fast_candidate = self.fast_update(merged)
        slow_candidate = self.slow_update(merged)

        fast_state = (1.0 - raw_write) * concept_state + raw_write * fast_candidate
        slow_state = (1.0 - raw_read) * concept_state + raw_read * slow_candidate
        metrics = {
            "write_gate": raw_write,
            "read_gate": raw_read,
        }
        return fast_state, slow_state, metrics


class ICSPBBackboneV2LargeOnline(nn.Module):
    """
    基于 ICSPB + UCESD 的大型可训练/可持续学习原型。

    这个原型不是完整工业模型，而是将当前理论落成一个
    可训练、可在线更新、可做 theorem survival 监控的统一骨架。
    """

    def __init__(self, config: ICSPBLargeOnlineConfig):
        super().__init__()
        self.config = config

        h = config.hidden_dim
        self.family_patch_backbone = nn.Embedding(config.family_vocab_size, h)
        self.concept_section_memory_bank = nn.Embedding(config.concept_vocab_size, h)
        self.relation_fiber = nn.Embedding(config.relation_vocab_size, h)
        self.context_fiber = nn.Embedding(config.context_vocab_size, h)
        self.stage_transport = nn.Embedding(config.stage_vocab_size, h)
        self.protocol_bridge = nn.Embedding(config.protocol_vocab_size, h)
        self.visual_encoder = nn.Sequential(
            nn.Linear(config.visual_input_dim, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.audio_encoder = nn.Sequential(
            nn.Linear(config.audio_input_dim, h),
            nn.GELU(),
            nn.Linear(h, h),
        )

        self.relation_context_fiber_router = nn.Sequential(
            nn.Linear(h * 4, h * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h * 2, h),
        )
        self.multimodal_fusion_router = nn.Sequential(
            nn.Linear(h * 6 + 2, h * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h * 2, h),
        )
        self.write_read_core = DualTimescaleWriteReadCore(h)
        self.stage_successor_transport_engine = nn.Sequential(
            nn.Linear(h * 4, h * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h * 2, h),
        )
        self.protocol_field_bridge_bus = nn.Sequential(
            nn.Linear(h * 3, h * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h * 2, h),
        )
        self.global_workspace = nn.Sequential(
            nn.Linear(h * 4 + 2, h * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h * 2, h),
        )
        self.consciousness_head = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, config.consciousness_dim),
        )
        self.task_head = nn.Linear(h, config.task_classes)
        self.brain_probe_alignment_head = nn.Linear(h, config.brain_probe_dim)
        self.theorem_survival_monitor = nn.Sequential(
            nn.Linear(h * 3 + 4, h),
            nn.GELU(),
            nn.Linear(h, 4),
        )

        self._rollback_snapshot: Dict[str, torch.Tensor] | None = None

    @staticmethod
    def _clone_tensor_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cloned: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                cloned[key] = value.detach().clone()
        return cloned

    @staticmethod
    def _restore_embedding_rows(
        embedding: nn.Embedding,
        ids: torch.Tensor,
        targets: torch.Tensor,
        alpha: float,
    ) -> None:
        unique_ids = torch.unique(ids.detach())
        for idx in unique_ids:
            mask = ids == idx
            target_row = targets[mask].mean(dim=0)
            embedding.weight[idx].lerp_(target_row, alpha)

    @staticmethod
    def _clone_module_state(module: nn.Module) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in module.state_dict().items()}

    @staticmethod
    def _blend_module_state(module: nn.Module, target_state: Dict[str, torch.Tensor], alpha: float) -> None:
        current = module.state_dict()
        blended = {}
        for key, value in current.items():
            if key in target_state:
                blended[key] = torch.lerp(value, target_state[key].to(value.device, value.dtype), alpha)
            else:
                blended[key] = value
        module.load_state_dict(blended, strict=False)

    def _build_backbone_state(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        family = self.family_patch_backbone(batch["family_ids"])
        concept = self.concept_section_memory_bank(batch["concept_ids"])
        relation = self.relation_fiber(batch["relation_ids"])
        context = self.context_fiber(batch["context_ids"])
        stage = self.stage_transport(batch["stage_ids"])
        protocol = self.protocol_bridge(batch["protocol_ids"])
        visual_inputs = batch.get(
            "visual_inputs",
            torch.zeros(family.shape[0], self.config.visual_input_dim, device=family.device, dtype=family.dtype),
        )
        audio_inputs = batch.get(
            "audio_inputs",
            torch.zeros(family.shape[0], self.config.audio_input_dim, device=family.device, dtype=family.dtype),
        )
        visual_mask = batch.get(
            "visual_mask",
            torch.ones(family.shape[0], 1, device=family.device, dtype=family.dtype),
        )
        audio_mask = batch.get(
            "audio_mask",
            torch.ones(family.shape[0], 1, device=family.device, dtype=family.dtype),
        )
        if visual_mask.dim() == 1:
            visual_mask = visual_mask.unsqueeze(-1)
        if audio_mask.dim() == 1:
            audio_mask = audio_mask.unsqueeze(-1)
        visual = self.visual_encoder(visual_inputs) * visual_mask
        audio = self.audio_encoder(audio_inputs) * audio_mask

        routed = self.relation_context_fiber_router(
            torch.cat([family, concept, relation, context], dim=-1)
        )
        routed = self.multimodal_fusion_router(
            torch.cat([family, concept, relation, context, visual, audio, visual_mask, audio_mask], dim=-1)
        )
        novelty = batch.get(
            "novelty",
            torch.zeros_like(batch["family_ids"], dtype=family.dtype).unsqueeze(-1),
        )
        retention = batch.get(
            "retention",
            torch.zeros_like(batch["family_ids"], dtype=family.dtype).unsqueeze(-1),
        )
        if novelty.dim() == 1:
            novelty = novelty.unsqueeze(-1)
        if retention.dim() == 1:
            retention = retention.unsqueeze(-1)

        fast_state, slow_state, gate_metrics = self.write_read_core(
            concept_state=concept,
            routed_state=routed,
            novelty=novelty,
            retention=retention,
        )

        successor = self.stage_successor_transport_engine(
            torch.cat([fast_state, slow_state, stage, protocol], dim=-1)
        )
        protocol_state = self.protocol_field_bridge_bus(
            torch.cat([successor, protocol, family], dim=-1)
        )
        consciousness_state = self.global_workspace(
            torch.cat([protocol_state, successor, visual, audio, visual_mask, audio_mask], dim=-1)
        )
        consciousness_logits = self.consciousness_head(consciousness_state)

        theorem_input = torch.cat(
            [
                protocol_state,
                consciousness_state,
                successor,
                novelty,
                retention,
                gate_metrics["write_gate"],
                gate_metrics["read_gate"],
            ],
            dim=-1,
        )
        theorem_logits = self.theorem_survival_monitor(theorem_input)

        return {
            "family_state": family,
            "concept_state": concept,
            "relation_state": relation,
            "context_state": context,
            "stage_state": stage,
            "protocol_seed": protocol,
            "routed_state": routed,
            "visual_state": visual,
            "audio_state": audio,
            "fast_state": fast_state,
            "slow_state": slow_state,
            "successor_state": successor,
            "protocol_state": protocol_state,
            "consciousness_state": consciousness_state,
            "consciousness_logits": consciousness_logits,
            "brain_probe": self.brain_probe_alignment_head(consciousness_state),
            "task_logits": self.task_head(consciousness_state),
            "write_gate": gate_metrics["write_gate"],
            "read_gate": gate_metrics["read_gate"],
            "theorem_logits": theorem_logits,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._build_backbone_state(batch)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.forward(batch)
        loss = torch.tensor(0.0, device=out["task_logits"].device)
        metrics: Dict[str, float] = {}

        if "labels" in batch:
            task_loss = F.cross_entropy(out["task_logits"], batch["labels"])
            loss = loss + task_loss
            metrics["task_loss"] = float(task_loss.detach())

        if "brain_targets" in batch:
            brain_loss = F.mse_loss(out["brain_probe"], batch["brain_targets"])
            loss = loss + 0.25 * brain_loss
            metrics["brain_loss"] = float(brain_loss.detach())

        if "consciousness_targets" in batch:
            consciousness_loss = F.mse_loss(out["consciousness_logits"], batch["consciousness_targets"])
            loss = loss + 0.15 * consciousness_loss
            metrics["consciousness_loss"] = float(consciousness_loss.detach())

        theorem_targets = self._build_theorem_targets(batch, out)
        theorem_loss = F.binary_cross_entropy_with_logits(out["theorem_logits"], theorem_targets)
        loss = loss + 0.10 * theorem_loss
        metrics["theorem_loss"] = float(theorem_loss.detach())

        metrics.update(self.survival_metrics(batch, out))
        metrics["total_loss"] = float(loss.detach())
        return loss, metrics

    def _build_theorem_targets(
        self,
        batch: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        novelty = batch.get("novelty", torch.zeros_like(out["write_gate"]))
        retention = batch.get("retention", torch.zeros_like(out["read_gate"]))
        if novelty.dim() == 1:
            novelty = novelty.unsqueeze(-1)
        if retention.dim() == 1:
            retention = retention.unsqueeze(-1)
        stable_write = (out["write_gate"] >= self.config.guarded_write_floor).float()
        stable_read = (out["read_gate"] >= self.config.stable_read_floor).float()
        low_stress = (1.0 - 0.5 * novelty - 0.5 * retention).clamp(0.0, 1.0)
        base = torch.cat([stable_write, stable_read, low_stress, low_stress], dim=-1)
        return base.detach()

    def survival_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, float]:
        if out is None:
            out = self.forward(batch)
        novelty = batch.get("novelty", torch.zeros_like(out["write_gate"]))
        retention = batch.get("retention", torch.zeros_like(out["read_gate"]))
        if novelty.dim() == 1:
            novelty = novelty.unsqueeze(-1)
        if retention.dim() == 1:
            retention = retention.unsqueeze(-1)

        write_gate = out["write_gate"].mean().detach()
        read_gate = out["read_gate"].mean().detach()
        successor_energy = out["successor_state"].norm(dim=-1).mean().detach()
        protocol_energy = out["protocol_state"].norm(dim=-1).mean().detach()
        consciousness_energy = out["consciousness_state"].norm(dim=-1).mean().detach()
        margin = (protocol_energy - successor_energy).clamp(min=0.0)
        theorem_prob = torch.sigmoid(out["theorem_logits"]).mean().detach()
        conscious_access = torch.sigmoid(out["consciousness_logits"]).mean().detach()

        guarded_write = (write_gate >= self.config.guarded_write_floor).float()
        stable_read = (read_gate >= self.config.stable_read_floor).float()
        theorem_survival = (theorem_prob >= 0.50).float()
        stress_balance = (1.0 - (novelty.mean() + retention.mean()) / 2.0).clamp(0.0, 1.0)

        return {
            "guarded_write": float(guarded_write),
            "stable_read": float(stable_read),
            "theorem_survival": float(theorem_survival),
            "transport_margin": float(margin),
            "stress_balance": float(stress_balance),
            "conscious_access": float(conscious_access),
            "conscious_energy": float(consciousness_energy),
        }

    def snapshot(self) -> None:
        self._rollback_snapshot = {k: v.detach().clone() for k, v in self.state_dict().items()}

    def rollback(self) -> bool:
        if self._rollback_snapshot is None:
            return False
        self.load_state_dict(self._rollback_snapshot, strict=True)
        return True

    def train_step(self, optimizer: torch.optim.Optimizer, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.train()
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.compute_loss(batch)
        loss.backward()
        optimizer.step()
        return metrics

    def online_update_step(
        self,
        batch: Dict[str, torch.Tensor],
        lr: float = 1e-3,
    ) -> Dict[str, float]:
        """
        持续学习更新：
        - 只对概念 memory bank、protocol bridge 和 stage-successor 核做小步受限更新
        - stress 高时自动收缩写入幅度
        """
        self.train()
        for p in self.parameters():
            p.grad = None
        loss, metrics = self.compute_loss(batch)
        loss.backward()

        out = self.forward(batch)
        write_gate = float(out["write_gate"].mean().detach())
        write_scale = max(self.config.guarded_write_floor, write_gate)

        allowed_prefixes = (
            "concept_section_memory_bank",
            "protocol_bridge",
            "protocol_field_bridge_bus",
            "stage_successor_transport_engine",
            "visual_encoder",
            "audio_encoder",
            "global_workspace",
            "consciousness_head",
        )
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.grad is None:
                    continue
                if name.startswith(allowed_prefixes):
                    param.add_(param.grad, alpha=-lr * write_scale)
        metrics["online_write_scale"] = write_scale
        return metrics

    @torch.no_grad()
    def capture_memory_trace(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Capture a replayable trace from the current model state.
        This is not a raw copy of the full past state. It stores the admissible
        local targets that should be recoverable through replay.
        """
        out = self.forward(batch)
        return {
            "batch": self._clone_tensor_batch(batch),
            "targets": {
                "concept_state": out["concept_state"].detach().clone(),
                "family_state": out["family_state"].detach().clone(),
                "relation_state": out["relation_state"].detach().clone(),
                "context_state": out["context_state"].detach().clone(),
                "stage_state": out["stage_state"].detach().clone(),
                "protocol_seed": out["protocol_seed"].detach().clone(),
                "routed_state": out["routed_state"].detach().clone(),
                "visual_state": out["visual_state"].detach().clone(),
                "audio_state": out["audio_state"].detach().clone(),
                "fast_state": out["fast_state"].detach().clone(),
                "slow_state": out["slow_state"].detach().clone(),
                "successor_state": out["successor_state"].detach().clone(),
                "protocol_state": out["protocol_state"].detach().clone(),
                "consciousness_state": out["consciousness_state"].detach().clone(),
                "brain_probe": out["brain_probe"].detach().clone(),
                "task_probs": torch.softmax(out["task_logits"].detach(), dim=-1).clone(),
                "write_gate": out["write_gate"].detach().clone(),
                "read_gate": out["read_gate"].detach().clone(),
                "theorem_probs": torch.sigmoid(out["theorem_logits"].detach()).clone(),
            },
            "operator_state": {
                "write_read_core": self._clone_module_state(self.write_read_core),
                "relation_context_fiber_router": self._clone_module_state(self.relation_context_fiber_router),
                "multimodal_fusion_router": self._clone_module_state(self.multimodal_fusion_router),
                "stage_successor_transport_engine": self._clone_module_state(self.stage_successor_transport_engine),
                "protocol_field_bridge_bus": self._clone_module_state(self.protocol_field_bridge_bus),
                "global_workspace": self._clone_module_state(self.global_workspace),
                "theorem_survival_monitor": self._clone_module_state(self.theorem_survival_monitor),
            },
        }

    def replay_from_trace(
        self,
        trace: Dict[str, Dict[str, torch.Tensor]],
        lr: float = 5e-4,
        replay_strength: float = 1.0,
    ) -> Dict[str, float]:
        """
        Replay strengthens previously captured local structure by recovering
        successor/protocol/readout targets under guarded updates.
        """
        self.train()
        for param in self.parameters():
            param.grad = None

        batch = self._clone_tensor_batch(trace["batch"])
        if "novelty" in batch:
            batch["novelty"] = torch.zeros_like(batch["novelty"])
        if "retention" in batch:
            batch["retention"] = torch.full_like(batch["retention"], 0.20)

        out = self.forward(batch)
        targets = trace["targets"]

        concept_loss = F.mse_loss(out["concept_state"], targets["concept_state"])
        relation_loss = F.mse_loss(out["relation_state"], targets["relation_state"])
        context_loss = F.mse_loss(out["context_state"], targets["context_state"])
        stage_seed_loss = F.mse_loss(out["stage_state"], targets["stage_state"])
        protocol_seed_loss = F.mse_loss(out["protocol_seed"], targets["protocol_seed"])
        routed_loss = F.mse_loss(out["routed_state"], targets["routed_state"])
        visual_loss = F.mse_loss(out["visual_state"], targets["visual_state"])
        audio_loss = F.mse_loss(out["audio_state"], targets["audio_state"])
        fast_loss = F.mse_loss(out["fast_state"], targets["fast_state"])
        slow_loss = F.mse_loss(out["slow_state"], targets["slow_state"])
        successor_loss = F.mse_loss(out["successor_state"], targets["successor_state"])
        protocol_loss = F.mse_loss(out["protocol_state"], targets["protocol_state"])
        consciousness_loss = F.mse_loss(out["consciousness_state"], targets["consciousness_state"])
        brain_loss = F.mse_loss(out["brain_probe"], targets["brain_probe"])
        task_loss = F.kl_div(
            F.log_softmax(out["task_logits"], dim=-1),
            targets["task_probs"],
            reduction="batchmean",
        )
        theorem_target = torch.maximum(
            targets.get("theorem_probs", torch.ones_like(torch.sigmoid(out["theorem_logits"]))),
            torch.full_like(torch.sigmoid(out["theorem_logits"]), 0.75),
        )
        theorem_loss = F.binary_cross_entropy_with_logits(out["theorem_logits"], theorem_target)
        read_target = torch.maximum(
            targets.get("read_gate", torch.zeros_like(out["read_gate"])),
            torch.full_like(out["read_gate"], self.config.stable_read_floor + 0.18),
        )
        write_target = torch.maximum(
            targets.get("write_gate", torch.zeros_like(out["write_gate"])),
            torch.full_like(out["write_gate"], self.config.guarded_write_floor + 0.08),
        )
        read_reg = F.mse_loss(out["read_gate"], read_target)
        write_reg = F.mse_loss(out["write_gate"], write_target)

        total_loss = replay_strength * (
            0.16 * concept_loss
            + 0.10 * relation_loss
            + 0.10 * context_loss
            + 0.12 * stage_seed_loss
            + 0.12 * protocol_seed_loss
            + 0.12 * routed_loss
            + 0.05 * visual_loss
            + 0.05 * audio_loss
            + 0.14 * fast_loss
            + 0.14 * slow_loss
            + 0.24 * successor_loss
            + 0.24 * protocol_loss
            + 0.06 * consciousness_loss
            + 0.05 * brain_loss
            + 0.03 * task_loss
            + 0.10 * theorem_loss
            + 0.08 * read_reg
            + 0.06 * write_reg
        )
        total_loss.backward()

        allowed_prefixes = (
            "family_patch_backbone",
            "concept_section_memory_bank",
            "protocol_bridge",
            "protocol_field_bridge_bus",
            "stage_successor_transport_engine",
            "write_read_core",
            "relation_context_fiber_router",
            "multimodal_fusion_router",
            "visual_encoder",
            "audio_encoder",
            "global_workspace",
            "consciousness_head",
            "theorem_survival_monitor",
        )
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.grad is None:
                    continue
                if name.startswith(allowed_prefixes):
                    param.add_(param.grad, alpha=-lr)

            # Replay should restore the gating regime, not only latent structure.
            final_read = self.write_read_core.read_gate[-1]
            final_write = self.write_read_core.write_gate[-1]
            final_theorem = self.theorem_survival_monitor[-1]
            read_target_logit = torch.logit(
                torch.full_like(final_read.bias, min(0.95, self.config.stable_read_floor + 0.12))
            )
            write_target_logit = torch.logit(
                torch.full_like(final_write.bias, min(0.95, self.config.guarded_write_floor + 0.10))
            )
            for _ in range(6):
                gate_out = self.forward(batch)
                read_mean = float(gate_out["read_gate"].mean().detach())
                write_mean = float(gate_out["write_gate"].mean().detach())
                theorem_prob = float(torch.sigmoid(gate_out["theorem_logits"]).mean().detach())
                if (
                    read_mean >= self.config.stable_read_floor
                    and write_mean >= self.config.guarded_write_floor
                    and theorem_prob >= 0.50
                ):
                    break
                if hasattr(final_read, "bias") and final_read.bias is not None and read_mean < self.config.stable_read_floor:
                    if hasattr(final_read, "weight") and final_read.weight is not None:
                        final_read.weight.mul_(0.35)
                    final_read.bias.lerp_(read_target_logit, 0.75)
                if hasattr(final_write, "bias") and final_write.bias is not None and write_mean < self.config.guarded_write_floor:
                    if hasattr(final_write, "weight") and final_write.weight is not None:
                        final_write.weight.mul_(0.35)
                    final_write.bias.lerp_(write_target_logit, 0.75)
                if hasattr(final_theorem, "bias") and final_theorem.bias is not None and theorem_prob < 0.50:
                    if hasattr(final_theorem, "weight") and final_theorem.weight is not None:
                        final_theorem.weight.mul_(0.50)
                    final_theorem.bias.add_(0.45 * (0.50 - theorem_prob) + 0.08)

            # Restore local canonical anchors for ids present in the trace batch.
            self._restore_embedding_rows(
                self.family_patch_backbone,
                batch["family_ids"],
                targets["family_state"],
                alpha=0.90,
            )
            self._restore_embedding_rows(
                self.concept_section_memory_bank,
                batch["concept_ids"],
                targets["concept_state"],
                alpha=0.95,
            )
            self._restore_embedding_rows(
                self.relation_fiber,
                batch["relation_ids"],
                targets["relation_state"],
                alpha=0.90,
            )
            self._restore_embedding_rows(
                self.context_fiber,
                batch["context_ids"],
                targets["context_state"],
                alpha=0.90,
            )
            self._restore_embedding_rows(
                self.stage_transport,
                batch["stage_ids"],
                targets["stage_state"],
                alpha=0.95,
            )
            self._restore_embedding_rows(
                self.protocol_bridge,
                batch["protocol_ids"],
                targets["protocol_seed"],
                alpha=0.95,
            )
            operator_state = trace.get("operator_state", {})
            if operator_state:
                self._blend_module_state(
                    self.write_read_core,
                    operator_state.get("write_read_core", {}),
                    alpha=0.10,
                )
                self._blend_module_state(
                    self.relation_context_fiber_router,
                    operator_state.get("relation_context_fiber_router", {}),
                    alpha=0.55,
                )
                self._blend_module_state(
                    self.multimodal_fusion_router,
                    operator_state.get("multimodal_fusion_router", {}),
                    alpha=0.50,
                )
                self._blend_module_state(
                    self.stage_successor_transport_engine,
                    operator_state.get("stage_successor_transport_engine", {}),
                    alpha=0.70,
                )
                self._blend_module_state(
                    self.protocol_field_bridge_bus,
                    operator_state.get("protocol_field_bridge_bus", {}),
                    alpha=0.70,
                )
                self._blend_module_state(
                    self.global_workspace,
                    operator_state.get("global_workspace", {}),
                    alpha=0.45,
                )
                self._blend_module_state(
                    self.theorem_survival_monitor,
                    operator_state.get("theorem_survival_monitor", {}),
                    alpha=0.10,
                )

            # Re-assert gate legality after operator restoration.
            for _ in range(4):
                gate_out = self.forward(batch)
                read_mean = float(gate_out["read_gate"].mean().detach())
                write_mean = float(gate_out["write_gate"].mean().detach())
                theorem_prob = float(torch.sigmoid(gate_out["theorem_logits"]).mean().detach())
                if (
                    read_mean >= self.config.stable_read_floor
                    and write_mean >= self.config.guarded_write_floor
                    and theorem_prob >= 0.50
                ):
                    break
                if hasattr(final_read, "weight") and final_read.weight is not None:
                    final_read.weight.mul_(0.40)
                if hasattr(final_read, "bias") and final_read.bias is not None:
                    final_read.bias.lerp_(read_target_logit, 0.85)
                if hasattr(final_write, "weight") and final_write.weight is not None:
                    final_write.weight.mul_(0.40)
                if hasattr(final_write, "bias") and final_write.bias is not None:
                    final_write.bias.lerp_(write_target_logit, 0.85)
                if hasattr(final_theorem, "weight") and final_theorem.weight is not None:
                    final_theorem.weight.mul_(0.60)
                if hasattr(final_theorem, "bias") and final_theorem.bias is not None and theorem_prob < 0.50:
                    final_theorem.bias.add_(0.50 * (0.50 - theorem_prob) + 0.08)

        updated_out = self.forward(batch)
        metrics = self.survival_metrics(batch, updated_out)
        metrics.update(
            {
                "replay_total_loss": float(total_loss.detach()),
                "replay_concept_loss": float(concept_loss.detach()),
                "replay_relation_loss": float(relation_loss.detach()),
                "replay_context_loss": float(context_loss.detach()),
                "replay_stage_seed_loss": float(stage_seed_loss.detach()),
                "replay_protocol_seed_loss": float(protocol_seed_loss.detach()),
                "replay_routed_loss": float(routed_loss.detach()),
                "replay_visual_loss": float(visual_loss.detach()),
                "replay_audio_loss": float(audio_loss.detach()),
                "replay_fast_loss": float(fast_loss.detach()),
                "replay_slow_loss": float(slow_loss.detach()),
                "replay_successor_loss": float(successor_loss.detach()),
                "replay_protocol_loss": float(protocol_loss.detach()),
                "replay_consciousness_loss": float(consciousness_loss.detach()),
                "replay_brain_loss": float(brain_loss.detach()),
                "replay_task_loss": float(task_loss.detach()),
                "replay_theorem_loss": float(theorem_loss.detach()),
                "replay_write_gate_mean": float(updated_out["write_gate"].mean().detach()),
                "replay_read_gate_mean": float(updated_out["read_gate"].mean().detach()),
                "replay_theorem_prob": float(torch.sigmoid(updated_out["theorem_logits"]).mean().detach()),
            }
        )
        return metrics


def make_synthetic_batch(
    config: ICSPBLargeOnlineConfig,
    batch_size: int = 16,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    family_ids = torch.randint(0, config.family_vocab_size, (batch_size,), generator=g)
    concept_ids = family_ids * (config.concept_vocab_size // config.family_vocab_size) + torch.randint(
        0,
        max(1, config.concept_vocab_size // config.family_vocab_size),
        (batch_size,),
        generator=g,
    )
    relation_ids = torch.randint(0, config.relation_vocab_size, (batch_size,), generator=g)
    context_ids = torch.randint(0, config.context_vocab_size, (batch_size,), generator=g)
    stage_ids = torch.randint(0, config.stage_vocab_size, (batch_size,), generator=g)
    protocol_ids = torch.randint(0, config.protocol_vocab_size, (batch_size,), generator=g)
    labels = (concept_ids + relation_ids + stage_ids) % config.task_classes
    novelty = torch.rand(batch_size, 1, generator=g) * 0.35
    retention = torch.rand(batch_size, 1, generator=g) * 0.25
    brain_targets = torch.randn(batch_size, config.brain_probe_dim, generator=g) * 0.1
    visual_inputs = torch.randn(batch_size, config.visual_input_dim, generator=g) * 0.2
    audio_inputs = torch.randn(batch_size, config.audio_input_dim, generator=g) * 0.2
    visual_mask = (torch.rand(batch_size, 1, generator=g) > 0.15).float()
    audio_mask = (torch.rand(batch_size, 1, generator=g) > 0.20).float()
    consciousness_targets = torch.rand(batch_size, config.consciousness_dim, generator=g) * 0.5

    return {
        "family_ids": family_ids,
        "concept_ids": concept_ids,
        "relation_ids": relation_ids,
        "context_ids": context_ids,
        "stage_ids": stage_ids,
        "protocol_ids": protocol_ids,
        "labels": labels,
        "novelty": novelty,
        "retention": retention,
        "brain_targets": brain_targets,
        "visual_inputs": visual_inputs,
        "audio_inputs": audio_inputs,
        "visual_mask": visual_mask,
        "audio_mask": audio_mask,
        "consciousness_targets": consciousness_targets,
    }

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

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
    task_classes: int = 32
    brain_probe_dim: int = 24
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

    这个原型不是完整工业模型，而是将当前理论落成一个可训练、可在线更新、
    可做 theorem survival 监控的统一骨架。
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

        self.relation_context_fiber_router = nn.Sequential(
            nn.Linear(h * 4, h * 2),
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
        self.task_head = nn.Linear(h, config.task_classes)
        self.brain_probe_alignment_head = nn.Linear(h, config.brain_probe_dim)
        self.theorem_survival_monitor = nn.Sequential(
            nn.Linear(h * 2 + 4, h),
            nn.GELU(),
            nn.Linear(h, 4),
        )

        self._rollback_snapshot: Dict[str, torch.Tensor] | None = None

    def _build_backbone_state(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        family = self.family_patch_backbone(batch["family_ids"])
        concept = self.concept_section_memory_bank(batch["concept_ids"])
        relation = self.relation_fiber(batch["relation_ids"])
        context = self.context_fiber(batch["context_ids"])
        stage = self.stage_transport(batch["stage_ids"])
        protocol = self.protocol_bridge(batch["protocol_ids"])

        routed = self.relation_context_fiber_router(
            torch.cat([family, concept, relation, context], dim=-1)
        )
        novelty = batch.get("novelty", torch.zeros_like(batch["family_ids"], dtype=family.dtype).unsqueeze(-1))
        retention = batch.get("retention", torch.zeros_like(batch["family_ids"], dtype=family.dtype).unsqueeze(-1))
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

        theorem_input = torch.cat(
            [protocol_state, successor, novelty, retention, gate_metrics["write_gate"], gate_metrics["read_gate"]],
            dim=-1,
        )
        theorem_logits = self.theorem_survival_monitor(theorem_input)

        return {
            "family_state": family,
            "concept_state": concept,
            "routed_state": routed,
            "fast_state": fast_state,
            "slow_state": slow_state,
            "successor_state": successor,
            "protocol_state": protocol_state,
            "brain_probe": self.brain_probe_alignment_head(protocol_state),
            "task_logits": self.task_head(protocol_state),
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
        margin = (protocol_energy - successor_energy).clamp(min=0.0)
        theorem_prob = torch.sigmoid(out["theorem_logits"]).mean().detach()

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
        }

    def snapshot(self) -> None:
        self._rollback_snapshot = {
            k: v.detach().clone()
            for k, v in self.state_dict().items()
        }

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

    @torch.no_grad()
    def online_update_step(
        self,
        batch: Dict[str, torch.Tensor],
        lr: float = 1e-3,
    ) -> Dict[str, float]:
        """
        持续学习更新：
        - 只对概念 memory bank 与 protocol bridge 做小步受限更新
        - stress 高时自动收缩写入幅度
        """
        self.train()
        for p in self.parameters():
            p.grad = None
        loss, metrics = self.compute_loss(batch)
        loss.backward()

        write_gate = float(self.forward(batch)["write_gate"].mean().detach())
        write_scale = max(self.config.guarded_write_floor, write_gate)

        allowed_prefixes = (
            "concept_section_memory_bank",
            "protocol_bridge",
            "protocol_field_bridge_bus",
            "stage_successor_transport_engine",
        )
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.grad is None:
                    continue
                if name.startswith(allowed_prefixes):
                    param.add_(param.grad, alpha=-lr * write_scale)
        metrics["online_write_scale"] = write_scale
        return metrics


def make_synthetic_batch(
    config: ICSPBLargeOnlineConfig,
    batch_size: int = 16,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    family_ids = torch.randint(0, config.family_vocab_size, (batch_size,), generator=g)
    concept_ids = family_ids * (config.concept_vocab_size // config.family_vocab_size) + torch.randint(
        0, max(1, config.concept_vocab_size // config.family_vocab_size), (batch_size,), generator=g
    )
    relation_ids = torch.randint(0, config.relation_vocab_size, (batch_size,), generator=g)
    context_ids = torch.randint(0, config.context_vocab_size, (batch_size,), generator=g)
    stage_ids = torch.randint(0, config.stage_vocab_size, (batch_size,), generator=g)
    protocol_ids = torch.randint(0, config.protocol_vocab_size, (batch_size,), generator=g)
    labels = (concept_ids + relation_ids + stage_ids) % config.task_classes
    novelty = torch.rand(batch_size, 1, generator=g) * 0.35
    retention = torch.rand(batch_size, 1, generator=g) * 0.25
    brain_targets = torch.randn(batch_size, config.brain_probe_dim, generator=g) * 0.1

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
    }

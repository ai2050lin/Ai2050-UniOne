from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpikeICSPBLMConfig:
    vocab_size: int = 256
    hidden_dim: int = 192
    patch_slots: int = 64
    max_seq_len: int = 128
    phase_dim: int = 16
    write_decay: float = 0.98
    eligibility_decay: float = 0.92
    local_lr: float = 0.05


class EventPatchSelector(nn.Module):
    def __init__(self, config: SpikeICSPBLMConfig):
        super().__init__()
        h = config.hidden_dim
        self.patch_keys = nn.Parameter(torch.randn(config.patch_slots, h) * 0.02)
        self.patch_values = nn.Parameter(torch.randn(config.patch_slots, h) * 0.02)
        self.selector = nn.Linear(h * 2, config.patch_slots)

    def forward(self, token_state: torch.Tensor, prev_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.selector(torch.cat([token_state, prev_state], dim=-1))
        weights = torch.softmax(logits, dim=-1)
        patch_state = weights @ self.patch_values
        return patch_state, weights


class BurstSectionBinder(nn.Module):
    def __init__(self, config: SpikeICSPBLMConfig):
        super().__init__()
        h = config.hidden_dim
        self.net = nn.Sequential(
            nn.Linear(h * 3, h),
            nn.GELU(),
            nn.Linear(h, h),
        )

    def forward(self, token_state: torch.Tensor, patch_state: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(torch.cat([token_state, patch_state, prev_state], dim=-1)))


class PhaseGatedSuccessorCore(nn.Module):
    def __init__(self, config: SpikeICSPBLMConfig):
        super().__init__()
        h = config.hidden_dim
        self.phase_embed = nn.Parameter(torch.randn(config.phase_dim, h) * 0.02)
        self.phase_router = nn.Linear(h * 2, config.phase_dim)
        self.successor = nn.Sequential(
            nn.Linear(h * 3, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.phase_gate = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )

    def forward(
        self,
        section_state: torch.Tensor,
        patch_state: torch.Tensor,
        prev_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        phase_logits = self.phase_router(torch.cat([section_state, prev_state], dim=-1))
        phase_weights = torch.softmax(phase_logits, dim=-1)
        phase_state = phase_weights @ self.phase_embed
        successor = torch.tanh(self.successor(torch.cat([section_state, patch_state, phase_state], dim=-1)))
        gate = torch.sigmoid(self.phase_gate(torch.cat([successor, prev_state], dim=-1)))
        hidden = gate * successor + (1.0 - gate) * prev_state
        return hidden, {
            "phase_weights": phase_weights,
            "successor_gate": gate,
        }


class PopulationReadout(nn.Module):
    def __init__(self, config: SpikeICSPBLMConfig):
        super().__init__()
        self.decoder = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(hidden)


class SpikeICSPBLMMinimal(nn.Module):
    """
    最小 SpikeICSPB-LM 原型：
    - 不使用 Attention
    - 训练更新默认走 local eligibility + modulatory error
    - 目标是验证 patch selector / successor core / population readout 的最小闭环
    """

    def __init__(self, config: SpikeICSPBLMConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.token_embedding = nn.Embedding(config.vocab_size, h)
        self.patch_selector = EventPatchSelector(config)
        self.section_binder = BurstSectionBinder(config)
        self.successor_core = PhaseGatedSuccessorCore(config)
        self.population_readout = PopulationReadout(config)
        self.output_norm = nn.LayerNorm(h)

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("序列长度超过 max_seq_len")

        prev_state = torch.zeros(bsz, self.config.hidden_dim, device=input_ids.device, dtype=torch.float32)
        logits_steps: List[torch.Tensor] = []
        hidden_steps: List[torch.Tensor] = []
        patch_steps: List[torch.Tensor] = []
        phase_steps: List[torch.Tensor] = []
        gate_steps: List[torch.Tensor] = []

        for t in range(seq_len):
            token_state = self.token_embedding(input_ids[:, t])
            patch_state, patch_weights = self.patch_selector(token_state, prev_state)
            section_state = self.section_binder(token_state, patch_state, prev_state)
            hidden, metrics = self.successor_core(section_state, patch_state, prev_state)
            hidden = self.output_norm(hidden)
            logits = self.population_readout(hidden)

            logits_steps.append(logits)
            hidden_steps.append(hidden)
            patch_steps.append(patch_weights)
            phase_steps.append(metrics["phase_weights"])
            gate_steps.append(metrics["successor_gate"])
            prev_state = hidden

        return {
            "logits": torch.stack(logits_steps, dim=1),
            "hidden_states": torch.stack(hidden_steps, dim=1),
            "patch_weights": torch.stack(patch_steps, dim=1),
            "phase_weights": torch.stack(phase_steps, dim=1),
            "successor_gate": torch.stack(gate_steps, dim=1),
        }

    def compute_loss(self, input_ids: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.forward(input_ids)
        logits = out["logits"]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            targets[:, 1:].reshape(-1),
        )
        patch_entropy = float((-(out["patch_weights"] * (out["patch_weights"] + 1e-8).log()).sum(dim=-1)).mean().item())
        successor_gate_mean = float(out["successor_gate"].mean().item())
        return loss, {
            "loss": float(loss.item()),
            "patch_entropy": patch_entropy,
            "successor_gate_mean": successor_gate_mean,
        }

    @torch.no_grad()
    def local_update_step(self, input_ids: torch.Tensor, targets: torch.Tensor, lr: float | None = None) -> Dict[str, float]:
        if lr is None:
            lr = self.config.local_lr
        out = self.forward(input_ids)
        logits = out["logits"]
        hidden = out["hidden_states"]
        patch_weights = out["patch_weights"]

        probs = torch.softmax(logits[:, :-1, :], dim=-1)
        target_onehot = F.one_hot(targets[:, 1:], num_classes=self.config.vocab_size).to(probs.dtype)
        error = target_onehot - probs

        hidden_flat = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
        error_flat = error.reshape(-1, error.size(-1))
        delta_decoder = error_flat.t() @ hidden_flat / max(1, hidden_flat.size(0))
        self.population_readout.decoder.weight.add_(lr * delta_decoder)

        token_states = self.token_embedding(input_ids[:, :-1]).reshape(-1, hidden.size(-1))
        mean_error_drive = error_flat @ self.population_readout.decoder.weight
        delta_embed = 0.05 * (mean_error_drive * token_states.sign())
        token_ids_flat = input_ids[:, :-1].reshape(-1)
        for tok in torch.unique(token_ids_flat):
            mask = token_ids_flat == tok
            self.token_embedding.weight[tok].mul_(self.config.write_decay)
            self.token_embedding.weight[tok].add_(lr * delta_embed[mask].mean(dim=0))

        patch_flat = patch_weights[:, :-1, :].reshape(-1, patch_weights.size(-1))
        patch_drive = patch_flat.t() @ mean_error_drive / max(1, mean_error_drive.size(0))
        self.patch_selector.patch_values.mul_(self.config.write_decay)
        self.patch_selector.patch_values.add_(lr * patch_drive)

        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            targets[:, 1:].reshape(-1),
        )
        return {
            "loss": float(loss.item()),
            "avg_error_norm": float(mean_error_drive.norm(dim=-1).mean().item()),
            "patch_mass_mean": float(patch_weights.mean().item()),
        }

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        generated = input_ids.clone()
        last_metrics: Dict[str, torch.Tensor] = {}
        for _ in range(max_new_tokens):
            window = generated[:, -self.config.max_seq_len :]
            out = self.forward(window)
            logits = out["logits"][:, -1, :]
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            last_metrics = {
                "patch_weights": out["patch_weights"][:, -1, :],
                "phase_weights": out["phase_weights"][:, -1, :],
                "successor_gate": out["successor_gate"][:, -1, :],
            }
        return {
            "generated_ids": generated,
            "last_metrics": last_metrics,
        }

    @staticmethod
    def decode_token_ids(token_ids: List[int] | torch.Tensor) -> str:
        if torch.is_tensor(token_ids):
            token_ids = token_ids.detach().cpu().tolist()
        return bytes([int(token) % 256 for token in token_ids]).decode("utf-8", errors="ignore")

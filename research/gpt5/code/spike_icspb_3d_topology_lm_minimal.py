from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpikeICSPB3DConfig:
    vocab_size: int = 256
    hidden_dim: int = 192
    patch_slots: int = 64
    max_seq_len: int = 128
    phase_dim: int = 16
    topology_radius: float = 0.42
    bridge_scale: float = 0.18
    bridge_topk: int = 6
    potential_limit: float = 1.25
    write_decay: float = 0.98
    local_lr: float = 0.05


def build_lattice_positions(patch_slots: int) -> torch.Tensor:
    side = round(patch_slots ** (1.0 / 3.0))
    if side ** 3 != patch_slots:
        raise ValueError("patch_slots must form a cube for the minimal 3D lattice")
    coords = torch.linspace(-1.0, 1.0, steps=side)
    rows = []
    for x in coords:
        for y in coords:
            for z in coords:
                rows.append([float(x), float(y), float(z)])
    return torch.tensor(rows, dtype=torch.float32)


class TopologyAwarePatchSelector(nn.Module):
    def __init__(self, config: SpikeICSPB3DConfig):
        super().__init__()
        h = config.hidden_dim
        self.config = config
        self.patch_keys = nn.Parameter(torch.randn(config.patch_slots, h) * 0.02)
        self.patch_values = nn.Parameter(torch.randn(config.patch_slots, h) * 0.02)
        self.selector = nn.Linear(h * 2, config.patch_slots)
        self.bridge_router = nn.Linear(h * 2, config.patch_slots)
        self.register_buffer("patch_positions", build_lattice_positions(config.patch_slots), persistent=False)

    def forward(
        self,
        token_state: torch.Tensor,
        prev_state: torch.Tensor,
        radius_override: torch.Tensor | None = None,
        bridge_scale_override: torch.Tensor | None = None,
        bridge_topk_override: int | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        query = torch.cat([token_state, prev_state], dim=-1)
        base_logits = self.selector(query)
        base_weights = torch.softmax(base_logits, dim=-1)

        anchor_pos = base_weights @ self.patch_positions
        distances = torch.cdist(anchor_pos.unsqueeze(1), self.patch_positions.unsqueeze(0)).squeeze(1)
        if radius_override is None:
            radius = torch.full(
                (token_state.size(0), 1),
                float(self.config.topology_radius),
                device=token_state.device,
                dtype=token_state.dtype,
            )
        else:
            radius = radius_override.to(device=token_state.device, dtype=token_state.dtype).clamp_min(1e-4)
        local_kernel = torch.exp(-(distances ** 2) / (2.0 * radius.pow(2)))
        local_mask = distances <= radius
        nearest_index = torch.argmin(distances, dim=-1, keepdim=True)
        local_mask = local_mask.scatter(dim=-1, index=nearest_index, value=True)

        bridge_logits = self.bridge_router(query)
        bridge_mask = ~local_mask
        bridge_logits = bridge_logits.masked_fill(~bridge_mask, -1e9)
        topk = min(bridge_topk_override or self.config.bridge_topk, bridge_logits.size(-1))
        topk_values, topk_indices = torch.topk(bridge_logits, k=topk, dim=-1)
        sparse_bridge = torch.zeros_like(bridge_logits)
        sparse_bridge.scatter_(dim=-1, index=topk_indices, src=torch.softmax(topk_values, dim=-1))

        if bridge_scale_override is None:
            bridge_scale = torch.full(
                (token_state.size(0), 1),
                float(self.config.bridge_scale),
                device=token_state.device,
                dtype=token_state.dtype,
            )
        else:
            bridge_scale = bridge_scale_override.to(device=token_state.device, dtype=token_state.dtype).clamp_min(0.0)

        mixed = base_weights * local_kernel * local_mask.to(base_weights.dtype) + bridge_scale * sparse_bridge
        weights = mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        patch_state = weights @ self.patch_values

        local_mass = (weights * local_mask.to(weights.dtype)).sum(dim=-1, keepdim=True)
        bridge_mass = (weights * bridge_mask.to(weights.dtype)).sum(dim=-1, keepdim=True)
        mean_distance = (weights * distances).sum(dim=-1, keepdim=True)

        return patch_state, {
            "weights": weights,
            "anchor_pos": anchor_pos,
            "local_mass": local_mass,
            "bridge_mass": bridge_mass,
            "mean_distance": mean_distance,
            "effective_radius": radius,
            "effective_bridge_scale": bridge_scale,
        }


class BurstSectionBinder(nn.Module):
    def __init__(self, config: SpikeICSPB3DConfig):
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
    def __init__(self, config: SpikeICSPB3DConfig):
        super().__init__()
        h = config.hidden_dim
        self.phase_embed = nn.Parameter(torch.randn(config.phase_dim, h) * 0.02)
        self.phase_router = nn.Linear(h * 2, config.phase_dim)
        self.successor = nn.Sequential(
            nn.Linear(h * 4, h),
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
        topology_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        phase_logits = self.phase_router(torch.cat([section_state, prev_state], dim=-1))
        phase_weights = torch.softmax(phase_logits, dim=-1)
        phase_state = phase_weights @ self.phase_embed
        successor = torch.tanh(
            self.successor(torch.cat([section_state, patch_state, phase_state, topology_state], dim=-1))
        )
        gate = torch.sigmoid(self.phase_gate(torch.cat([successor, prev_state], dim=-1)))
        hidden = gate * successor + (1.0 - gate) * prev_state
        return hidden, {
            "phase_weights": phase_weights,
            "successor_gate": gate,
        }


class PopulationReadout(nn.Module):
    def __init__(self, config: SpikeICSPB3DConfig):
        super().__init__()
        self.decoder = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(hidden)


class SpikeICSPB3DLMMinimal(nn.Module):
    def __init__(self, config: SpikeICSPB3DConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.token_embedding = nn.Embedding(config.vocab_size, h)
        self.patch_selector = TopologyAwarePatchSelector(config)
        self.section_binder = BurstSectionBinder(config)
        self.successor_core = PhaseGatedSuccessorCore(config)
        self.population_readout = PopulationReadout(config)
        self.output_norm = nn.LayerNorm(h)
        self.topology_projector = nn.Linear(5, h)

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

        prev_state = torch.zeros(bsz, self.config.hidden_dim, device=input_ids.device, dtype=torch.float32)
        logits_steps: List[torch.Tensor] = []
        hidden_steps: List[torch.Tensor] = []
        patch_steps: List[torch.Tensor] = []
        phase_steps: List[torch.Tensor] = []
        gate_steps: List[torch.Tensor] = []
        local_mass_steps: List[torch.Tensor] = []
        bridge_mass_steps: List[torch.Tensor] = []
        distance_steps: List[torch.Tensor] = []

        for t in range(seq_len):
            token_state = self.token_embedding(input_ids[:, t])
            patch_state, topo = self.patch_selector(token_state, prev_state)
            section_state = self.section_binder(token_state, patch_state, prev_state)
            topo_features = torch.cat(
                [
                    topo["anchor_pos"],
                    topo["local_mass"],
                    topo["bridge_mass"],
                ],
                dim=-1,
            )
            topology_state = torch.tanh(self.topology_projector(topo_features))
            hidden, metrics = self.successor_core(section_state, patch_state, prev_state, topology_state)
            hidden = self.output_norm(hidden)
            hidden = hidden.clamp(min=-self.config.potential_limit, max=self.config.potential_limit)
            logits = self.population_readout(hidden)

            logits_steps.append(logits)
            hidden_steps.append(hidden)
            patch_steps.append(topo["weights"])
            phase_steps.append(metrics["phase_weights"])
            gate_steps.append(metrics["successor_gate"])
            local_mass_steps.append(topo["local_mass"])
            bridge_mass_steps.append(topo["bridge_mass"])
            distance_steps.append(topo["mean_distance"])
            prev_state = hidden

        return {
            "logits": torch.stack(logits_steps, dim=1),
            "hidden_states": torch.stack(hidden_steps, dim=1),
            "patch_weights": torch.stack(patch_steps, dim=1),
            "phase_weights": torch.stack(phase_steps, dim=1),
            "successor_gate": torch.stack(gate_steps, dim=1),
            "local_mass": torch.stack(local_mass_steps, dim=1),
            "bridge_mass": torch.stack(bridge_mass_steps, dim=1),
            "mean_distance": torch.stack(distance_steps, dim=1),
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
        local_mass_mean = float(out["local_mass"].mean().item())
        bridge_mass_mean = float(out["bridge_mass"].mean().item())
        mean_distance = float(out["mean_distance"].mean().item())
        potential_abs_mean = float(out["hidden_states"].abs().mean().item())
        potential_saturation = float((out["hidden_states"].abs() >= 0.95 * self.config.potential_limit).to(torch.float32).mean().item())
        return loss, {
            "loss": float(loss.item()),
            "patch_entropy": patch_entropy,
            "successor_gate_mean": successor_gate_mean,
            "local_mass_mean": local_mass_mean,
            "bridge_mass_mean": bridge_mass_mean,
            "mean_topology_distance": mean_distance,
            "potential_abs_mean": potential_abs_mean,
            "potential_saturation_fraction": potential_saturation,
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
            "local_mass_mean": float(out["local_mass"].mean().item()),
            "bridge_mass_mean": float(out["bridge_mass"].mean().item()),
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
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            last_metrics = {
                "patch_weights": out["patch_weights"][:, -1, :],
                "phase_weights": out["phase_weights"][:, -1, :],
                "successor_gate": out["successor_gate"][:, -1, :],
                "local_mass": out["local_mass"][:, -1, :],
                "bridge_mass": out["bridge_mass"][:, -1, :],
                "mean_distance": out["mean_distance"][:, -1, :],
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

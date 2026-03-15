from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.gpt5.code.spike_icspb_3d_topology_lm_minimal import (
    BurstSectionBinder,
    PhaseGatedSuccessorCore,
    PopulationReadout,
    SpikeICSPB3DConfig,
    TopologyAwarePatchSelector,
)


@dataclass
class SpikeICSPB3DMultiRegionConfig(SpikeICSPB3DConfig):
    region_names: Tuple[str, ...] = ("syntax", "semantic", "memory")
    replay_decay: float = 0.96
    consolidation_lr: float = 0.08
    bridge_mix: float = 0.22
    region_hidden_dim: int = 128
    homeostasis_target_abs: float = 0.52
    homeostasis_gain: float = 0.18
    homeostasis_lr: float = 0.05


class RegionCore(nn.Module):
    def __init__(self, config: SpikeICSPB3DMultiRegionConfig):
        super().__init__()
        regional = SpikeICSPB3DConfig(
            vocab_size=config.vocab_size,
            hidden_dim=config.region_hidden_dim,
            patch_slots=config.patch_slots,
            max_seq_len=config.max_seq_len,
            phase_dim=config.phase_dim,
            topology_radius=config.topology_radius,
            bridge_scale=config.bridge_scale,
            bridge_topk=config.bridge_topk,
            write_decay=config.write_decay,
            local_lr=config.local_lr,
        )
        self.patch_selector = TopologyAwarePatchSelector(regional)
        self.section_binder = BurstSectionBinder(regional)
        self.successor_core = PhaseGatedSuccessorCore(regional)
        self.topology_projector = nn.Linear(5, config.region_hidden_dim)
        self.output_norm = nn.LayerNorm(config.region_hidden_dim)
        self.topology_controller = nn.Sequential(
            nn.Linear(config.region_hidden_dim * 2, config.region_hidden_dim),
            nn.GELU(),
            nn.Linear(config.region_hidden_dim, 2),
        )
        self.homeostasis_controller = nn.Sequential(
            nn.Linear(config.region_hidden_dim * 3, config.region_hidden_dim),
            nn.GELU(),
            nn.Linear(config.region_hidden_dim, 3),
        )
        self.base_radius = float(config.topology_radius)
        self.base_bridge_scale = float(config.bridge_scale)
        self.bridge_topk = int(config.bridge_topk)
        self.homeostasis_target_abs = float(config.homeostasis_target_abs)
        self.homeostasis_gain = float(config.homeostasis_gain)

    def forward(
        self,
        token_state: torch.Tensor,
        prev_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        controls = self.topology_controller(torch.cat([token_state, prev_state], dim=-1))
        radius_factor = 0.6 + 1.0 * torch.sigmoid(controls[:, :1])
        bridge_factor = 0.3 + 1.2 * torch.sigmoid(controls[:, 1:2])
        effective_radius = radius_factor * self.base_radius
        effective_bridge_scale = bridge_factor * self.base_bridge_scale

        patch_state, topo = self.patch_selector(
            token_state,
            prev_state,
            radius_override=effective_radius,
            bridge_scale_override=effective_bridge_scale,
            bridge_topk_override=self.bridge_topk,
        )
        section_state = self.section_binder(token_state, patch_state, prev_state)
        topo_features = torch.cat([topo["anchor_pos"], topo["local_mass"], topo["bridge_mass"]], dim=-1)
        topology_state = torch.tanh(self.topology_projector(topo_features))
        hidden, metrics = self.successor_core(section_state, patch_state, prev_state, topology_state)
        hidden = self.output_norm(hidden)
        homeo = self.homeostasis_controller(torch.cat([token_state, prev_state, hidden], dim=-1))
        homeo_gain = 0.70 + 0.60 * torch.sigmoid(homeo[:, :1])
        homeo_bias = 0.20 * torch.tanh(homeo[:, 1:2])
        homeo_leak = 0.04 + 0.26 * torch.sigmoid(homeo[:, 2:3])
        target_anchor = torch.tanh(0.55 * token_state + 0.30 * prev_state + 0.15 * patch_state)
        hidden = homeo_gain * hidden + homeo_bias - homeo_leak * (hidden - target_anchor)
        hidden = hidden.clamp(
            min=-self.patch_selector.config.potential_limit,
            max=self.patch_selector.config.potential_limit,
        )
        potential_pressure = (hidden.abs().mean(dim=-1, keepdim=True) / self.patch_selector.config.potential_limit).clamp(0.0, 2.0)
        return hidden, {
            "patch_weights": topo["weights"],
            "local_mass": topo["local_mass"],
            "bridge_mass": topo["bridge_mass"],
            "mean_distance": topo["mean_distance"],
            "phase_weights": metrics["phase_weights"],
            "successor_gate": metrics["successor_gate"],
            "effective_radius": topo["effective_radius"],
            "effective_bridge_scale": topo["effective_bridge_scale"],
            "homeostatic_gain": homeo_gain,
            "homeostatic_bias": homeo_bias,
            "homeostatic_leak": homeo_leak,
            "potential_pressure": potential_pressure,
        }


class SpikeICSPB3DMultiRegionPhaseA(nn.Module):
    def __init__(self, config: SpikeICSPB3DMultiRegionConfig):
        super().__init__()
        self.config = config
        h = config.region_hidden_dim
        self.token_embedding = nn.Embedding(config.vocab_size, h)
        self.region_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(1, h) * 0.02) for name in config.region_names}
        )
        self.regions = nn.ModuleDict({name: RegionCore(config) for name in config.region_names})
        self.region_bridge = nn.Parameter(torch.randn(len(config.region_names), len(config.region_names)) * 0.02)
        self.bridge_projector = nn.Linear(h * 2, h)
        self.protocol_router = nn.Sequential(
            nn.Linear(h * 3, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.population_readout = PopulationReadout(
            SpikeICSPB3DConfig(vocab_size=config.vocab_size, hidden_dim=h)
        )
        self.replay_head = nn.Linear(h, h)
        self.replay_gate = nn.Linear(h * 2, 1)
        self.register_buffer("slow_scaffold", torch.zeros(h), persistent=False)

    def _region_bridge_mix(self, region_states: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        names = list(self.config.region_names)
        stack = torch.stack([region_states[name] for name in names], dim=1)
        weights = torch.softmax(self.region_bridge, dim=-1)
        mixed = {}
        for idx, name in enumerate(names):
            bridge_state = torch.einsum("r,brh->bh", weights[idx], stack)
            mixed[name] = bridge_state
        return mixed, weights

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

        prev_states = {
            name: torch.zeros(bsz, self.config.region_hidden_dim, device=input_ids.device, dtype=torch.float32)
            for name in self.config.region_names
        }
        replay_state = self.slow_scaffold.unsqueeze(0).expand(bsz, -1).to(input_ids.device)

        logits_steps: List[torch.Tensor] = []
        fused_steps: List[torch.Tensor] = []
        replay_steps: List[torch.Tensor] = []
        protocol_steps: List[torch.Tensor] = []
        region_hidden_steps: Dict[str, List[torch.Tensor]] = {name: [] for name in self.config.region_names}
        patch_weight_steps: Dict[str, List[torch.Tensor]] = {name: [] for name in self.config.region_names}
        local_mass_steps: Dict[str, List[torch.Tensor]] = {name: [] for name in self.config.region_names}
        bridge_mass_steps: Dict[str, List[torch.Tensor]] = {name: [] for name in self.config.region_names}
        distance_steps: Dict[str, List[torch.Tensor]] = {name: [] for name in self.config.region_names}
        radius_steps: Dict[str, List[torch.Tensor]] = {name: [] for name in self.config.region_names}
        bridge_scale_steps: Dict[str, List[torch.Tensor]] = {name: [] for name in self.config.region_names}
        gate_steps: Dict[str, List[torch.Tensor]] = {name: [] for name in self.config.region_names}
        phase_steps: Dict[str, List[torch.Tensor]] = {name: [] for name in self.config.region_names}
        gain_steps: Dict[str, List[torch.Tensor]] = {name: [] for name in self.config.region_names}
        leak_steps: Dict[str, List[torch.Tensor]] = {name: [] for name in self.config.region_names}
        pressure_steps: Dict[str, List[torch.Tensor]] = {name: [] for name in self.config.region_names}
        bridge_weight_steps: List[torch.Tensor] = []

        for t in range(seq_len):
            base_token = self.token_embedding(input_ids[:, t])
            raw_region_states: Dict[str, torch.Tensor] = {}
            raw_metrics: Dict[str, Dict[str, torch.Tensor]] = {}
            for name in self.config.region_names:
                token_state = base_token + self.region_embeddings[name]
                region_hidden, region_metrics = self.regions[name](token_state, prev_states[name])
                raw_region_states[name] = region_hidden
                raw_metrics[name] = region_metrics

            bridge_states, bridge_weights = self._region_bridge_mix(raw_region_states)
            bridge_weight_steps.append(bridge_weights.unsqueeze(0).expand(bsz, -1, -1))

            next_states: Dict[str, torch.Tensor] = {}
            for name in self.config.region_names:
                bridge_state = bridge_states[name]
                mixed_hidden = torch.tanh(
                    self.bridge_projector(torch.cat([raw_region_states[name], bridge_state], dim=-1))
                )
                next_hidden = (1.0 - self.config.bridge_mix) * raw_region_states[name] + self.config.bridge_mix * mixed_hidden
                next_hidden = next_hidden.clamp(min=-self.config.potential_limit, max=self.config.potential_limit)
                next_states[name] = next_hidden
                region_hidden_steps[name].append(next_hidden)
                patch_weight_steps[name].append(raw_metrics[name]["patch_weights"])
                local_mass_steps[name].append(raw_metrics[name]["local_mass"])
                bridge_mass_steps[name].append(raw_metrics[name]["bridge_mass"])
                distance_steps[name].append(raw_metrics[name]["mean_distance"])
                radius_steps[name].append(raw_metrics[name]["effective_radius"])
                bridge_scale_steps[name].append(raw_metrics[name]["effective_bridge_scale"])
                gate_steps[name].append(raw_metrics[name]["successor_gate"])
                phase_steps[name].append(raw_metrics[name]["phase_weights"])
                gain_steps[name].append(raw_metrics[name]["homeostatic_gain"])
                leak_steps[name].append(raw_metrics[name]["homeostatic_leak"])
                pressure_steps[name].append(raw_metrics[name]["potential_pressure"])

            fused = torch.stack([next_states[name] for name in self.config.region_names], dim=1).mean(dim=1)
            replay_input = next_states["memory"]
            replay_gate = torch.sigmoid(self.replay_gate(torch.cat([replay_input, replay_state], dim=-1)))
            replay_candidate = torch.tanh(self.replay_head(replay_input))
            replay_state = self.config.replay_decay * replay_state + (1.0 - self.config.replay_decay) * replay_gate * replay_candidate
            replay_state = replay_state.clamp(min=-self.config.potential_limit, max=self.config.potential_limit)
            protocol_anchor = torch.stack([next_states[name] for name in self.config.region_names], dim=1).amax(dim=1)
            protocol_state = torch.tanh(self.protocol_router(torch.cat([fused, replay_state, protocol_anchor], dim=-1)))
            protocol_state = protocol_state.clamp(min=-self.config.potential_limit, max=self.config.potential_limit)
            fused_with_memory = (
                fused
                + 0.25 * replay_state
                + 0.18 * protocol_state
                + 0.10 * self.slow_scaffold.to(input_ids.device)
            )
            fused_with_memory = fused_with_memory.clamp(min=-self.config.potential_limit, max=self.config.potential_limit)
            logits = self.population_readout(fused_with_memory)

            logits_steps.append(logits)
            fused_steps.append(fused_with_memory)
            replay_steps.append(replay_state)
            protocol_steps.append(protocol_state)
            prev_states = next_states

        return {
            "logits": torch.stack(logits_steps, dim=1),
            "fused_states": torch.stack(fused_steps, dim=1),
            "replay_states": torch.stack(replay_steps, dim=1),
            "protocol_states": torch.stack(protocol_steps, dim=1),
            "bridge_weights": torch.stack(bridge_weight_steps, dim=1),
            "region_hidden_states": {
                name: torch.stack(rows, dim=1) for name, rows in region_hidden_steps.items()
            },
            "patch_weights": {name: torch.stack(rows, dim=1) for name, rows in patch_weight_steps.items()},
            "local_mass": {name: torch.stack(rows, dim=1) for name, rows in local_mass_steps.items()},
            "bridge_mass": {name: torch.stack(rows, dim=1) for name, rows in bridge_mass_steps.items()},
            "mean_distance": {name: torch.stack(rows, dim=1) for name, rows in distance_steps.items()},
            "effective_radius": {name: torch.stack(rows, dim=1) for name, rows in radius_steps.items()},
            "effective_bridge_scale": {name: torch.stack(rows, dim=1) for name, rows in bridge_scale_steps.items()},
            "successor_gate": {name: torch.stack(rows, dim=1) for name, rows in gate_steps.items()},
            "phase_weights": {name: torch.stack(rows, dim=1) for name, rows in phase_steps.items()},
            "homeostatic_gain": {name: torch.stack(rows, dim=1) for name, rows in gain_steps.items()},
            "homeostatic_leak": {name: torch.stack(rows, dim=1) for name, rows in leak_steps.items()},
            "potential_pressure": {name: torch.stack(rows, dim=1) for name, rows in pressure_steps.items()},
        }

    def compute_loss(self, input_ids: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.forward(input_ids)
        logits = out["logits"]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            targets[:, 1:].reshape(-1),
        )
        region_means = []
        local_mass = []
        bridge_mass = []
        radius_means = []
        bridge_scale_means = []
        gain_means = []
        leak_means = []
        pressure_means = []
        for name in self.config.region_names:
            region_hidden = out["region_hidden_states"][name]
            region_means.append(region_hidden.mean(dim=(0, 1)))
            local_mass.append(float(out["local_mass"][name].mean().item()))
            bridge_mass.append(float(out["bridge_mass"][name].mean().item()))
            radius_means.append(float(out["effective_radius"][name].mean().item()))
            bridge_scale_means.append(float(out["effective_bridge_scale"][name].mean().item()))
            gain_means.append(float(out["homeostatic_gain"][name].mean().item()))
            leak_means.append(float(out["homeostatic_leak"][name].mean().item()))
            pressure_means.append(float(out["potential_pressure"][name].mean().item()))
        region_diversity = 0.0
        for i in range(len(region_means)):
            for j in range(i + 1, len(region_means)):
                region_diversity += float(torch.norm(region_means[i] - region_means[j]).item())
        num_pairs = max(1, len(region_means) * (len(region_means) - 1) / 2)
        region_diversity /= num_pairs
        radius_diversity = float(max(radius_means) - min(radius_means))
        bridge_scale_diversity = float(max(bridge_scale_means) - min(bridge_scale_means))
        replay_energy = float(out["replay_states"].norm(dim=-1).mean().item())
        protocol_energy = float(out["protocol_states"].norm(dim=-1).mean().item())
        successor_energy = float(
            sum(out["region_hidden_states"][name].norm(dim=-1).mean().item() for name in self.config.region_names)
            / len(self.config.region_names)
        )
        successor_protocol_margin = float(max(0.0, protocol_energy - successor_energy))
        bridge_entropy = float((-(out["bridge_weights"] * (out["bridge_weights"] + 1e-8).log()).sum(dim=-1)).mean().item())
        region_potential_abs = []
        region_saturation = []
        for name in self.config.region_names:
            region_states = out["region_hidden_states"][name]
            region_potential_abs.append(float(region_states.abs().mean().item()))
            region_saturation.append(
                float((region_states.abs() >= 0.95 * self.config.potential_limit).to(torch.float32).mean().item())
            )
        fused_potential_abs = float(out["fused_states"].abs().mean().item())
        fused_saturation = float(
            (out["fused_states"].abs() >= 0.95 * self.config.potential_limit).to(torch.float32).mean().item()
        )
        return loss, {
            "loss": float(loss.item()),
            "region_diversity": region_diversity,
            "mean_local_mass": float(sum(local_mass) / len(local_mass)),
            "mean_bridge_mass": float(sum(bridge_mass) / len(bridge_mass)),
            "mean_effective_radius": float(sum(radius_means) / len(radius_means)),
            "mean_effective_bridge_scale": float(sum(bridge_scale_means) / len(bridge_scale_means)),
            "radius_diversity": radius_diversity,
            "bridge_scale_diversity": bridge_scale_diversity,
            "mean_homeostatic_gain": float(sum(gain_means) / len(gain_means)),
            "mean_homeostatic_leak": float(sum(leak_means) / len(leak_means)),
            "mean_potential_pressure": float(sum(pressure_means) / len(pressure_means)),
            "replay_energy": replay_energy,
            "protocol_energy": protocol_energy,
            "successor_energy": successor_energy,
            "successor_protocol_margin": successor_protocol_margin,
            "bridge_entropy": bridge_entropy,
            "mean_region_potential_abs": float(sum(region_potential_abs) / len(region_potential_abs)),
            "mean_region_saturation_fraction": float(sum(region_saturation) / len(region_saturation)),
            "fused_potential_abs": fused_potential_abs,
            "fused_saturation_fraction": fused_saturation,
        }

    @torch.no_grad()
    def local_update_step(self, input_ids: torch.Tensor, targets: torch.Tensor, lr: float | None = None) -> Dict[str, float]:
        if lr is None:
            lr = self.config.local_lr
        out = self.forward(input_ids)
        logits = out["logits"]
        fused = out["fused_states"]
        probs = torch.softmax(logits[:, :-1, :], dim=-1)
        target_onehot = F.one_hot(targets[:, 1:], num_classes=self.config.vocab_size).to(probs.dtype)
        error = target_onehot - probs

        fused_flat = fused[:, :-1, :].reshape(-1, fused.size(-1))
        error_flat = error.reshape(-1, error.size(-1))
        delta_decoder = error_flat.t() @ fused_flat / max(1, fused_flat.size(0))
        self.population_readout.decoder.weight.add_(lr * delta_decoder)

        token_states = self.token_embedding(input_ids[:, :-1]).reshape(-1, fused.size(-1))
        mean_error_drive = error_flat @ self.population_readout.decoder.weight
        delta_embed = 0.04 * (mean_error_drive * token_states.sign())
        token_ids_flat = input_ids[:, :-1].reshape(-1)
        for tok in torch.unique(token_ids_flat):
            mask = token_ids_flat == tok
            self.token_embedding.weight[tok].mul_(self.config.write_decay)
            self.token_embedding.weight[tok].add_(lr * delta_embed[mask].mean(dim=0))

        for name in self.config.region_names:
            patch_weights = out["patch_weights"][name][:, :-1, :].reshape(-1, out["patch_weights"][name].size(-1))
            patch_drive = patch_weights.t() @ mean_error_drive / max(1, mean_error_drive.size(0))
            self.regions[name].patch_selector.patch_values.mul_(self.config.write_decay)
            self.regions[name].patch_selector.patch_values.add_(0.25 * lr * patch_drive)
            region_states = out["region_hidden_states"][name]
            state_abs = float(region_states.abs().mean().item())
            pressure_error = float(self.config.homeostasis_target_abs - state_abs)
            mean_state = float(region_states.mean().item())
            controller_bias = self.regions[name].homeostasis_controller[-1].bias
            controller_bias.data[0].add_(self.config.homeostasis_lr * pressure_error)
            controller_bias.data[1].add_(-0.5 * self.config.homeostasis_lr * mean_state)
            controller_bias.data[2].add_(-self.config.homeostasis_lr * pressure_error)

        return {
            "loss": float(
                F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    targets[:, 1:].reshape(-1),
                ).item()
            ),
            "avg_error_norm": float(mean_error_drive.norm(dim=-1).mean().item()),
            "replay_energy": float(out["replay_states"].norm(dim=-1).mean().item()),
        }

    @torch.no_grad()
    def replay_consolidate(self, input_ids: torch.Tensor) -> Dict[str, float]:
        out = self.forward(input_ids)
        replay_mean = out["replay_states"].mean(dim=(0, 1))
        pre_norm = float(self.slow_scaffold.norm().item())
        self.slow_scaffold.mul_(1.0 - self.config.consolidation_lr)
        self.slow_scaffold.add_(self.config.consolidation_lr * replay_mean)
        post_norm = float(self.slow_scaffold.norm().item())
        replay_gain = float((post_norm - pre_norm))
        return {
            "pre_scaffold_norm": pre_norm,
            "post_scaffold_norm": post_norm,
            "replay_gain": replay_gain,
            "replay_mean_norm": float(replay_mean.norm().item()),
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
                "bridge_weights": out["bridge_weights"][:, -1, :, :],
                "replay_state": out["replay_states"][:, -1, :],
                "syntax_local_mass": out["local_mass"]["syntax"][:, -1, :],
                "memory_local_mass": out["local_mass"]["memory"][:, -1, :],
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


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def estimate_parameter_count_analytic(config: SpikeICSPB3DMultiRegionConfig) -> int:
    h = int(config.region_hidden_dim)
    slots = int(config.patch_slots)
    phase = int(config.phase_dim)
    vocab = int(config.vocab_size)
    regions = int(len(config.region_names))

    patch_selector = 2 * slots * h + 2 * (2 * h * slots + slots)
    section_binder = (3 * h * h + h) + (h * h + h)
    successor_core = (
        phase * h
        + (2 * h * phase + phase)
        + (4 * h * h + h)
        + (h * h + h)
        + (2 * h * h + h)
        + (h + 1)
    )
    topology_projector = 5 * h + h
    topology_controller = (2 * h * h + h) + (2 * h + 2)
    homeostasis_controller = (3 * h * h + h) + (3 * h + 3)
    region_norm = 2 * h
    region_core = (
        patch_selector
        + section_binder
        + successor_core
        + topology_projector
        + topology_controller
        + homeostasis_controller
        + region_norm
    )

    token_embedding = vocab * h
    region_embeddings = regions * h
    region_bridge = regions * regions
    bridge_projector = 2 * h * h + h
    protocol_router = (3 * h * h + h) + (h * h + h)
    population_readout = h * vocab
    replay_head = h * h + h
    replay_gate = 2 * h + 1

    total = (
        token_embedding
        + region_embeddings
        + regions * region_core
        + region_bridge
        + bridge_projector
        + protocol_router
        + population_readout
        + replay_head
        + replay_gate
    )
    return int(total)


def estimate_scaling_profile(
    config: SpikeICSPB3DMultiRegionConfig,
    seq_len: int,
    batch_size: int,
    dtype_bytes: int = 2,
) -> Dict[str, float]:
    params = float(estimate_parameter_count_analytic(config))
    region_count = float(len(config.region_names))
    hidden = float(config.region_hidden_dim)
    patch_slots = float(config.patch_slots)
    bridge_topk = float(config.bridge_topk)

    token_activation = batch_size * seq_len * region_count * hidden
    patch_activation = batch_size * seq_len * region_count * patch_slots
    replay_activation = batch_size * seq_len * hidden
    activation_bytes = dtype_bytes * (token_activation + 0.35 * patch_activation + 0.4 * replay_activation)
    activation_mib = activation_bytes / (1024.0 ** 2)
    param_mib = params * dtype_bytes / (1024.0 ** 2)

    # Sparse local-topology and bridge routing stay linear in sequence length.
    sparse_topology_ops = batch_size * seq_len * region_count * (hidden * patch_slots + patch_slots * bridge_topk)
    dense_attention_reference = batch_size * region_count * (seq_len ** 2) * hidden
    sparse_to_dense_ratio = sparse_topology_ops / max(dense_attention_reference, 1.0)

    encoding_capacity = params * region_count * (1.0 + 0.15 * patch_slots / max(hidden, 1.0))

    return {
        "parameter_count": params,
        "parameter_mib_fp16": param_mib,
        "activation_mib_fp16": activation_mib,
        "estimated_step_work": float(sparse_topology_ops),
        "dense_attention_reference_work": float(dense_attention_reference),
        "sparse_to_dense_work_ratio": float(sparse_to_dense_ratio),
        "encoding_capacity_proxy": float(encoding_capacity),
    }

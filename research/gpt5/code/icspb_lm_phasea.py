from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ICSPBLMPhaseAConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    hidden_dim: int = 768
    num_layers: int = 8
    num_heads: int = 12
    ff_mult: int = 4
    dropout: float = 0.1
    memory_slots: int = 256
    online_rank: int = 64
    guarded_write_floor: float = 0.12
    stable_read_floor: float = 0.52
    theorem_margin_floor: float = 0.03


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ICSPBLMPhaseAConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        if self.head_dim * config.num_heads != config.hidden_dim:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.dropout(self.proj(attn))


class FeedForward(nn.Module):
    def __init__(self, config: ICSPBLMPhaseAConfig):
        super().__init__()
        inner = config.hidden_dim * config.ff_mult
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, inner),
            nn.GELU(),
            nn.Linear(inner, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ICSPBLMPhaseAConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class OnlineMemoryAdapter(nn.Module):
    def __init__(self, config: ICSPBLMPhaseAConfig):
        super().__init__()
        h = config.hidden_dim
        r = config.online_rank
        self.memory_keys = nn.Parameter(torch.randn(config.memory_slots, h) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(config.memory_slots, h) * 0.02)
        self.read_proj = nn.Sequential(
            nn.Linear(h, r),
            nn.GELU(),
            nn.Linear(r, h),
        )
        self.write_gate = nn.Sequential(
            nn.Linear(h + 2, r),
            nn.GELU(),
            nn.Linear(r, 1),
        )
        self.read_gate = nn.Sequential(
            nn.Linear(h + 2, r),
            nn.GELU(),
            nn.Linear(r, 1),
        )
        self.margin_head = nn.Sequential(
            nn.Linear(h, r),
            nn.GELU(),
            nn.Linear(r, 3),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        novelty: torch.Tensor | None = None,
        retention: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pooled = hidden_states.mean(dim=1)
        if novelty is None:
            novelty = torch.zeros(pooled.size(0), 1, device=pooled.device, dtype=pooled.dtype)
        if retention is None:
            retention = torch.ones(pooled.size(0), 1, device=pooled.device, dtype=pooled.dtype) * 0.5
        gate_input = torch.cat([pooled, novelty, retention], dim=-1)
        write_gate = torch.sigmoid(self.write_gate(gate_input))
        read_gate = torch.sigmoid(self.read_gate(gate_input))

        attn_scores = pooled @ self.memory_keys.t() / (pooled.size(-1) ** 0.5)
        attn = torch.softmax(attn_scores, dim=-1)
        read_vec = attn @ self.memory_values
        adapted = hidden_states + read_gate.unsqueeze(1) * self.read_proj(read_vec).unsqueeze(1)

        margins = self.margin_head(pooled)
        theorem_survival = torch.sigmoid(margins[:, 0:1])
        stable_read = torch.sigmoid(margins[:, 1:2])
        transport_margin = torch.tanh(margins[:, 2:3])
        metrics = {
            "write_gate": write_gate,
            "read_gate": read_gate,
            "theorem_survival": theorem_survival,
            "stable_read": stable_read,
            "transport_margin": transport_margin,
            "memory_attention": attn,
            "pooled_state": pooled,
        }
        return adapted, metrics


class ICSPBLMPhaseA(nn.Module):
    """
    Phase-A 目标：
    - 提供接近 100M 级别的正式 token-level 语言主干
    - 同时保留 ICSPB 的受控在线学习分支
    """

    def __init__(self, config: ICSPBLMPhaseAConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim

        self.token_embedding = nn.Embedding(config.vocab_size, h)
        self.position_embedding = nn.Embedding(config.max_seq_len, h)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(h)
        self.online_adapter = OnlineMemoryAdapter(config)
        self.lm_head = nn.Linear(h, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self._snapshot: Dict[str, Dict[str, torch.Tensor]] | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        novelty: torch.Tensor | None = None,
        retention: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("序列长度超过 max_seq_len")
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        x, adapter_metrics = self.online_adapter(x, novelty=novelty, retention=retention)
        logits = self.lm_head(x)

        out = {
            "logits": logits,
            "adapter_metrics": adapter_metrics,
        }
        if targets is not None:
            out["loss"] = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                targets[:, 1:].contiguous().view(-1),
            )
        return out

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.forward(
            input_ids=batch["input_ids"],
            targets=batch["input_ids"],
            novelty=batch.get("novelty"),
            retention=batch.get("retention"),
        )
        loss = out["loss"]
        metrics = out["adapter_metrics"]
        theorem_survival = float(metrics["theorem_survival"].mean().item())
        stable_read = float(metrics["stable_read"].mean().item())
        transport_margin = float(metrics["transport_margin"].abs().mean().item())
        return loss, {
            "loss": float(loss.item()),
            "theorem_survival": theorem_survival,
            "stable_read": stable_read,
            "transport_margin": transport_margin,
        }

    def online_update_step(
        self,
        batch: Dict[str, torch.Tensor],
        lr: float = 1e-3,
    ) -> Dict[str, float]:
        # 仅更新在线适配器，避免破坏语言主干。
        params = list(self.online_adapter.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.compute_loss(batch)
        loss.backward()
        optimizer.step()
        return metrics

    def snapshot(self) -> None:
        self._snapshot = {
            "online_adapter": {k: v.detach().clone() for k, v in self.online_adapter.state_dict().items()},
        }

    def rollback(self) -> None:
        if self._snapshot is None:
            return
        self.online_adapter.load_state_dict(self._snapshot["online_adapter"], strict=True)

    @staticmethod
    def decode_token_ids(token_ids: List[int] | torch.Tensor) -> str:
        if torch.is_tensor(token_ids):
            token_ids = token_ids.detach().cpu().tolist()
        return bytes([int(token) % 256 for token in token_ids]).decode("utf-8", errors="ignore")

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        novelty: torch.Tensor | None = None,
        retention: torch.Tensor | None = None,
        temperature: float = 1.0,
        vocab_limit: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        generated = input_ids.clone()
        last_metrics: Dict[str, torch.Tensor] = {}

        for _ in range(max_new_tokens):
            window = generated[:, -self.config.max_seq_len :]
            out = self.forward(
                input_ids=window,
                novelty=novelty,
                retention=retention,
            )
            logits = out["logits"][:, -1, :]
            if vocab_limit is not None:
                logits = logits[:, :vocab_limit]
            if temperature != 1.0:
                logits = logits / max(1e-6, temperature)
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            last_metrics = out["adapter_metrics"]

        return {
            "generated_ids": generated,
            "new_token_ids": generated[:, input_ids.size(1) :],
            "adapter_metrics": last_metrics,
        }


def make_phasea_batch(
    config: ICSPBLMPhaseAConfig,
    batch_size: int = 2,
    seq_len: int = 32,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), generator=g)
    novelty = torch.rand(batch_size, 1, generator=g)
    retention = torch.rand(batch_size, 1, generator=g)
    return {
        "input_ids": input_ids,
        "novelty": novelty,
        "retention": retention,
    }

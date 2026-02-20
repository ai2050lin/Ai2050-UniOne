import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vision_projector import VisionProjector


class FiberVisionEncoder(nn.Module):
    """Vision fiber: image -> shared latent vector."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.backbone = VisionProjector(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        z = self.backbone(images)
        return self.norm(z)


class FiberLanguageEncoder(nn.Module):
    """Language fiber: token sequence -> shared latent vector."""

    def __init__(self, vocab_size: int, d_model: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        x, _ = self.encoder(x)
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.norm(pooled)


class FiberConnector(nn.Module):
    """
    Connection module between fibers.
    Projects modality vectors into the same base-space and fuses them.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.vision_to_base = nn.Linear(d_model, d_model)
        self.language_to_base = nn.Linear(d_model, d_model)
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(
        self, vision_z: torch.Tensor, language_z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_base = F.normalize(self.vision_to_base(vision_z), dim=1)
        l_base = F.normalize(self.language_to_base(language_z), dim=1)
        gate = self.fusion_gate(torch.cat([v_base, l_base], dim=1))
        fused = F.normalize(gate * v_base + (1.0 - gate) * l_base, dim=1)
        return v_base, l_base, fused


class FiberMultimodalSystem(nn.Module):
    """Minimal combined system for train/infer."""

    def __init__(self, vocab_size: int, d_model: int = 128, num_classes: int = 10):
        super().__init__()
        self.vision = FiberVisionEncoder(d_model=d_model)
        self.language = FiberLanguageEncoder(vocab_size=vocab_size, d_model=d_model)
        self.connector = FiberConnector(d_model=d_model)
        self.vision_head = nn.Linear(d_model, num_classes)
        self.language_head = nn.Linear(d_model, num_classes)
        self.fused_head = nn.Linear(d_model, num_classes)

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        vision_z = self.vision(images)
        language_z = self.language(token_ids, attention_mask)
        v_base, l_base, fused = self.connector(vision_z, language_z)
        return {
            "vision_base": v_base,
            "language_base": l_base,
            "fused_base": fused,
            "vision_logits": self.vision_head(v_base),
            "language_logits": self.language_head(l_base),
            "fused_logits": self.fused_head(fused),
        }

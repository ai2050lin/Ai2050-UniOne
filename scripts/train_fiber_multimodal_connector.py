import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.fiber_multimodal import FiberMultimodalSystem
from server.runtime.contracts import (
    AnalysisSpec,
    ConclusionCard,
    Metric,
    RunRecord,
    RunSummary,
)
from server.runtime.experiment_store import ExperimentTimelineStore


DIGIT_TO_SEGMENTS = {
    0: ("A", "B", "C", "D", "E", "F"),
    1: ("B", "C"),
    2: ("A", "B", "G", "E", "D"),
    3: ("A", "B", "C", "D", "G"),
    4: ("F", "G", "B", "C"),
    5: ("A", "F", "G", "C", "D"),
    6: ("A", "F", "E", "D", "C", "G"),
    7: ("A", "B", "C"),
    8: ("A", "B", "C", "D", "E", "F", "G"),
    9: ("A", "B", "C", "D", "F", "G"),
}

BASE_SEGMENTS = {
    "A": (6, 3, 21, 5),
    "B": (20, 6, 22, 13),
    "C": (20, 14, 22, 21),
    "D": (6, 22, 21, 24),
    "E": (5, 14, 7, 21),
    "F": (5, 6, 7, 13),
    "G": (6, 12, 21, 15),
}

VOCAB = ["<PAD>", "<BOS>", "<EOS>"] + [f"SYM_{i}" for i in range(10)]
TOKEN_TO_ID = {t: i for i, t in enumerate(VOCAB)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _expand_rect(rect: Tuple[int, int, int, int], thickness: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    if (x2 - x1) >= (y2 - y1):
        return x1, y1 - thickness, x2, y2 + thickness
    return x1 - thickness, y1, x2 + thickness, y2


def render_synthetic_digit(label: int, rng: np.random.Generator) -> np.ndarray:
    canvas = np.zeros((28, 28), dtype=np.float32)
    dx = int(rng.integers(-2, 3))
    dy = int(rng.integers(-2, 3))
    thickness = int(rng.integers(0, 2))
    intensity = float(rng.uniform(0.8, 1.2))
    dropout_p = float(rng.uniform(0.0, 0.08))

    for seg in DIGIT_TO_SEGMENTS[label]:
        if rng.random() < dropout_p:
            continue
        x1, y1, x2, y2 = _expand_rect(BASE_SEGMENTS[seg], thickness)
        x1 += dx
        x2 += dx
        y1 += dy
        y2 += dy
        x1 = max(0, min(27, x1))
        x2 = max(0, min(27, x2))
        y1 = max(0, min(27, y1))
        y2 = max(0, min(27, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        canvas[y1:y2, x1:x2] = intensity

    noise_std = float(rng.uniform(0.03, 0.10))
    canvas += rng.normal(0.0, noise_std, size=canvas.shape).astype(np.float32)
    canvas = np.clip(canvas, 0.0, 1.0)
    canvas = (canvas - 0.1307) / 0.3081
    return canvas


def make_token_sequence(label: int) -> Tuple[np.ndarray, np.ndarray]:
    tokens = np.array(
        [TOKEN_TO_ID["<BOS>"], TOKEN_TO_ID[f"SYM_{label}"], TOKEN_TO_ID["<EOS>"]],
        dtype=np.int64,
    )
    mask = np.ones_like(tokens, dtype=np.int64)
    return tokens, mask


class SyntheticFiberMultimodalDataset(Dataset):
    def __init__(self, total_samples: int, seed: int):
        self.total_samples = int(total_samples)
        self.seed = int(seed)
        self.images = np.zeros((self.total_samples, 1, 28, 28), dtype=np.float32)
        self.labels = np.zeros((self.total_samples,), dtype=np.int64)
        self.tokens = np.zeros((self.total_samples, 3), dtype=np.int64)
        self.masks = np.zeros((self.total_samples, 3), dtype=np.int64)
        self._build()

    def _build(self) -> None:
        rng = np.random.default_rng(self.seed)
        for i in range(self.total_samples):
            label = int(rng.integers(0, 10))
            self.labels[i] = label
            self.images[i, 0] = render_synthetic_digit(label, rng)
            token_seq, mask = make_token_sequence(label)
            self.tokens[i] = token_seq
            self.masks[i] = mask

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image": torch.from_numpy(self.images[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "tokens": torch.from_numpy(self.tokens[idx]),
            "mask": torch.from_numpy(self.masks[idx]),
        }


class MNISTFiberMultimodalDataset(Dataset):
    def __init__(self, root: str, total_samples: int, seed: int, download: bool = False):
        from torchvision import datasets, transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_set = datasets.MNIST(root=root, train=True, download=download, transform=transform)
        test_set = datasets.MNIST(root=root, train=False, download=download, transform=transform)
        full_set = ConcatDataset([train_set, test_set])
        sample_count = min(int(total_samples), len(full_set))
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(full_set), size=sample_count, replace=False)
        self.base = full_set
        self.indices = indices.tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image, label = self.base[self.indices[idx]]
        token_seq, mask = make_token_sequence(int(label))
        return {
            "image": image,
            "label": torch.tensor(int(label), dtype=torch.long),
            "tokens": torch.from_numpy(token_seq),
            "mask": torch.from_numpy(mask),
        }


def build_dataset(
    dataset_name: str,
    total_samples: int,
    seed: int,
    mnist_root: str,
    mnist_download: bool,
) -> Tuple[Dataset, str]:
    mode = (dataset_name or "synthetic").lower().strip()
    if mode not in {"synthetic", "mnist", "auto"}:
        raise ValueError(f"Unsupported dataset mode: {dataset_name}")

    if mode in {"mnist", "auto"}:
        try:
            dataset = MNISTFiberMultimodalDataset(
                root=mnist_root,
                total_samples=total_samples,
                seed=seed,
                download=mnist_download if mode == "mnist" else False,
            )
            return dataset, "mnist"
        except Exception:
            if mode == "mnist":
                raise

    return SyntheticFiberMultimodalDataset(total_samples=total_samples, seed=seed), "synthetic"


@dataclass
class EpochStat:
    epoch: int
    train_loss: float
    train_vision_acc: float
    train_language_acc: float
    train_fused_acc: float
    val_loss: float
    val_vision_acc: float
    val_language_acc: float
    val_fused_acc: float
    val_retrieval_top1: float
    val_alignment_cos: float
    val_smoothness: float
    val_curvature: float


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _supervised_contrastive_from_logits(logits: torch.Tensor, label_a: torch.Tensor, label_b: torch.Tensor) -> torch.Tensor:
    mask = (label_a[:, None] == label_b[None, :]).to(logits.dtype)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = mask.sum(dim=1).clamp_min(1.0)
    return -((mask * log_prob).sum(dim=1) / pos_count).mean()


def contrastive_loss(
    vision_base: torch.Tensor,
    language_base: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    sim = vision_base @ language_base.T / temperature
    loss_v2l = _supervised_contrastive_from_logits(sim, labels, labels)
    loss_l2v = _supervised_contrastive_from_logits(sim.T, labels, labels)
    return 0.5 * (loss_v2l + loss_l2v)


def connection_smoothness_loss(
    fused_base: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    pairwise = torch.cdist(fused_base, fused_base, p=2)
    eye_mask = torch.eye(labels.size(0), device=labels.device, dtype=torch.bool)
    same_raw = labels[:, None] == labels[None, :]
    same_mask = same_raw & (~eye_mask)
    diff_mask = (~same_raw) & (~eye_mask)

    if same_mask.any():
        same_loss = pairwise[same_mask].pow(2).mean()
    else:
        same_loss = torch.tensor(0.0, device=fused_base.device)

    if diff_mask.any():
        diff_loss = F.relu(margin - pairwise[diff_mask]).pow(2).mean()
    else:
        diff_loss = torch.tensor(0.0, device=fused_base.device)

    return same_loss + diff_loss


def curvature_regularization(
    vision_base: torch.Tensor,
    fused_base: torch.Tensor,
    language_base: torch.Tensor,
) -> torch.Tensor:
    # Curvature surrogate on the connection path V -> F -> L.
    bend = vision_base - 2.0 * fused_base + language_base
    return bend.pow(2).sum(dim=1).mean()


def evaluate(
    model: FiberMultimodalSystem,
    loader: DataLoader,
    device: torch.device,
    weights: Dict[str, float],
    temperature: float,
    smooth_margin: float,
) -> Dict[str, float]:
    model.eval()
    total = 0
    total_loss = 0.0
    correct_v = 0
    correct_l = 0
    correct_f = 0
    all_v = []
    all_l = []
    all_labels = []
    smoothness_accum = 0.0
    curvature_accum = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            out = model(batch["image"], batch["tokens"], batch["mask"])
            labels = batch["label"]

            cls_v = F.cross_entropy(out["vision_logits"], labels)
            cls_l = F.cross_entropy(out["language_logits"], labels)
            cls_f = F.cross_entropy(out["fused_logits"], labels)
            ctr = contrastive_loss(
                out["vision_base"],
                out["language_base"],
                labels=labels,
                temperature=temperature,
            )
            align = F.mse_loss(out["vision_base"], out["language_base"])
            smooth = connection_smoothness_loss(
                out["fused_base"],
                labels=labels,
                margin=smooth_margin,
            )
            curve = curvature_regularization(
                out["vision_base"],
                out["fused_base"],
                out["language_base"],
            )

            loss = (
                weights["vision_cls"] * cls_v
                + weights["language_cls"] * cls_l
                + weights["fused_cls"] * cls_f
                + weights["contrastive"] * ctr
                + weights["alignment"] * align
                + weights["smoothness"] * smooth
                + weights["curvature"] * curve
            )

            bsz = labels.size(0)
            total += bsz
            total_loss += float(loss.item()) * bsz
            correct_v += int((out["vision_logits"].argmax(dim=1) == labels).sum().item())
            correct_l += int((out["language_logits"].argmax(dim=1) == labels).sum().item())
            correct_f += int((out["fused_logits"].argmax(dim=1) == labels).sum().item())
            smoothness_accum += float(smooth.item()) * bsz
            curvature_accum += float(curve.item()) * bsz

            all_v.append(out["vision_base"].detach().cpu())
            all_l.append(out["language_base"].detach().cpu())
            all_labels.append(labels.detach().cpu())

    if total == 0:
        return {
            "loss": 0.0,
            "vision_acc": 0.0,
            "language_acc": 0.0,
            "fused_acc": 0.0,
            "retrieval_top1": 0.0,
            "alignment_cos": 0.0,
            "smoothness": 0.0,
            "curvature": 0.0,
        }

    vision_all = torch.cat(all_v, dim=0)
    lang_all = torch.cat(all_l, dim=0)
    labels_all = torch.cat(all_labels, dim=0)
    sim = vision_all @ lang_all.T
    v2l_idx = sim.argmax(dim=1)
    l2v_idx = sim.argmax(dim=0)
    v2l_label_match = (labels_all[v2l_idx] == labels_all).float().mean()
    l2v_label_match = (labels_all[l2v_idx] == labels_all).float().mean()
    retrieval_top1 = float(0.5 * (v2l_label_match + l2v_label_match))
    alignment_cos = float((vision_all * lang_all).sum(dim=1).mean().item())

    return {
        "loss": total_loss / total,
        "vision_acc": correct_v / total,
        "language_acc": correct_l / total,
        "fused_acc": correct_f / total,
        "retrieval_top1": retrieval_top1,
        "alignment_cos": alignment_cos,
        "smoothness": smoothness_accum / total,
        "curvature": curvature_accum / total,
    }


def maybe_save_projection_plot(
    model: FiberMultimodalSystem,
    loader: DataLoader,
    device: torch.device,
    out_path: str,
    max_points: int = 1000,
) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except Exception:
        return None

    model.eval()
    v_list = []
    l_list = []
    labels_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            out = model(batch["image"], batch["tokens"], batch["mask"])
            v_list.append(out["vision_base"].detach().cpu().numpy())
            l_list.append(out["language_base"].detach().cpu().numpy())
            labels_list.append(batch["label"].detach().cpu().numpy())
            if sum(len(x) for x in labels_list) >= max_points:
                break
    if not v_list:
        return None

    v = np.concatenate(v_list, axis=0)[:max_points]
    l = np.concatenate(l_list, axis=0)[:max_points]
    labels = np.concatenate(labels_list, axis=0)[:max_points]
    merged = np.vstack([v, l])
    pca = PCA(n_components=2)
    proj = pca.fit_transform(merged)
    p_v = proj[: len(v)]
    p_l = proj[len(v) :]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")
    for d in range(10):
        mask = labels == d
        plt.scatter(p_v[mask, 0], p_v[mask, 1], s=8, alpha=0.35, c=[cmap(d)])
        plt.scatter(p_l[mask, 0], p_l[mask, 1], s=10, alpha=0.35, marker="x", c=[cmap(d)])
    plt.title("Fiber Multimodal Projection (dot=vision, x=language)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path


def write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_md(path: str, report: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    best = report["summary"]["best"]
    lines = [
        "# Fiber Multimodal Connector Report",
        "",
        f"- Generated At: {report['meta']['generated_at']}",
        f"- Device: {report['meta']['device']}",
        f"- Dataset: {report['meta'].get('dataset', '-')}",
        f"- Train Size: {report['meta']['train_size']}",
        f"- Val Size: {report['meta']['val_size']}",
        "",
        "## Best Validation",
        f"- Epoch: {best['epoch']}",
        f"- Fused Acc: {best['val_fused_acc']:.6f}",
        f"- Retrieval Top1: {best['val_retrieval_top1']:.6f}",
        f"- Alignment Cos: {best['val_alignment_cos']:.6f}",
        "",
        "## Epoch Table",
        "| Epoch | Train Loss | Train V Acc | Train L Acc | Train F Acc | Val Loss | Val V Acc | Val L Acc | Val F Acc | Val Ret@1 | Val Align Cos | Val Smooth | Val Curvature |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["history"]:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.6f} | {row['train_vision_acc']:.6f} | "
            f"{row['train_language_acc']:.6f} | {row['train_fused_acc']:.6f} | {row['val_loss']:.6f} | "
            f"{row['val_vision_acc']:.6f} | {row['val_language_acc']:.6f} | {row['val_fused_acc']:.6f} | "
            f"{row['val_retrieval_top1']:.6f} | {row['val_alignment_cos']:.6f} | "
            f"{row['val_smoothness']:.6f} | {row['val_curvature']:.6f} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def append_timeline(
    timeline_path: str,
    route: str,
    analysis_type: str,
    model_name: str,
    config: Dict,
    best: Dict,
    elapsed: float,
    artifacts: List[Dict],
) -> str:
    store = ExperimentTimelineStore(path=timeline_path)
    created_at = time.time()
    run_id = f"run_fm_{int(created_at)}_{random.randint(1000, 9999)}"
    record = RunRecord(
        run_id=run_id,
        spec=AnalysisSpec(
            route=route,
            analysis_type=analysis_type,
            model=model_name,
            params=config,
            input_payload={},
        ),
        status="completed",
        created_at=created_at,
        updated_at=created_at + 0.001,
        event_count=0,
        summary=RunSummary(
            metrics=[
                Metric(key="val_fused_accuracy", value=float(best["val_fused_acc"]), min_value=0.0, max_value=1.0),
                Metric(key="val_retrieval_top1", value=float(best["val_retrieval_top1"]), min_value=0.0, max_value=1.0),
                Metric(key="val_alignment_cos", value=float(best["val_alignment_cos"]), min_value=-1.0, max_value=1.0),
                Metric(key="val_smoothness", value=float(best["val_smoothness"]), min_value=0.0),
                Metric(key="val_curvature", value=float(best["val_curvature"]), min_value=0.0),
                Metric(key="train_seconds", value=float(elapsed), min_value=0.0),
            ],
            conclusion=ConclusionCard(
                objective="Train separate visual/language fibers and connect them in a shared base space.",
                method=(
                    "Jointly optimize classification, contrastive alignment, "
                    "connection smoothness, and curvature regularization."
                ),
                evidence=[
                    f"dataset={config.get('dataset_selected')}",
                    f"val_fused_acc={best['val_fused_acc']:.4f}",
                    f"val_retrieval_top1={best['val_retrieval_top1']:.4f}",
                    f"val_alignment_cos={best['val_alignment_cos']:.4f}",
                    f"val_smoothness={best['val_smoothness']:.4f}",
                    f"val_curvature={best['val_curvature']:.4f}",
                ],
                result=(
                    "Fiber multimodal connector training completed with synchronized vision-language embeddings."
                ),
                confidence=max(0.35, min(0.95, 0.25 + 0.5 * best["val_fused_acc"] + 0.25 * best["val_retrieval_top1"])),
                limitations=[
                    "Current experiment uses synthetic symbol images and short symbolic text.",
                    "No large-scale real-world corpus or long-context language yet.",
                ],
                next_action="Scale to real image-text pairs and add geometry-level connection constraints.",
            ),
            artifacts=artifacts,
        ),
    )
    store.append_run(record)
    return run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FiberVision + FiberLanguage + Connector.")
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "mnist", "auto"])
    parser.add_argument("--mnist-root", default="tempdata/data")
    parser.add_argument("--mnist-download", action="store_true")
    parser.add_argument("--total-samples", type=int, default=30000)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--w-vision-cls", type=float, default=1.0)
    parser.add_argument("--w-language-cls", type=float, default=1.0)
    parser.add_argument("--w-fused-cls", type=float, default=1.2)
    parser.add_argument("--w-contrastive", type=float, default=0.1)
    parser.add_argument("--w-alignment", type=float, default=0.1)
    parser.add_argument("--w-smoothness", type=float, default=0.05)
    parser.add_argument("--w-curvature", type=float, default=0.05)
    parser.add_argument("--smooth-margin", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--weights-out", default="tempdata/fiber_multimodal_connector.pth")
    parser.add_argument("--plot-out", default="tempdata/fiber_multimodal_projection.png")
    parser.add_argument("--report-json", default="tempdata/fiber_multimodal_report.json")
    parser.add_argument("--report-md", default="tempdata/fiber_multimodal_report.md")

    parser.add_argument("--timeline", default="tempdata/agi_route_test_timeline.json")
    parser.add_argument("--route", default="fiber_bundle")
    parser.add_argument("--analysis-type", default="multimodal_connector")
    parser.add_argument("--skip-timeline", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> Dict:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    full_dataset, dataset_selected = build_dataset(
        dataset_name=args.dataset,
        total_samples=args.total_samples,
        seed=args.seed,
        mnist_root=args.mnist_root,
        mnist_download=args.mnist_download,
    )
    dataset_size = len(full_dataset)
    val_size = max(100, int(dataset_size * args.val_ratio))
    val_size = min(val_size, max(1, dataset_size - 1))
    train_size = dataset_size - val_size
    split_gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=split_gen)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FiberMultimodalSystem(vocab_size=len(VOCAB), d_model=args.d_model, num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    weights = {
        "vision_cls": args.w_vision_cls,
        "language_cls": args.w_language_cls,
        "fused_cls": args.w_fused_cls,
        "contrastive": args.w_contrastive,
        "alignment": args.w_alignment,
        "smoothness": args.w_smoothness,
        "curvature": args.w_curvature,
    }

    history: List[EpochStat] = []
    best = {
        "epoch": 0,
        "val_fused_acc": 0.0,
        "val_retrieval_top1": 0.0,
        "val_alignment_cos": -1.0,
        "val_smoothness": float("inf"),
        "val_curvature": float("inf"),
        "val_loss": float("inf"),
    }

    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        total_loss = 0.0
        correct_v = 0
        correct_l = 0
        correct_f = 0

        for batch in train_loader:
            batch = batch_to_device(batch, device)
            out = model(batch["image"], batch["tokens"], batch["mask"])
            labels = batch["label"]

            cls_v = F.cross_entropy(out["vision_logits"], labels)
            cls_l = F.cross_entropy(out["language_logits"], labels)
            cls_f = F.cross_entropy(out["fused_logits"], labels)
            ctr = contrastive_loss(
                out["vision_base"],
                out["language_base"],
                labels=labels,
                temperature=args.temperature,
            )
            align = F.mse_loss(out["vision_base"], out["language_base"])
            smooth = connection_smoothness_loss(
                out["fused_base"],
                labels=labels,
                margin=args.smooth_margin,
            )
            curve = curvature_regularization(
                out["vision_base"],
                out["fused_base"],
                out["language_base"],
            )
            loss = (
                weights["vision_cls"] * cls_v
                + weights["language_cls"] * cls_l
                + weights["fused_cls"] * cls_f
                + weights["contrastive"] * ctr
                + weights["alignment"] * align
                + weights["smoothness"] * smooth
                + weights["curvature"] * curve
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bsz = labels.size(0)
            total += bsz
            total_loss += float(loss.item()) * bsz
            correct_v += int((out["vision_logits"].argmax(dim=1) == labels).sum().item())
            correct_l += int((out["language_logits"].argmax(dim=1) == labels).sum().item())
            correct_f += int((out["fused_logits"].argmax(dim=1) == labels).sum().item())

        train_loss = total_loss / max(1, total)
        train_v = correct_v / max(1, total)
        train_l = correct_l / max(1, total)
        train_f = correct_f / max(1, total)

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            weights=weights,
            temperature=args.temperature,
            smooth_margin=args.smooth_margin,
        )

        stat = EpochStat(
            epoch=epoch,
            train_loss=train_loss,
            train_vision_acc=train_v,
            train_language_acc=train_l,
            train_fused_acc=train_f,
            val_loss=val_metrics["loss"],
            val_vision_acc=val_metrics["vision_acc"],
            val_language_acc=val_metrics["language_acc"],
            val_fused_acc=val_metrics["fused_acc"],
            val_retrieval_top1=val_metrics["retrieval_top1"],
            val_alignment_cos=val_metrics["alignment_cos"],
            val_smoothness=val_metrics["smoothness"],
            val_curvature=val_metrics["curvature"],
        )
        history.append(stat)

        if (
            stat.val_fused_acc > best["val_fused_acc"]
            or (
                stat.val_fused_acc == best["val_fused_acc"]
                and stat.val_retrieval_top1 > best["val_retrieval_top1"]
            )
        ):
            best = {
                "epoch": epoch,
                "val_fused_acc": stat.val_fused_acc,
                "val_retrieval_top1": stat.val_retrieval_top1,
                "val_alignment_cos": stat.val_alignment_cos,
                "val_smoothness": stat.val_smoothness,
                "val_curvature": stat.val_curvature,
                "val_loss": stat.val_loss,
            }

        print(
            f"[epoch {epoch:02d}] train_f={train_f:.4f} val_f={stat.val_fused_acc:.4f} "
            f"val_ret@1={stat.val_retrieval_top1:.4f} val_align={stat.val_alignment_cos:.4f} "
            f"val_s={stat.val_smoothness:.4f} val_k={stat.val_curvature:.4f}"
        )

    elapsed = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.weights_out) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "vocab": VOCAB,
        },
        args.weights_out,
    )

    artifacts = [{"type": "weights", "path": args.weights_out}]
    plot_path = maybe_save_projection_plot(
        model=model,
        loader=val_loader,
        device=device,
        out_path=args.plot_out,
    )
    if plot_path:
        artifacts.append({"type": "plot", "path": plot_path})

    config_dict = dict(vars(args))
    config_dict["dataset_selected"] = dataset_selected
    config_dict["dataset_size"] = dataset_size
    config_dict["train_size"] = train_size
    config_dict["val_size"] = val_size

    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "dataset": dataset_selected,
            "dataset_size": dataset_size,
            "train_size": train_size,
            "val_size": val_size,
        },
        "config": config_dict,
        "summary": {
            "best": best,
            "elapsed_seconds": float(elapsed),
            "artifacts": artifacts,
        },
        "history": [asdict(x) for x in history],
    }
    write_json(args.report_json, report)
    write_md(args.report_md, report)

    timeline_run_id = None
    if not args.skip_timeline:
        timeline_run_id = append_timeline(
            timeline_path=args.timeline,
            route=args.route,
            analysis_type=args.analysis_type,
            model_name=f"FiberMultimodalSystem(d_model={args.d_model})",
            config=report["config"],
            best=best,
            elapsed=elapsed,
            artifacts=artifacts,
        )

    return {
        "weights": args.weights_out,
        "report_json": args.report_json,
        "report_md": args.report_md,
        "best": best,
        "timeline_run_id": timeline_run_id,
    }


def main() -> None:
    args = parse_args()
    result = train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

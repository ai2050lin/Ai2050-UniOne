import argparse
import json
import math
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
from torch.utils.data import DataLoader, Dataset, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.vision_projector import create_vision_model
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


class SyntheticDigitsDataset(Dataset):
    def __init__(self, total_samples: int, seed: int) -> None:
        self.total_samples = int(total_samples)
        self.seed = int(seed)
        self.images = np.zeros((self.total_samples, 1, 28, 28), dtype=np.float32)
        self.labels = np.zeros((self.total_samples,), dtype=np.int64)
        self._build()

    def _build(self) -> None:
        rng = np.random.default_rng(self.seed)
        for idx in range(self.total_samples):
            label = int(rng.integers(0, 10))
            self.labels[idx] = label
            self.images[idx, 0] = render_synthetic_digit(label, rng)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


@dataclass
class EpochStat:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_anchor_cos: float


def _make_anchor_bank(
    d_model: int,
    device: torch.device,
    seed: int,
    logic_model_path: str,
    logic_corpus_path: str,
    anchor_source: str,
) -> Tuple[torch.Tensor, str]:
    if anchor_source in {"auto", "logic"}:
        try:
            from models.fibernet_logic import create_logic_model
            from scripts.train_logic_core import LogicCorpus

            if os.path.exists(logic_model_path) and os.path.exists(logic_corpus_path):
                corpus = LogicCorpus(logic_corpus_path)
                model = create_logic_model(
                    vocab_size=corpus.vocab_size,
                    d_model=d_model,
                    n_layers=4,
                    n_heads=4,
                ).model
                state = torch.load(logic_model_path, map_location="cpu")
                model.load_state_dict(state, strict=False)
                model.eval()
                anchors = []
                with torch.no_grad():
                    for digit in range(10):
                        token = f"SYM_{digit}"
                        if token not in corpus.token_to_id:
                            raise ValueError(f"missing token: {token}")
                        idx = corpus.token_to_id[token]
                        anchors.append(model.embed.W_E[idx].detach().cpu())
                anchor_tensor = torch.stack(anchors, dim=0).to(device)
                return anchor_tensor, "logic"
        except Exception:
            if anchor_source == "logic":
                raise

    generator = torch.Generator(device="cpu").manual_seed(seed + 1337)
    anchor_tensor = torch.randn((10, d_model), generator=generator)
    anchor_tensor = F.normalize(anchor_tensor, dim=1).to(device) * math.sqrt(d_model)
    return anchor_tensor, "random"


def _build_dataloaders(
    dataset_name: str,
    total_samples: int,
    val_ratio: float,
    batch_size: int,
    seed: int,
    mnist_root: str,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], str]:
    dataset_name = dataset_name.lower().strip()
    if dataset_name not in {"synthetic", "mnist", "auto"}:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    selected = dataset_name
    dataset: Dataset

    if dataset_name in {"mnist", "auto"}:
        try:
            from torchvision import datasets, transforms

            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            train_data = datasets.MNIST(
                mnist_root,
                train=True,
                download=(dataset_name == "mnist"),
                transform=transform,
            )
            test_data = datasets.MNIST(
                mnist_root,
                train=False,
                download=(dataset_name == "mnist"),
                transform=transform,
            )
            full_len = min(total_samples, len(train_data) + len(test_data))
            if full_len <= len(train_data):
                dataset = torch.utils.data.Subset(train_data, list(range(full_len)))
            else:
                extra = full_len - len(train_data)
                dataset = torch.utils.data.ConcatDataset(
                    [train_data, torch.utils.data.Subset(test_data, list(range(extra)))]
                )
            selected = "mnist"
        except Exception:
            if dataset_name == "mnist":
                raise
            selected = "synthetic"
            dataset = SyntheticDigitsDataset(total_samples=total_samples, seed=seed)
    else:
        dataset = SyntheticDigitsDataset(total_samples=total_samples, seed=seed)

    val_size = max(100, int(len(dataset) * val_ratio))
    train_size = max(1, len(dataset) - val_size)
    if train_size <= 0:
        raise ValueError("Dataset too small after split.")

    split_generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=split_generator)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    meta = {"train_size": train_size, "val_size": val_size, "total_samples": len(dataset)}
    return train_loader, val_loader, meta, selected


def _epoch_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    anchors: torch.Tensor,
    align_weight: float,
    temperature: float,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    anchor_cos_accum = 0.0
    anchors_norm = F.normalize(anchors, dim=1)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            logits = F.normalize(outputs, dim=1) @ anchors_norm.T / temperature
            ce_loss = F.cross_entropy(logits, labels)
            align_loss = F.mse_loss(F.normalize(outputs, dim=1), anchors_norm[labels])
            loss = ce_loss + align_weight * align_loss
            total_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
            anchor_cos_accum += float(
                (F.normalize(outputs, dim=1) * anchors_norm[labels]).sum(dim=1).mean().item()
            ) * labels.size(0)

    if total == 0:
        return 0.0, 0.0, 0.0
    return total_loss / total, correct / total, anchor_cos_accum / total


def _safe_float(value: float) -> float:
    return float(round(float(value), 6))


def _generate_projection_artifact(
    model: torch.nn.Module,
    val_loader: DataLoader,
    anchors: torch.Tensor,
    output_path: str,
    device: torch.device,
    max_points: int = 1200,
) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except Exception:
        return None

    model.eval()
    feats = []
    labels_all = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images).detach().cpu().numpy()
            feats.append(outputs)
            labels_all.append(labels.numpy())
            if sum(len(x) for x in labels_all) >= max_points:
                break
    if not feats:
        return None

    feats = np.concatenate(feats, axis=0)[:max_points]
    labels_np = np.concatenate(labels_all, axis=0)[:max_points]
    anchor_np = anchors.detach().cpu().numpy()
    merged = np.vstack([anchor_np, feats])
    pca = PCA(n_components=2)
    projected = pca.fit_transform(merged)
    anchor_2d = projected[:10]
    feat_2d = projected[10:]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")
    for digit in range(10):
        mask = labels_np == digit
        plt.scatter(
            feat_2d[mask, 0],
            feat_2d[mask, 1],
            s=8,
            alpha=0.35,
            c=[cmap(digit)],
        )
        plt.scatter(
            anchor_2d[digit, 0],
            anchor_2d[digit, 1],
            marker="*",
            s=230,
            edgecolors="black",
            c=[cmap(digit)],
        )
    plt.title("Vision Alignment Projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()
    return output_path


def _write_report_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_report_md(path: str, report: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = []
    lines.append("# Vision Training Report")
    lines.append("")
    lines.append(f"- Generated At: {report['meta']['generated_at']}")
    lines.append(f"- Device: {report['meta']['device']}")
    lines.append(f"- Dataset: {report['meta']['dataset']}")
    lines.append(f"- Anchor Source: {report['meta']['anchor_source']}")
    lines.append("")
    best = report["summary"]["best"]
    lines.append("## Best Validation")
    lines.append(f"- Epoch: {best['epoch']}")
    lines.append(f"- Val Accuracy: {best['val_acc']:.6f}")
    lines.append(f"- Val Loss: {best['val_loss']:.6f}")
    lines.append(f"- Val Anchor Cosine: {best['val_anchor_cos']:.6f}")
    lines.append("")
    lines.append("## Epoch Table")
    lines.append("| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Anchor Cos |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in report["history"]:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.6f} | {row['train_acc']:.6f} | "
            f"{row['val_loss']:.6f} | {row['val_acc']:.6f} | {row['val_anchor_cos']:.6f} |"
        )
    lines.append("")
    lines.append("## Config")
    for k, v in report["config"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _timeline_append(
    timeline_path: str,
    route: str,
    analysis_type: str,
    model_name: str,
    config: Dict,
    best: Dict,
    elapsed_seconds: float,
    artifacts: List[Dict],
) -> Dict:
    store = ExperimentTimelineStore(path=timeline_path)
    created_at = time.time()
    run_id = f"run_vis_{int(created_at)}_{random.randint(1000, 9999)}"
    summary = RunSummary(
        metrics=[
            Metric(
                key="val_accuracy",
                value=float(best["val_acc"]),
                min_value=0.0,
                max_value=1.0,
                description="Validation classification accuracy against anchor labels.",
            ),
            Metric(
                key="val_loss",
                value=float(best["val_loss"]),
                min_value=0.0,
                description="Validation objective loss.",
            ),
            Metric(
                key="val_anchor_cosine",
                value=float(best["val_anchor_cos"]),
                min_value=-1.0,
                max_value=1.0,
                description="Cosine similarity between projections and target anchors.",
            ),
            Metric(
                key="train_seconds",
                value=float(elapsed_seconds),
                min_value=0.0,
                description="Wall clock time for training.",
            ),
        ],
        conclusion=ConclusionCard(
            objective="Train a vision encoder that maps images into AGI route anchor manifold.",
            method="Optimize VisionProjector with classification + anchor alignment losses.",
            evidence=[
                f"dataset={config.get('dataset_selected')}",
                f"epochs={config.get('epochs')}",
                f"val_acc={best['val_acc']:.4f}",
                f"val_anchor_cos={best['val_anchor_cos']:.4f}",
            ],
            result=(
                "Vision alignment training finished. "
                f"Best val_acc={best['val_acc']:.4f}, "
                f"anchor_cos={best['val_anchor_cos']:.4f}."
            ),
            confidence=max(0.35, min(0.95, 0.25 + best["val_acc"] * 0.7)),
            limitations=[
                "Current benchmark focuses on digit-style visual symbols.",
                "Cross-domain real-image transfer is not included in this run.",
            ],
            next_action="Scale to larger visual corpora and add multilingual semantic grounding.",
        ),
        artifacts=artifacts,
    )

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
        summary=summary,
        event_count=0,
    )
    entry = store.append_run(record)
    return {"run_id": run_id, "entry": entry}


def train(args: argparse.Namespace) -> Dict:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader, val_loader, data_meta, dataset_selected = _build_dataloaders(
        dataset_name=args.dataset,
        total_samples=args.total_samples,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        seed=args.seed,
        mnist_root=args.mnist_root,
    )
    anchors, anchor_source = _make_anchor_bank(
        d_model=args.d_model,
        device=device,
        seed=args.seed,
        logic_model_path=args.logic_model_path,
        logic_corpus_path=args.logic_corpus_path,
        anchor_source=args.anchor_source,
    )
    anchors_norm = F.normalize(anchors, dim=1)

    model = create_vision_model(d_model=args.d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[EpochStat] = []
    best = {
        "epoch": 0,
        "val_loss": float("inf"),
        "val_acc": 0.0,
        "val_anchor_cos": -1.0,
    }

    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            logits = F.normalize(outputs, dim=1) @ anchors_norm.T / args.temperature
            ce_loss = F.cross_entropy(logits, labels)
            align_loss = F.mse_loss(F.normalize(outputs, dim=1), anchors_norm[labels])
            loss = ce_loss + args.align_weight * align_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_loss, val_acc, val_anchor_cos = _epoch_eval(
            model=model,
            loader=val_loader,
            anchors=anchors,
            align_weight=args.align_weight,
            temperature=args.temperature,
            device=device,
        )

        if val_acc > best["val_acc"] or (val_acc == best["val_acc"] and val_loss < best["val_loss"]):
            best.update(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_anchor_cos": val_anchor_cos,
                }
            )

        history.append(
            EpochStat(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                val_anchor_cos=val_anchor_cos,
            )
        )
        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_anchor_cos={val_anchor_cos:.4f}"
        )

    elapsed_seconds = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.weights_out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.weights_out)

    artifacts = [{"type": "weights", "path": args.weights_out}]
    projection_path = _generate_projection_artifact(
        model=model,
        val_loader=val_loader,
        anchors=anchors,
        output_path=args.projection_out,
        device=device,
        max_points=args.projection_points,
    )
    if projection_path:
        artifacts.append({"type": "plot", "path": projection_path})

    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "dataset": dataset_selected,
            "anchor_source": anchor_source,
        },
        "config": {
            "dataset_requested": args.dataset,
            "dataset_selected": dataset_selected,
            "total_samples": data_meta["total_samples"],
            "train_size": data_meta["train_size"],
            "val_size": data_meta["val_size"],
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "d_model": args.d_model,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "align_weight": args.align_weight,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "summary": {
            "best": {
                "epoch": int(best["epoch"]),
                "val_loss": _safe_float(best["val_loss"]),
                "val_acc": _safe_float(best["val_acc"]),
                "val_anchor_cos": _safe_float(best["val_anchor_cos"]),
            },
            "elapsed_seconds": _safe_float(elapsed_seconds),
            "artifacts": artifacts,
        },
        "history": [asdict(h) for h in history],
    }
    _write_report_json(args.report_json, report)
    _write_report_md(args.report_md, report)

    timeline_result = None
    if not args.skip_timeline:
        timeline_result = _timeline_append(
            timeline_path=args.timeline,
            route=args.route,
            analysis_type=args.analysis_type,
            model_name=f"VisionProjector(d_model={args.d_model})",
            config=report["config"],
            best=report["summary"]["best"],
            elapsed_seconds=elapsed_seconds,
            artifacts=artifacts,
        )

    return {
        "weights": args.weights_out,
        "report_json": args.report_json,
        "report_md": args.report_md,
        "timeline": timeline_result,
        "best": report["summary"]["best"],
        "dataset": dataset_selected,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train vision encoder and append result to AGI route timeline."
    )
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "mnist", "auto"])
    parser.add_argument("--total-samples", type=int, default=20000)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--align-weight", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--anchor-source", default="auto", choices=["auto", "logic", "random"])
    parser.add_argument("--logic-model-path", default="tempdata/StructInit.pth")
    parser.add_argument("--logic-corpus-path", default="data/logic_core/logic_corpus_v1.txt")
    parser.add_argument("--mnist-root", default="tempdata/data")

    parser.add_argument("--weights-out", default="tempdata/vision_projector.pth")
    parser.add_argument("--projection-out", default="tempdata/vision_alignment_projection.png")
    parser.add_argument("--projection-points", type=int, default=1200)
    parser.add_argument("--report-json", default="tempdata/vision_training_report.json")
    parser.add_argument("--report-md", default="tempdata/vision_training_report.md")

    parser.add_argument("--timeline", default="tempdata/agi_route_test_timeline.json")
    parser.add_argument("--route", default="fiber_bundle")
    parser.add_argument("--analysis-type", default="vision_alignment")
    parser.add_argument("--skip-timeline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

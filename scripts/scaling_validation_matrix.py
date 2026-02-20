import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ModelScale:
    name: str
    d_model: int
    n_heads: int
    n_layers: int
    ff_mult: int = 4


@dataclass
class DataScale:
    name: str
    total_samples: int
    train_ratio: float


class ModularAdditionTransformer(nn.Module):
    """Small Transformer for controlled scaling experiments."""

    def __init__(
        self,
        modulus: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_mult: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(modulus, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(2, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, modulus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x) + self.pos_embedding
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_data_scales(preset: str) -> List[DataScale]:
    if preset == "quick":
        return [
            DataScale("d_40k", 40_000, 0.1),
            DataScale("d_120k", 120_000, 0.1),
        ]
    if preset == "full":
        return [
            DataScale("d_100k", 100_000, 0.1),
            DataScale("d_300k", 300_000, 0.1),
            DataScale("d_700k", 700_000, 0.1),
            DataScale("d_1200k", 1_200_000, 0.1),
        ]
    # large
    return [
        DataScale("d_80k", 80_000, 0.1),
        DataScale("d_250k", 250_000, 0.1),
        DataScale("d_600k", 600_000, 0.1),
    ]


def build_custom_data_scales(raw_sizes: str, train_ratio: float) -> List[DataScale]:
    parts = [p.strip() for p in raw_sizes.split(",") if p.strip()]
    values: List[int] = []
    for p in parts:
        values.append(int(p))
    values = sorted(set(values))
    scales: List[DataScale] = []
    for total in values:
        if total <= 0:
            continue
        scales.append(DataScale(f"d_{int(total / 1000)}k", int(total), train_ratio))
    return scales


def build_model_scales(preset: str) -> List[ModelScale]:
    if preset == "quick":
        return [
            ModelScale("m_0.4m", 128, 4, 2),
            ModelScale("m_1.4m", 192, 6, 3),
        ]
    if preset == "full":
        return [
            ModelScale("m_0.4m", 128, 4, 2),
            ModelScale("m_1.4m", 192, 6, 3),
            ModelScale("m_3.2m", 256, 8, 4),
            ModelScale("m_8.5m", 384, 8, 6),
        ]
    # large
    return [
        ModelScale("m_0.4m", 128, 4, 2),
        ModelScale("m_3.2m", 256, 8, 4),
        ModelScale("m_8.5m", 384, 8, 6),
    ]


def create_dataset(total_samples: int, train_ratio: float, modulus: int, seed: int) -> Tuple[TensorDataset, TensorDataset]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, modulus, (total_samples, 2), generator=g, dtype=torch.long)
    y = (x[:, 0] + x[:, 1]) % modulus
    split = max(1, int(total_samples * train_ratio))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]
    return TensorDataset(x_train, y_train), TensorDataset(x_val, y_val)


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    min_lr_scale: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * max(0.0, min(1.0, warmup_ratio)))
    min_scale = max(0.0, min(1.0, min_lr_scale))

    def _lr_lambda(current_step: int) -> float:
        step = min(current_step, total_steps)
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_scale + (1.0 - min_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return (correct / total) if total > 0 else 0.0


def train_one_run(
    model_cfg: ModelScale,
    data_cfg: DataScale,
    args: argparse.Namespace,
    run_seed: int,
    device: torch.device,
) -> Dict:
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    train_ds, val_ds = create_dataset(
        total_samples=data_cfg.total_samples,
        train_ratio=data_cfg.train_ratio,
        modulus=args.modulus,
        seed=run_seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

    model = ModularAdditionTransformer(
        modulus=args.modulus,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        ff_mult=model_cfg.ff_mult,
        dropout=args.dropout,
    ).to(device)
    param_count = count_trainable_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr_scale=args.min_lr_scale,
    )
    criterion = nn.CrossEntropyLoss()

    history: List[Dict] = []
    best_val = 0.0
    start_time = time.time()
    samples_seen = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for step_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            raw_loss = criterion(logits, yb)
            loss = raw_loss / max(1, args.grad_accum_steps)
            loss.backward()

            should_step = (step_idx % max(1, args.grad_accum_steps) == 0) or (step_idx == len(train_loader))
            if should_step:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            epoch_loss += raw_loss.item() * yb.size(0)
            pred = logits.argmax(dim=1)
            epoch_correct += (pred == yb).sum().item()
            epoch_total += yb.size(0)
            samples_seen += yb.size(0)

        train_acc = (epoch_correct / epoch_total) if epoch_total else 0.0
        train_loss = (epoch_loss / epoch_total) if epoch_total else 0.0
        val_acc = evaluate(model, val_loader, device)
        best_val = max(best_val, val_acc)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_acc": round(val_acc, 6),
                "lr": round(float(optimizer.param_groups[0]["lr"]), 10),
            }
        )

    total_time = time.time() - start_time
    final = history[-1] if history else {}
    throughput = samples_seen / total_time if total_time > 0 else 0.0

    result = {
        "run_id": f"{model_cfg.name}__{data_cfg.name}",
        "model_scale": asdict(model_cfg),
        "data_scale": asdict(data_cfg),
        "modulus": args.modulus,
        "seed": run_seed,
        "device": str(device),
        "param_count": param_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": {
            "name": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "min_lr_scale": args.min_lr_scale,
            "grad_accum_steps": args.grad_accum_steps,
        },
        "metrics": {
            "best_val_acc": round(best_val, 6),
            "final_val_acc": round(float(final.get("val_acc", 0.0)), 6),
            "final_train_acc": round(float(final.get("train_acc", 0.0)), 6),
            "generalization_gap": round(float(final.get("train_acc", 0.0)) - float(final.get("val_acc", 0.0)), 6),
            "train_seconds": round(total_time, 4),
            "samples_seen": samples_seen,
            "samples_per_second": round(throughput, 2),
        },
        "history": history,
    }
    return result


def summarize(results: List[Dict]) -> Dict:
    by_model: Dict[str, List[Dict]] = {}
    by_data: Dict[str, List[Dict]] = {}
    for row in results:
        by_model.setdefault(row["model_scale"]["name"], []).append(row)
        by_data.setdefault(row["data_scale"]["name"], []).append(row)

    def _best(rows: List[Dict]) -> Dict:
        return max(rows, key=lambda r: r["metrics"]["best_val_acc"])

    model_summary = {}
    for name, rows in by_model.items():
        best = _best(rows)
        model_summary[name] = {
            "runs": len(rows),
            "best_data_scale": best["data_scale"]["name"],
            "best_val_acc": best["metrics"]["best_val_acc"],
            "avg_val_acc": round(sum(r["metrics"]["final_val_acc"] for r in rows) / len(rows), 6),
        }

    data_summary = {}
    for name, rows in by_data.items():
        best = _best(rows)
        data_summary[name] = {
            "runs": len(rows),
            "best_model_scale": best["model_scale"]["name"],
            "best_val_acc": best["metrics"]["best_val_acc"],
            "avg_val_acc": round(sum(r["metrics"]["final_val_acc"] for r in rows) / len(rows), 6),
        }

    global_best = _best(results) if results else {}
    return {
        "model_summary": model_summary,
        "data_summary": data_summary,
        "global_best": {
            "run_id": global_best.get("run_id"),
            "best_val_acc": global_best.get("metrics", {}).get("best_val_acc"),
            "model_scale": global_best.get("model_scale", {}).get("name"),
            "data_scale": global_best.get("data_scale", {}).get("name"),
        },
    }


def to_markdown(report: Dict) -> str:
    lines = [
        "# Scaling Validation Report",
        "",
        f"- Generated At: {report['meta']['generated_at']}",
        f"- Preset: {report['meta']['preset']}",
        f"- Device: {report['meta']['device']}",
        f"- Runs: {len(report['runs'])}",
        "",
        "## Global Best",
        f"- Run ID: {report['summary']['global_best'].get('run_id')}",
        f"- Best Val Acc: {report['summary']['global_best'].get('best_val_acc')}",
        f"- Model Scale: {report['summary']['global_best'].get('model_scale')}",
        f"- Data Scale: {report['summary']['global_best'].get('data_scale')}",
        "",
        "## Run Table",
        "| Run | Params | Data | Best Val Acc | Final Val Acc | Samples/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in report["runs"]:
        lines.append(
            f"| {r['run_id']} | {r['param_count']} | {r['data_scale']['total_samples']} | "
            f"{r['metrics']['best_val_acc']:.4f} | {r['metrics']['final_val_acc']:.4f} | "
            f"{r['metrics']['samples_per_second']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model/data scaling validation matrix.")
    parser.add_argument("--preset", choices=["quick", "large", "full"], default="large")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--min-lr-scale", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--modulus", type=int, default=113)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run full matrix.")
    parser.add_argument("--model-filter", default="", help="Comma-separated model names, e.g. m_8.5m")
    parser.add_argument("--data-filter", default="", help="Comma-separated data names, e.g. d_700k,d_1200k")
    parser.add_argument("--custom-data-sizes", default="", help="Comma-separated total samples, e.g. 2000000,3000000")
    parser.add_argument("--train-ratio", type=float, default=0.1)
    parser.add_argument("--output-json", default="tempdata/scaling_validation_report.json")
    parser.add_argument("--output-md", default="tempdata/scaling_validation_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model_scales = build_model_scales(args.preset)
    data_scales = (
        build_custom_data_scales(args.custom_data_sizes, args.train_ratio)
        if args.custom_data_sizes.strip()
        else build_data_scales(args.preset)
    )
    if args.model_filter.strip():
        model_allow = {x.strip() for x in args.model_filter.split(",") if x.strip()}
        model_scales = [m for m in model_scales if m.name in model_allow]
    if args.data_filter.strip():
        data_allow = {x.strip() for x in args.data_filter.split(",") if x.strip()}
        data_scales = [d for d in data_scales if d.name in data_allow]
    run_plan = [(m, d) for m in model_scales for d in data_scales]
    if args.max_runs > 0:
        run_plan = run_plan[: args.max_runs]
    if not run_plan:
        raise ValueError("Empty run plan after filters. Check --model-filter / --data-filter.")

    print(f"[Scaling] preset={args.preset} runs={len(run_plan)} device={device}")
    all_results: List[Dict] = []

    for idx, (m_cfg, d_cfg) in enumerate(run_plan, start=1):
        run_seed = args.seed + idx * 17
        print(
            f"[Run {idx}/{len(run_plan)}] model={m_cfg.name} "
            f"data={d_cfg.name} samples={d_cfg.total_samples} seed={run_seed}"
        )
        result = train_one_run(
            model_cfg=m_cfg,
            data_cfg=d_cfg,
            args=args,
            run_seed=run_seed,
            device=device,
        )
        all_results.append(result)
        print(
            f"  -> best_val_acc={result['metrics']['best_val_acc']:.4f}, "
            f"final_val_acc={result['metrics']['final_val_acc']:.4f}, "
            f"samples_per_second={result['metrics']['samples_per_second']:.2f}"
        )

    report = {
        "schema_version": "1.0",
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "preset": args.preset,
            "device": str(device),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "min_lr_scale": args.min_lr_scale,
            "grad_accum_steps": args.grad_accum_steps,
            "dropout": args.dropout,
            "modulus": args.modulus,
            "seed": args.seed,
            "train_ratio": args.train_ratio,
        },
        "runs": all_results,
        "summary": summarize(all_results),
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md = to_markdown(report)
    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[Done] JSON report: {args.output_json}")
    print(f"[Done] Markdown report: {args.output_md}")


if __name__ == "__main__":
    main()

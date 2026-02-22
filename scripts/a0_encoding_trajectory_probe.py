import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.runtime.contracts import AnalysisSpec, ConclusionCard, Metric, RunRecord, RunSummary
from server.runtime.experiment_store import ExperimentTimelineStore


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TinyEncodingNet(nn.Module):
    def __init__(self, modulus: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(modulus, d_model)
        self.pos = nn.Parameter(torch.randn(2, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, modulus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x) + self.pos
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x) + self.pos
        h = self.encoder(h)
        return h.mean(dim=1)


def build_dataset(total_samples: int, train_ratio: float, modulus: int, seed: int) -> Tuple[TensorDataset, TensorDataset]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, modulus, (total_samples, 2), generator=g, dtype=torch.long)
    y = (x[:, 0] + x[:, 1]) % modulus
    split = max(1, int(total_samples * train_ratio))
    train = TensorDataset(x[:split], y[:split])
    val = TensorDataset(x[split:], y[split:])
    return train, val


def _compute_rsm_signature(reps: torch.Tensor, max_points: int = 400) -> torch.Tensor:
    if reps.size(0) > max_points:
        idx = torch.randperm(reps.size(0), device=reps.device)[:max_points]
        reps = reps[idx]
    reps = F.normalize(reps, dim=-1)
    sim = reps @ reps.T
    tri = torch.triu_indices(sim.size(0), sim.size(1), offset=1, device=sim.device)
    return sim[tri[0], tri[1]].detach().cpu()


def _effective_rank_from_singular(s: torch.Tensor) -> float:
    if s.numel() == 0:
        return 0.0
    p = (s + 1e-12) / (s.sum() + 1e-12)
    h = -(p * torch.log(p + 1e-12)).sum()
    return float(torch.exp(h).item())


def _k95_from_singular(s: torch.Tensor) -> int:
    if s.numel() == 0:
        return 0
    cumsum = torch.cumsum(s, dim=0)
    total = float(cumsum[-1].item()) + 1e-12
    idx = torch.searchsorted(cumsum, torch.tensor(0.95 * total, device=s.device))
    return int(idx.item() + 1)


def _specificity_proxy(reps: torch.Tensor, labels: torch.Tensor, max_classes: int = 32) -> float:
    unique = torch.unique(labels)
    if unique.numel() <= 1:
        return 0.0
    unique = unique[:max_classes]
    centroids = []
    within_vals = []
    for c in unique:
        mask = labels == c
        if int(mask.sum().item()) < 2:
            continue
        rc = reps[mask]
        centroid = rc.mean(dim=0)
        centroids.append(centroid)
        within_vals.append((rc - centroid).norm(dim=1).mean().item())
    if len(centroids) <= 1:
        return 0.0
    centroids = torch.stack(centroids, dim=0)
    d = torch.cdist(centroids, centroids)
    tri = torch.triu_indices(d.size(0), d.size(1), offset=1)
    between = d[tri[0], tri[1]].mean().item()
    within = float(np.mean(within_vals)) if within_vals else 0.0
    return _clamp01(between / (between + within + 1e-8))


def evaluate_encoding_metrics(
    model: TinyEncodingNet,
    val_loader: DataLoader,
    device: torch.device,
    prev_rsm_signature: Optional[torch.Tensor],
    rep_samples: int = 3000,
) -> Tuple[Dict[str, float], torch.Tensor]:
    model.eval()
    correct = 0
    total = 0
    reps_list = []
    y_list = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))
            reps = model.encode(xb)
            reps_list.append(reps.detach().cpu())
            y_list.append(yb.detach().cpu())
            if sum(x.size(0) for x in reps_list) >= rep_samples:
                break

    reps = torch.cat(reps_list, dim=0)[:rep_samples]
    labels = torch.cat(y_list, dim=0)[:rep_samples]
    centered = reps - reps.mean(dim=0, keepdim=True)
    _, s, _ = torch.linalg.svd(centered, full_matrices=False)
    eff_rank = _effective_rank_from_singular(s)
    k95 = _k95_from_singular(s)

    d_model = float(reps.size(1))
    abstraction = _clamp01(0.6 * (k95 / d_model) + 0.4 * (eff_rank / d_model))
    precision = _clamp01(correct / max(1, total))
    specificity = _specificity_proxy(reps, labels)

    rsm_sig = _compute_rsm_signature(reps)
    if prev_rsm_signature is None:
        systematicity = 0.5
    else:
        n = min(prev_rsm_signature.numel(), rsm_sig.numel())
        if n < 16:
            systematicity = 0.5
        else:
            a = prev_rsm_signature[:n].numpy()
            b = rsm_sig[:n].numpy()
            corr = np.corrcoef(a, b)[0, 1] if np.std(a) > 1e-8 and np.std(b) > 1e-8 else 0.0
            systematicity = _clamp01((float(corr) + 1.0) / 2.0)

    core = _clamp01(0.30 * abstraction + 0.30 * precision + 0.20 * specificity + 0.20 * systematicity)
    metrics = {
        "abstraction_score": round(abstraction, 4),
        "precision_score": round(precision, 4),
        "specificity_score": round(specificity, 4),
        "systematicity_score": round(systematicity, 4),
        "encoding_core_score": round(core, 4),
        "effective_rank": round(eff_rank, 4),
        "k95": int(k95),
    }
    return metrics, rsm_sig


def run_probe(args: argparse.Namespace) -> Dict[str, any]:
    _seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    train_ds, val_ds = build_dataset(
        total_samples=args.total_samples,
        train_ratio=args.train_ratio,
        modulus=args.modulus,
        seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

    model = TinyEncodingNet(
        modulus=args.modulus,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    checkpoints = sorted(set(int(x.strip()) for x in args.checkpoints.split(",") if x.strip()))
    prev_rsm = None
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        if epoch in checkpoints:
            m, prev_rsm = evaluate_encoding_metrics(model, val_loader, device, prev_rsm, rep_samples=args.rep_samples)
            m["epoch"] = epoch
            history.append(m)

    elapsed = round(time.time() - start, 2)
    threshold = args.pass_threshold
    birth_epoch = None
    for i, m in enumerate(history):
        if m["encoding_core_score"] >= threshold:
            tail = [x["encoding_core_score"] for x in history[i:]]
            if all(v >= threshold for v in tail):
                birth_epoch = m["epoch"]
                break

    first = history[0]["encoding_core_score"] if history else 0.0
    final = history[-1]["encoding_core_score"] if history else 0.0
    delta = round(final - first, 4)
    status = "pass" if (final >= threshold and birth_epoch is not None) else "watch"

    result = {
        "schema_version": "1.0",
        "test_date": datetime.now(timezone.utc).date().isoformat(),
        "analysis_type": "encoding_genesis_trajectory",
        "config": {
            "device": str(device),
            "modulus": args.modulus,
            "total_samples": args.total_samples,
            "train_ratio": args.train_ratio,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "epochs": args.epochs,
            "checkpoints": checkpoints,
            "pass_threshold": threshold,
        },
        "summary": {
            "status": status,
            "birth_epoch": birth_epoch,
            "first_core_score": round(first, 4),
            "final_core_score": round(final, 4),
            "delta_core_score": delta,
            "elapsed_seconds": elapsed,
        },
        "history": history,
    }
    return result


def append_timeline(result: Dict[str, any], report_path: Path, timeline_path: Path) -> None:
    store = ExperimentTimelineStore(path=str(timeline_path))
    s = result.get("summary", {})
    now = time.time()
    record = RunRecord(
        run_id=f"run_a0_encoding_traj_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        spec=AnalysisSpec(
            route="fiber_bundle",
            analysis_type="encoding_genesis_trajectory",
            model="TinyEncodingNet",
            params={"source_report": str(report_path).replace("\\", "/")},
            input_payload={},
        ),
        status="completed",
        created_at=now,
        updated_at=now + 0.001,
        summary=RunSummary(
            metrics=[
                Metric(key="first_core_score", value=float(s.get("first_core_score", 0.0)), min_value=0.0, max_value=1.0),
                Metric(key="final_core_score", value=float(s.get("final_core_score", 0.0)), min_value=0.0, max_value=1.0),
                Metric(key="delta_core_score", value=float(s.get("delta_core_score", 0.0)), min_value=-1.0, max_value=1.0),
            ],
            conclusion=ConclusionCard(
                objective="Probe how encoding emerges over training trajectory (A0).",
                method="Train tiny model and evaluate encoding capability at multiple checkpoints.",
                evidence=[
                    f"birth_epoch={s.get('birth_epoch')}",
                    f"first={s.get('first_core_score')}",
                    f"final={s.get('final_core_score')}",
                ],
                result=f"Encoding trajectory probe completed with status={s.get('status')}.",
                confidence=0.72 if s.get("status") == "pass" else 0.62,
                limitations=["Tiny-model proxy; extend to larger models and real corpora."],
                next_action="Transfer trajectory probing to larger architecture checkpoints.",
            ),
            artifacts=[{"path": str(report_path).replace("\\", "/")}],
        ),
        event_count=0,
    )
    store.append_run(record)


def main() -> None:
    parser = argparse.ArgumentParser(description="A0 encoding genesis trajectory probe.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--modulus", type=int, default=31)
    parser.add_argument("--total-samples", type=int, default=30000)
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--checkpoints", type=str, default="1,3,6,9,12")
    parser.add_argument("--rep-samples", type=int, default=3000)
    parser.add_argument("--pass-threshold", type=float, default=0.62)
    parser.add_argument("--seed", type=int, default=20260221)
    parser.add_argument("--timeline", default="tempdata/agi_route_test_timeline.json")
    parser.add_argument("--output", default="tempdata/a0_encoding_trajectory_20260221.json")
    args = parser.parse_args()

    result = run_probe(args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    append_timeline(result, report_path=out_path, timeline_path=Path(args.timeline))
    print(json.dumps({"output": str(out_path), "status": result["summary"]["status"], "final_core_score": result["summary"]["final_core_score"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

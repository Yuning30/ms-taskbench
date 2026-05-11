"""Train the pick-feasibility verifier (v1 MLP).

Usage:
    uv run python -m taskbench.verifier.train \
        --data-dir /common/users/shared/pracsys/ms-taskbench-data/datasets/pick_2dgrid_mplib_500k \
        --experiments-dir /common/users/shared/pracsys/ms-taskbench-data/experiments \
        --name v1_mlp --epochs 20 --batch-size 2048 --lr 1e-3

Outputs (under <experiments-dir>/<name>/):
    model.pt          best-by-val-AUC checkpoint
    metrics.json      final + per-epoch metrics
    train.log         stdout copy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from taskbench.verifier.dataset import (
    FEATURE_DIM,
    PickFeasibilityDataset,
    build_splits,
)
from taskbench.verifier.model import PickFeasibilityMLP

logger = logging.getLogger("taskbench.verifier.train")


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    val_auc: float
    elapsed_s: float


def _auc(logits: torch.Tensor, y: torch.Tensor) -> float:
    """ROC-AUC via tied-rank Mann-Whitney U. No sklearn dependency."""
    scores = logits.detach().cpu().numpy()
    labels = y.detach().cpu().numpy().astype(bool)
    n_pos = labels.sum()
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # Average ranks for ties.
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.size:
        j = i + 1
        while j < sorted_scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = (ranks[order[i]] + ranks[order[j - 1]]) / 2.0
            ranks[order[i:j]] = avg
        i = j
    sum_ranks_pos = ranks[labels].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _evaluate(model, loader, device, loss_fn):
    model.eval()
    losses, all_logits, all_y = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            losses.append(loss_fn(logits, y).item() * y.numel())
            all_logits.append(logits.cpu())
            all_y.append(y.cpu())
    n = sum(t.numel() for t in all_y)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    preds = (logits >= 0).float()  # sigmoid >= 0.5 <=> logit >= 0
    acc = (preds == y).float().mean().item()
    return {
        "loss": sum(losses) / max(n, 1),
        "acc": acc,
        "auc": _auc(logits, y),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Source dataset dir (containing task_*/shard_*.parquet).")
    p.add_argument("--experiments-dir", type=Path, required=True,
                   help="Root for splits/ and per-experiment subdirs.")
    p.add_argument("--name", type=str, required=True,
                   help="Experiment name (subdir under experiments-dir).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--train-n", type=int, default=50_000)
    p.add_argument("--val-n", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for both split shuffle and weight init.")
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu (auto-detected if omitted).")
    p.add_argument("--rebuild-splits", action="store_true",
                   help="Re-create splits even if they already exist.")
    return p


def main():
    args = _build_parser().parse_args()
    exp_dir = args.experiments_dir / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = exp_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="w")],
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits_dir = args.experiments_dir / "splits"
    logger.info("Preparing splits at %s ...", splits_dir)
    paths = build_splits(
        source_dir=args.data_dir,
        out_dir=splits_dir,
        train_n=args.train_n,
        val_n=args.val_n,
        seed=args.seed,
        rebuild=args.rebuild_splits,
    )
    logger.info("Splits: train=%s val=%s eval=%s", paths["train"], paths["val"], paths["eval"])

    train_ds = PickFeasibilityDataset(paths["train"])
    val_ds = PickFeasibilityDataset(paths["val"])
    logger.info("Loaded train=%d, val=%d, feature_dim=%d", len(train_ds), len(val_ds), FEATURE_DIM)
    logger.info("Train success rate: %.3f", train_ds.y.mean().item())
    logger.info("Val   success rate: %.3f", val_ds.y.mean().item())

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(device == "cuda"),
    )

    model = PickFeasibilityMLP(input_dim=FEATURE_DIM, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    history: list[dict] = []
    best_auc = -1.0
    best_path = exp_dir / "model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running_loss, running_n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * y.numel()
            running_n += y.numel()
        train_loss = running_loss / running_n

        val = _evaluate(model, val_loader, device, loss_fn)
        elapsed = time.perf_counter() - t0
        em = EpochMetrics(
            epoch=epoch, train_loss=train_loss,
            val_loss=val["loss"], val_acc=val["acc"], val_auc=val["auc"],
            elapsed_s=elapsed,
        )
        history.append(em.__dict__)
        logger.info(
            "epoch %2d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  val_auc=%.4f  (%.1fs)",
            epoch, train_loss, val["loss"], val["acc"], val["auc"], elapsed,
        )

        if val["auc"] > best_auc:
            best_auc = val["auc"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_auc": val["auc"],
                "val_acc": val["acc"],
                "feature_dim": FEATURE_DIM,
                "args": vars(args) | {"data_dir": str(args.data_dir),
                                       "experiments_dir": str(args.experiments_dir)},
            }, best_path)
            logger.info("  ↳ saved new best to %s", best_path)

    metrics = {
        "best_val_auc": best_auc,
        "epochs": history,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Done. Best val AUC: %.4f. Checkpoint at %s", best_auc, best_path)


if __name__ == "__main__":
    main()

"""Evaluate a trained pick-feasibility verifier on the eval split.

Writes:
    <experiments-dir>/<name>/eval_predictions.parquet  per-row (scene_id, prob, label, source)
    <experiments-dir>/<name>/eval_metrics.json         overall + per-source AUC/acc/PR

Usage:
    uv run python -m taskbench.verifier.eval \
        --experiments-dir /common/users/shared/pracsys/ms-taskbench-data/experiments \
        --name v1_mlp
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from taskbench.verifier.dataset import FEATURE_DIM, PickFeasibilityDataset
from taskbench.verifier.model import PickFeasibilityMLP
from taskbench.verifier.train import _auc

logger = logging.getLogger("taskbench.verifier.eval")


def _metrics_at_threshold(probs: np.ndarray, labels: np.ndarray, t: float = 0.5):
    preds = (probs >= t).astype(np.int32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    acc = (tp + tn) / max(len(labels), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(labels)}


def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments-dir", type=Path, required=True)
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    return p


def main():
    args = _build_parser().parse_args()
    exp_dir = args.experiments_dir / args.name
    splits_dir = args.experiments_dir / "splits"
    eval_path = splits_dir / "eval.parquet"
    ckpt_path = exp_dir / "model.pt"
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval split missing: {eval_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    ds = PickFeasibilityDataset(eval_path)
    logger.info("Eval samples: %d", len(ds))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PickFeasibilityMLP(input_dim=FEATURE_DIM).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_y.append(y)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    probs = torch.sigmoid(logits).numpy()
    labels = y.numpy().astype(np.int32)

    sources = np.asarray(ds.source)
    scene_ids = np.asarray(ds.scene_id)

    pq.write_table(
        pa.table({
            "scene_id": pa.array(scene_ids),
            "source": pa.array(sources),
            "label": pa.array(labels.astype(bool)),
            "prob": pa.array(probs.astype(np.float64)),
            "pred": pa.array((probs >= args.threshold).astype(bool)),
        }),
        exp_dir / "eval_predictions.parquet",
    )
    logger.info("Wrote eval_predictions.parquet (%d rows)", len(probs))

    overall = {
        "auc": _auc(logits, y),
        "threshold": args.threshold,
        **_metrics_at_threshold(probs, labels, args.threshold),
        "pos_rate": float(labels.mean()),
    }
    per_source = {}
    for src in np.unique(sources):
        mask = sources == src
        if mask.sum() == 0:
            continue
        per_source[str(src)] = {
            "auc": _auc(logits[mask], y[mask]),
            **_metrics_at_threshold(probs[mask], labels[mask], args.threshold),
            "pos_rate": float(labels[mask].mean()),
        }

    metrics = {"overall": overall, "per_source": per_source,
               "checkpoint_epoch": int(ckpt.get("epoch", -1))}
    (exp_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(
        "Overall  AUC=%.4f  acc=%.4f  prec=%.4f  recall=%.4f  f1=%.4f  n=%d",
        overall["auc"], overall["acc"], overall["precision"],
        overall["recall"], overall["f1"], overall["n"],
    )
    for src, m in per_source.items():
        logger.info("  %s: AUC=%.4f  acc=%.4f  f1=%.4f  pos=%.3f  n=%d",
                    src, m["auc"], m["acc"], m["f1"], m["pos_rate"], m["n"])


if __name__ == "__main__":
    main()

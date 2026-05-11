"""Calibrate the pick-feasibility verifier.

Two-step calibration:
    1. Temperature scaling on raw logits — single scalar T fit by minimizing
       BCE on the calibration set. Preserves AUC, fixes confidence.
    2. Per-source decision thresholds — scan a grid for best F1 per source
       (plus a global F1 threshold for reference).

Reads <experiments-dir>/<name>/model.pt and <experiments-dir>/splits/eval.parquet
(carving out 10K for calibration on first run), writes:
    <experiments-dir>/<name>/calibration.json
    <experiments-dir>/<name>/eval_holdout_metrics.json
    <experiments-dir>/<name>/eval_holdout_predictions.parquet

Usage:
    uv run python -m taskbench.verifier.calibrate \
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

from taskbench.verifier.dataset import (
    FEATURE_DIM,
    PickFeasibilityDataset,
    build_calibration_split,
)
from taskbench.verifier.model import PickFeasibilityMLP
from taskbench.verifier.train import _auc

logger = logging.getLogger("taskbench.verifier.calibrate")


# ----------------------------------------------------------------------------
# Calibration math
# ----------------------------------------------------------------------------

def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a single scalar T s.t. sigmoid(logits/T) minimizes BCE on (logits, labels).

    Uses scipy-free L-BFGS via torch.optim. Constrained to T in [0.05, 50] so
    runaway values from degenerate inputs don't blow up downstream.
    """
    z = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    log_T = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        T = log_T.exp()
        loss = bce(z / T, y)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(log_T.exp().item())
    return float(min(max(T, 0.05), 50.0))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: bin probs into n_bins, average |bin_acc - bin_conf| weighted by bin size."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        in_bin = (probs >= bins[i]) & (probs < bins[i + 1] if i < n_bins - 1 else probs <= bins[i + 1])
        if not in_bin.any():
            continue
        bin_conf = probs[in_bin].mean()
        bin_acc = labels[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def reliability_diagram(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """Return per-bin counts, mean confidence, mean accuracy. Plot-ready."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        if i < n_bins - 1:
            in_bin = (probs >= bins[i]) & (probs < bins[i + 1])
        else:
            in_bin = (probs >= bins[i]) & (probs <= bins[i + 1])
        n_in = int(in_bin.sum())
        out.append({
            "bin_lo": float(bins[i]),
            "bin_hi": float(bins[i + 1]),
            "n": n_in,
            "mean_conf": float(probs[in_bin].mean()) if n_in else None,
            "mean_acc": float(labels[in_bin].mean()) if n_in else None,
        })
    return {"bins": out}


def best_f1_threshold(probs: np.ndarray, labels: np.ndarray, grid: np.ndarray | None = None) -> dict:
    """Scan a threshold grid, return the threshold maximizing F1 + the F1 it achieves.

    Returns NaN F1 and threshold=0.5 if there are no positives.
    """
    if labels.sum() == 0:
        return {"threshold": 0.5, "f1": float("nan"),
                "precision": float("nan"), "recall": float("nan")}
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for t in grid:
        preds = probs >= t
        tp = float(((preds) & (labels == 1)).sum())
        fp = float(((preds) & (labels == 0)).sum())
        fn = float(((~preds) & (labels == 1)).sum())
        prec = tp / max(tp + fp, 1.0)
        rec = tp / max(tp + fn, 1.0)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1),
                    "precision": float(prec), "recall": float(rec)}
    return best


# ----------------------------------------------------------------------------
# Inference utilities
# ----------------------------------------------------------------------------

def _logits_for_split(model, split_path: Path, device: str, batch_size: int = 4096):
    ds = PickFeasibilityDataset(split_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            all_logits.append(model(x).cpu())
            all_y.append(y)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_y).numpy().astype(np.int32)
    return logits, labels, np.asarray(ds.source), np.asarray(ds.scene_id)


def _metrics(probs: np.ndarray, labels: np.ndarray, t: float) -> dict:
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
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(labels),
            "threshold": t}


def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments-dir", type=Path, required=True)
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--calib-n", type=int, default=10_000)
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--rebuild-calib", action="store_true")
    return p


def main():
    args = _build_parser().parse_args()
    exp_dir = args.experiments_dir / args.name
    splits_dir = args.experiments_dir / "splits"
    ckpt_path = exp_dir / "model.pt"
    eval_path = splits_dir / "eval.parquet"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval split missing: {eval_path}")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)

    # ---- 1. Carve calibration / holdout splits ------------------------------
    cal_paths = build_calibration_split(
        eval_path=eval_path, out_dir=splits_dir,
        calib_n=args.calib_n, seed=args.calib_seed,
        rebuild=args.rebuild_calib,
    )
    calib_path = cal_paths["calib"]
    holdout_path = cal_paths["eval_holdout"]
    logger.info("Calibration split: %s", calib_path)
    logger.info("Eval holdout:      %s", holdout_path)

    # ---- 2. Load model ------------------------------------------------------
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PickFeasibilityMLP(input_dim=FEATURE_DIM).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- 3. Calibrate on the 10K split --------------------------------------
    logger.info("Running model on calibration split ...")
    c_logits, c_labels, c_sources, _ = _logits_for_split(model, calib_path, device)

    pre_probs = 1.0 / (1.0 + np.exp(-c_logits))
    pre_ece = expected_calibration_error(pre_probs, c_labels)
    pre_auc = _auc(torch.tensor(c_logits), torch.tensor(c_labels.astype(np.float32)))

    T = fit_temperature(c_logits, c_labels.astype(np.float64))
    post_probs = 1.0 / (1.0 + np.exp(-c_logits / T))
    post_ece = expected_calibration_error(post_probs, c_labels)

    logger.info("Temperature T = %.4f", T)
    logger.info("ECE   pre = %.4f   post = %.4f   (calib set)", pre_ece, post_ece)
    logger.info("AUC   calib = %.4f", pre_auc)

    # ---- 4. Find best F1 thresholds (global + per-source) -------------------
    global_thr = best_f1_threshold(post_probs, c_labels)
    per_source_thr = {}
    for src in np.unique(c_sources):
        m = c_sources == src
        per_source_thr[str(src)] = best_f1_threshold(post_probs[m], c_labels[m])
        per_source_thr[str(src)]["n_calib"] = int(m.sum())
        per_source_thr[str(src)]["pos_rate_calib"] = float(c_labels[m].mean())
    logger.info("Global best-F1 threshold: %.3f  (F1=%.4f, P=%.4f, R=%.4f)",
                global_thr["threshold"], global_thr["f1"],
                global_thr["precision"], global_thr["recall"])
    for src, m in per_source_thr.items():
        logger.info("  %s: T*=%.3f  F1=%.4f  (pos_rate=%.4f, n=%d)",
                    src, m["threshold"], m["f1"],
                    m["pos_rate_calib"], m["n_calib"])

    calibration = {
        "temperature": T,
        "global_threshold": global_thr,
        "per_source_threshold": per_source_thr,
        "ece_pre": pre_ece,
        "ece_post": post_ece,
        "auc_calib": pre_auc,
        "reliability_pre": reliability_diagram(pre_probs, c_labels),
        "reliability_post": reliability_diagram(post_probs, c_labels),
        "calib_n": int(len(c_labels)),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
    }
    (exp_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))

    # ---- 5. Apply to the eval holdout --------------------------------------
    logger.info("Running model on eval holdout ...")
    h_logits, h_labels, h_sources, h_scenes = _logits_for_split(model, holdout_path, device)
    h_probs = 1.0 / (1.0 + np.exp(-h_logits / T))

    # Per-row pred using per-source threshold; also keep global threshold pred for comparison.
    src_to_t = {s: per_source_thr[s]["threshold"] for s in per_source_thr}
    per_source_pred = np.zeros(len(h_probs), dtype=bool)
    for s, t in src_to_t.items():
        mask = h_sources == s
        per_source_pred[mask] = h_probs[mask] >= t
    global_pred = h_probs >= global_thr["threshold"]

    pq.write_table(
        pa.table({
            "scene_id": pa.array(h_scenes),
            "source": pa.array(h_sources),
            "label": pa.array(h_labels.astype(bool)),
            "prob_calibrated": pa.array(h_probs.astype(np.float64)),
            "pred_global": pa.array(global_pred),
            "pred_per_source": pa.array(per_source_pred),
        }),
        exp_dir / "eval_holdout_predictions.parquet",
    )

    overall = {
        "auc": _auc(torch.tensor(h_logits), torch.tensor(h_labels.astype(np.float32))),
        "global": _metrics(h_probs, h_labels, global_thr["threshold"]),
        "per_source_global_overall": {  # using per-source thresholds aggregated to overall
            "acc": float((per_source_pred == h_labels.astype(bool)).mean()),
            "n": int(len(h_labels)),
        },
        "ece_calibrated": expected_calibration_error(h_probs, h_labels),
        "n_eval_holdout": int(len(h_labels)),
    }
    per_source_metrics = {}
    for src in np.unique(h_sources):
        m = h_sources == src
        t = src_to_t.get(str(src), 0.5)
        per_source_metrics[str(src)] = {
            "auc": _auc(torch.tensor(h_logits[m]),
                        torch.tensor(h_labels[m].astype(np.float32))),
            "threshold_used": float(t),
            **_metrics(h_probs[m], h_labels[m], t),
            "pos_rate": float(h_labels[m].mean()),
        }

    metrics = {"overall": overall, "per_source": per_source_metrics}
    (exp_dir / "eval_holdout_metrics.json").write_text(json.dumps(metrics, indent=2))

    logger.info("Holdout overall  AUC=%.4f  ECE(calibrated)=%.4f  n=%d",
                overall["auc"], overall["ece_calibrated"], overall["n_eval_holdout"])
    logger.info("With per-source thresholds, overall accuracy = %.4f",
                overall["per_source_global_overall"]["acc"])
    for src, m in per_source_metrics.items():
        logger.info("  %s  AUC=%.4f  F1=%.4f  P=%.4f  R=%.4f  t=%.3f  pos=%.4f  n=%d",
                    src, m["auc"], m["f1"], m["precision"], m["recall"],
                    m["threshold_used"], m["pos_rate"], m["n"])


if __name__ == "__main__":
    main()

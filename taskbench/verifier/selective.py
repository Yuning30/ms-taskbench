"""Three-way (success / uncertain / failure) selective classification.

Given calibrated probabilities p_success and two parameters (lambda, delta):
    p > lambda + delta  -> success
    p < lambda - delta  -> failure
    otherwise           -> uncertain

(lambda, delta) are fit on the calibration split, then applied to the
eval-holdout split. We grid-search over (lambda, delta), maximize selective
accuracy among covered samples subject to a coverage floor, separately for
each source and once globally.

Usage:
    uv run python -m taskbench.verifier.selective \
        --experiments-dir /common/users/shared/pracsys/ms-taskbench-data/experiments \
        --name v1_mlp --min-coverage 0.80
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

logger = logging.getLogger("taskbench.verifier.selective")


# ----------------------------------------------------------------------------
# Selective-classification math
# ----------------------------------------------------------------------------

def apply_three_way(probs: np.ndarray, lam: float, delta: float) -> np.ndarray:
    """Return an int8 vector: 1=success, 0=failure, -1=uncertain."""
    up = lam + delta
    lo = lam - delta
    out = np.full(probs.shape, -1, dtype=np.int8)
    out[probs > up] = 1
    out[probs < lo] = 0
    return out


def selective_metrics(probs: np.ndarray, labels: np.ndarray,
                      lam: float, delta: float) -> dict:
    """Confusion and rate metrics for the 3-way decision."""
    pred = apply_three_way(probs, lam, delta)
    n = len(probs)
    covered = pred != -1
    uncertain = ~covered

    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    n_uncertain = int(uncertain.sum())
    n_uncertain_pos = int((uncertain & (labels == 1)).sum())
    n_uncertain_neg = int((uncertain & (labels == 0)).sum())

    n_cov = max(tp + fp + tn + fn, 1)
    sel_acc = (tp + tn) / n_cov
    sel_prec = tp / max(tp + fp, 1)
    sel_rec = tp / max(tp + fn, 1)
    sel_f1 = 2 * sel_prec * sel_rec / max(sel_prec + sel_rec, 1e-12)

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    tnr = tn / max(tn + fp, 1)
    fnr = fn / max(fn + tp, 1)

    return {
        "n": n,
        "lambda": float(lam),
        "delta": float(delta),
        "coverage": float(covered.sum() / max(n, 1)),
        "uncertain": n_uncertain,
        "uncertain_pos": n_uncertain_pos,
        "uncertain_neg": n_uncertain_neg,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "selective_accuracy": float(sel_acc),
        "selective_precision": float(sel_prec),
        "selective_recall": float(sel_rec),
        "selective_f1": float(sel_f1),
        "tpr": float(tpr), "fpr": float(fpr),
        "tnr": float(tnr), "fnr": float(fnr),
    }


def find_lambda_delta(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    min_coverage: float = 0.80,
    n_lambda: int = 50,
    n_delta: int = 30,
) -> dict:
    """Grid search (lambda, delta) maximizing selective_accuracy at coverage >= min_coverage.

    Falls back to maximizing coverage if the constraint can't be met (e.g.,
    degenerate templated with virtually no positives).
    """
    lambdas = np.linspace(0.05, 0.95, n_lambda)
    deltas = np.linspace(0.0, 0.45, n_delta)
    n = len(probs)
    if n == 0:
        return {"lambda": 0.5, "delta": 0.0,
                "selective_accuracy": float("nan"), "coverage": 0.0}

    best = None
    best_feasible = None
    for lam in lambdas:
        for delt in deltas:
            up = lam + delt
            lo = lam - delt
            pos = probs > up
            neg = probs < lo
            covered = pos | neg
            cov_frac = covered.sum() / n
            if covered.sum() == 0:
                continue
            tp = int((pos & (labels == 1)).sum())
            tn = int((neg & (labels == 0)).sum())
            sel_acc = (tp + tn) / covered.sum()
            cand = {"lambda": float(lam), "delta": float(delt),
                    "selective_accuracy": float(sel_acc),
                    "coverage": float(cov_frac)}
            if cov_frac >= min_coverage:
                if best_feasible is None or sel_acc > best_feasible["selective_accuracy"]:
                    best_feasible = cand
            # also track unconstrained best in case nothing meets the floor
            if best is None or cov_frac > best["coverage"]:
                best = cand
    return best_feasible if best_feasible is not None else best


# ----------------------------------------------------------------------------
# Plumbing
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


def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments-dir", type=Path, required=True)
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--min-coverage", type=float, default=0.80,
                   help="Coverage floor (fraction of non-uncertain predictions).")
    p.add_argument("--n-lambda", type=int, default=50)
    p.add_argument("--n-delta", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", type=str, default=None)
    return p


def main():
    args = _build_parser().parse_args()
    exp_dir = args.experiments_dir / args.name
    splits_dir = args.experiments_dir / "splits"

    calib_path = splits_dir / "calib.parquet"
    holdout_path = splits_dir / "eval_holdout.parquet"
    ckpt_path = exp_dir / "model.pt"
    calibration_path = exp_dir / "calibration.json"

    for required in (calib_path, holdout_path, ckpt_path, calibration_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing required file: {required}")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    T = float(json.loads(calibration_path.read_text())["temperature"])
    logger.info("Using temperature T = %.4f from calibration.json", T)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PickFeasibilityMLP(input_dim=FEATURE_DIM).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- Fit (lambda, delta) on calib --------------------------------------
    logger.info("Scoring calibration split ...")
    c_logits, c_labels, c_sources, _ = _logits_for_split(model, calib_path, device,
                                                         args.batch_size)
    c_probs = 1.0 / (1.0 + np.exp(-c_logits / T))

    fitted = {"global": find_lambda_delta(c_probs, c_labels,
                                           min_coverage=args.min_coverage,
                                           n_lambda=args.n_lambda,
                                           n_delta=args.n_delta)}
    for src in np.unique(c_sources):
        m = c_sources == src
        fitted[str(src)] = find_lambda_delta(c_probs[m], c_labels[m],
                                              min_coverage=args.min_coverage,
                                              n_lambda=args.n_lambda,
                                              n_delta=args.n_delta)

    logger.info("Fitted params (max sel_acc with coverage >= %.2f):", args.min_coverage)
    for key, f in fitted.items():
        logger.info("  %-12s lambda=%.3f delta=%.3f  sel_acc=%.4f  coverage=%.3f",
                    key, f["lambda"], f["delta"],
                    f["selective_accuracy"], f["coverage"])

    # ---- Apply on eval holdout ---------------------------------------------
    logger.info("Scoring eval holdout ...")
    h_logits, h_labels, h_sources, h_scenes = _logits_for_split(model, holdout_path, device,
                                                                args.batch_size)
    h_probs = 1.0 / (1.0 + np.exp(-h_logits / T))

    # Per-source predictions using each source's (lambda, delta).
    per_source_pred = np.full(len(h_probs), -1, dtype=np.int8)
    for src in np.unique(h_sources):
        m = h_sources == src
        f = fitted[str(src)]
        per_source_pred[m] = apply_three_way(h_probs[m], f["lambda"], f["delta"])

    # Global predictions using global (lambda, delta).
    g = fitted["global"]
    global_pred = apply_three_way(h_probs, g["lambda"], g["delta"])

    pq.write_table(
        pa.table({
            "scene_id": pa.array(h_scenes),
            "source": pa.array(h_sources),
            "label": pa.array(h_labels.astype(bool)),
            "prob_calibrated": pa.array(h_probs.astype(np.float64)),
            "pred_global": pa.array(global_pred.astype(np.int8)),
            "pred_per_source": pa.array(per_source_pred.astype(np.int8)),
        }),
        exp_dir / "selective_predictions.parquet",
    )

    # ---- Metrics on holdout ------------------------------------------------
    overall_metrics = {
        "global_params_overall": selective_metrics(h_probs, h_labels, g["lambda"], g["delta"]),
        "per_source_params_overall": {
            "n": int(len(h_labels)),
            "coverage": float((per_source_pred != -1).mean()),
            "selective_accuracy": float(
                ((per_source_pred == h_labels.astype(np.int8)) & (per_source_pred != -1)).sum()
                / max((per_source_pred != -1).sum(), 1)
            ),
        },
    }
    per_source_metrics = {}
    for src in np.unique(h_sources):
        m = h_sources == src
        f = fitted[str(src)]
        per_source_metrics[str(src)] = selective_metrics(
            h_probs[m], h_labels[m], f["lambda"], f["delta"]
        )

    out = {
        "min_coverage_constraint": args.min_coverage,
        "temperature": T,
        "fitted_params": fitted,
        "holdout_overall": overall_metrics,
        "holdout_per_source": per_source_metrics,
    }
    (exp_dir / "selective_metrics.json").write_text(json.dumps(out, indent=2))

    # ---- Log results -------------------------------------------------------
    logger.info("Holdout (per-source params):")
    for src, m in per_source_metrics.items():
        logger.info("  %-12s cov=%.3f  sel_acc=%.4f  sel_F1=%.4f  uncertain=%d  TP=%d FP=%d FN=%d TN=%d",
                    src, m["coverage"], m["selective_accuracy"],
                    m["selective_f1"], m["uncertain"],
                    m["tp"], m["fp"], m["fn"], m["tn"])
    g_overall = overall_metrics["global_params_overall"]
    logger.info("Holdout (global params)  cov=%.3f  sel_acc=%.4f  sel_F1=%.4f  uncertain=%d",
                g_overall["coverage"], g_overall["selective_accuracy"],
                g_overall["selective_f1"], g_overall["uncertain"])


if __name__ == "__main__":
    main()

import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.architectures.gnn import MultiTaskToxGNN
from src.config.tox21_model_config import (
    DROPOUT,
    HIDDEN_DIM,
    MODEL_PATH_1,
    N_LAYERS,
    N_TASKS,
    TASKS,
)
from src.data_pipeline.tox21_dataset import get_dataloaders

METRICS_PATH = "results/tox21_test_metrics.json"

# Hanley-McNeil standard error tiers for the UI's per-assay reliability
# badge. NR-ER-LBD (28 test positives, SE~0.05) is the assay that
# motivated this: its AUC has roughly double the standard error of a
# well-supported assay like NR-AhR (92 positives, SE~0.03), so the
# headline number alone can't tell a user which assays to trust.
RELIABILITY_SE_HIGH = 0.03
RELIABILITY_SE_MEDIUM = 0.05


def _auc_standard_error(auc, n_pos, n_neg):
    """Hanley & McNeil (1982) standard error of an empirical AUC."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    variance = (
        auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)
    ) / (n_pos * n_neg)
    return variance**0.5


def _reliability_tier(se):
    if se < RELIABILITY_SE_HIGH:
        return "high"
    if se < RELIABILITY_SE_MEDIUM:
        return "medium"
    return "low"


def evaluate(model_path=None, metrics_path=None):
    model_path = model_path or MODEL_PATH_1
    metrics_path = metrics_path or METRICS_PATH
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, test_loader = get_dataloaders()

    model = MultiTaskToxGNN(
        n_tasks=N_TASKS, hidden=HIDDEN_DIM, n_layers=N_LAYERS, dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    probs_all, targets_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            probs_all.append(torch.sigmoid(model(batch)).cpu())
            targets_all.append(batch.y.cpu())
    probs_all = torch.cat(probs_all).numpy()
    targets_all = torch.cat(targets_all).numpy()

    per_assay = {}
    aucs, auprcs = [], []
    for i, task in enumerate(TASKS):
        mask = ~np.isnan(targets_all[:, i])
        y_true = targets_all[mask, i]
        y_prob = probs_all[mask, i]

        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos

        # AUC/AUPRC are undefined with no positives (or no negatives) in
        # this assay's test-fold labels -- report as null rather than 0/1.
        if len(y_true) == 0 or n_pos == 0 or n_neg == 0:
            per_assay[task] = {
                "auc": None,
                "auprc": None,
                "n_test": int(len(y_true)),
                "n_pos": n_pos,
                "reliability": "low",
            }
            continue

        auc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        se = _auc_standard_error(auc, n_pos, n_neg)
        per_assay[task] = {
            "auc": auc,
            "auprc": auprc,
            "n_test": int(len(y_true)),
            "n_pos": n_pos,
            "auc_se": se,
            "reliability": _reliability_tier(se),
        }
        aucs.append(auc)
        auprcs.append(auprc)

    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    mean_auprc = float(np.mean(auprcs)) if auprcs else float("nan")

    print("Test set performance (per assay):")
    for task, stats in per_assay.items():
        if stats["auc"] is None:
            print(f"   {task:<15} undefined (n_test={stats['n_test']}, n_pos={stats['n_pos']})")
        else:
            print(
                f"   {task:<15} AUC={stats['auc']:.3f}  AUPRC={stats['auprc']:.3f}  "
                f"reliability={stats['reliability']:<6} (n_test={stats['n_test']}, n_pos={stats['n_pos']})"
            )
    print(f"\nMean AUC:   {mean_auc:.3f}")
    print(f"Mean AUPRC: {mean_auprc:.3f}")

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {"per_assay": per_assay, "mean_auc": mean_auc, "mean_auprc": mean_auprc}, f, indent=2
        )
    print(f"Saved metrics to {metrics_path}")

    return per_assay, mean_auc, mean_auprc


if __name__ == "__main__":
    evaluate()

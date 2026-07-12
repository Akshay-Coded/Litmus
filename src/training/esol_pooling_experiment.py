"""Trains SolubilityGNN once per readout/pooling strategy and compares
test-set performance, holding every other hyperparameter fixed.
"""

import json
import os

from src.architectures.gnn import POOLING_CHOICES
from src.training.esol_evaluate import evaluate
from src.training.esol_train import train

RESULTS_PATH = "results/esol_pooling_comparison.json"


def run_experiment():
    results = {}
    for pooling in POOLING_CHOICES:
        print(f"\n{'=' * 60}\nPooling: {pooling}\n{'=' * 60}")
        model_path = f"models/esol_gnn_{pooling}.pt"
        metrics_path = f"results/esol_test_metrics_{pooling}.json"

        best_val_loss = train(pooling=pooling, model_path=model_path)
        rmse, mae, r2 = evaluate(pooling=pooling, model_path=model_path, metrics_path=metrics_path)

        results[pooling] = {
            "best_val_loss": best_val_loss,
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
        }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}\nSummary (test set, original log-solubility units)\n{'=' * 60}")
    print(f"{'pooling':<12} {'RMSE':>8} {'MAE':>8} {'R^2':>8}")
    for pooling, m in results.items():
        print(f"{pooling:<12} {m['test_rmse']:>8.3f} {m['test_mae']:>8.3f} {m['test_r2']:>8.3f}")

    best_pooling = min(results, key=lambda p: results[p]["test_rmse"])
    print(f"\nBest by test RMSE: {best_pooling}")
    print(f"Saved full comparison to {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    run_experiment()

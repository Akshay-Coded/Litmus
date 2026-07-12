"""Evaluates the ensemble on the test set two ways: each member model on
its own, and all five combined under each strategy (mean / median /
weighted) -- so it's clear whether combining them actually beats the best
individual model, not just the average one. Every model's forward pass
runs exactly once (via collect_predictions) and is reused for both views.
The combination logic itself lives in src/inference/esol_predictor.py,
shared with the UI.
"""

import json
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_pipeline.esol_dataset import get_dataloaders
from src.inference.esol_predictor import (
    STRATEGIES,
    collect_predictions,
    combine_predictions,
    load_ensemble,
    load_manifest,
)

METRICS_PATH = "results/esol_test_metrics_ensemble.json"


def _regression_metrics(targets, preds):
    return {
        "rmse": mean_squared_error(targets, preds) ** 0.5,
        "mae": mean_absolute_error(targets, preds),
        "r2": r2_score(targets, preds),
    }


def evaluate_ensemble():
    _, _, test_loader = get_dataloaders()
    models, weights = load_ensemble()
    manifest = load_manifest()

    stacked, targets = collect_predictions(test_loader, models)
    targets_np = targets.numpy()

    print("Individual models:")
    individual_results = {}
    for entry, preds in zip(manifest, stacked):
        label = f"seed{entry['seed']}"
        metrics = _regression_metrics(targets_np, preds.numpy())
        individual_results[label] = metrics
        print(f"   {label:<10} RMSE {metrics['rmse']:.3f}  MAE {metrics['mae']:.3f}  R^2 {metrics['r2']:.3f}")

    best_individual = min(individual_results, key=lambda k: individual_results[k]["rmse"])
    print(f"   best individual model: {best_individual} (RMSE {individual_results[best_individual]['rmse']:.3f})")

    print("\nEnsemble (all 5 combined):")
    ensemble_results = {}
    disagreement = stacked.std(dim=0)
    for strategy in STRATEGIES:
        combined = combine_predictions(stacked, strategy=strategy, weights=weights)
        metrics = _regression_metrics(targets_np, combined.numpy())
        metrics["avg_disagreement"] = disagreement.mean().item()
        ensemble_results[strategy] = metrics
        print(
            f"   {strategy:<10} RMSE {metrics['rmse']:.3f}  MAE {metrics['mae']:.3f}  R^2 {metrics['r2']:.3f}  "
            f"avg_disagreement {metrics['avg_disagreement']:.3f}"
        )

    best_strategy = min(ensemble_results, key=lambda k: ensemble_results[k]["rmse"])
    print(f"   best ensemble strategy: {best_strategy} (RMSE {ensemble_results[best_strategy]['rmse']:.3f})")

    verdict = (
        "ensemble beats best individual model"
        if ensemble_results[best_strategy]["rmse"] < individual_results[best_individual]["rmse"]
        else "best individual model beats the ensemble"
    )
    print(f"\nVerdict: {verdict}")

    results = {
        "individual_models": individual_results,
        "ensemble": ensemble_results,
        "verdict": verdict,
    }

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved comparison to {METRICS_PATH}")

    return results


if __name__ == "__main__":
    evaluate_ensemble()

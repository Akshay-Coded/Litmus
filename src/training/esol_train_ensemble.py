"""Trains SolubilityGNN once per seed, saving each checkpoint to its own
path (models/esol_gnn_seed{1..5}.pt) for use as an ensemble.

Writes ENSEMBLE_MANIFEST_PATH recording which checkpoint goes with which
seed/pooling/validation score -- src/inference/esol_predictor.py reads
this to load the ensemble, so it never needs to hardcode the checkpoint
list.
"""

import json
import os

from src.config.esol_model_config import (
    ENSEMBLE_MANIFEST_PATH,
    MODEL_PATH_1,
    MODEL_PATH_2,
    MODEL_PATH_3,
    MODEL_PATH_4,
    MODEL_PATH_5,
    SEED_1,
    SEED_2,
    SEED_3,
    SEED_4,
    SEED_5,
)
from src.training.esol_train import train

SEED_MODEL_PATHS = [
    (SEED_1, MODEL_PATH_1),
    (SEED_2, MODEL_PATH_2),
    (SEED_3, MODEL_PATH_3),
    (SEED_4, MODEL_PATH_4),
    (SEED_5, MODEL_PATH_5),
]


def train_ensemble(pooling="mean"):
    manifest = []
    for seed, model_path in SEED_MODEL_PATHS:
        print(f"\n{'=' * 60}\nSeed: {seed} -> {model_path}\n{'=' * 60}")
        best_val_loss = train(pooling=pooling, model_path=model_path, seed=seed)
        manifest.append(
            {
                "seed": seed,
                "model_path": model_path,
                "pooling": pooling,
                "best_val_loss": best_val_loss,
            }
        )

    print(f"\n{'=' * 60}\nEnsemble training summary\n{'=' * 60}")
    for entry in manifest:
        print(f"seed {entry['seed']:<8} best_val_loss (normalized) {entry['best_val_loss']:.4f}")

    os.makedirs(os.path.dirname(ENSEMBLE_MANIFEST_PATH), exist_ok=True)
    with open(ENSEMBLE_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved ensemble manifest to {ENSEMBLE_MANIFEST_PATH}")

    return manifest


if __name__ == "__main__":
    train_ensemble()

"""Combines predictions from the trained ensemble of SolubilityGNN
checkpoints. This is the single place ensembling strategy lives -- both
offline evaluation (src/training/esol_evaluate_ensemble.py) and the
Streamlit UI call into this module, so switching strategies, or swapping
which checkpoints make up the ensemble (via ENSEMBLE_MANIFEST_PATH), only
needs to happen here.
"""

import json

import torch
from torch_geometric.data import Batch

from src.architectures.gnn import SolubilityGNN
from src.config.esol_model_config import DROPOUT, ENSEMBLE_MANIFEST_PATH, HIDDEN_DIM, N_LAYERS
from src.data_pipeline.featurizer import smiles_to_graph

SCALER_PATH = "data/processed/esol_target_scaler.json"
STRATEGIES = ("mean", "median", "weighted")


def load_manifest():
    with open(ENSEMBLE_MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_scaler():
    with open(SCALER_PATH, encoding="utf-8") as f:
        scaler = json.load(f)
    return scaler["mean"], scaler["std"]


def load_ensemble(device=None):
    """Loads every checkpoint listed in the ensemble manifest.

    Returns (models, weights). `weights` are normalized inverse-val-loss
    scores -- read only by the "weighted" strategy, ignored by the others.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest()

    models = []
    val_losses = []
    for entry in manifest:
        model = SolubilityGNN(
            hidden=HIDDEN_DIM, n_layers=N_LAYERS, dropout=DROPOUT, pooling=entry["pooling"]
        ).to(device)
        model.load_state_dict(torch.load(entry["model_path"], map_location=device))
        model.eval()
        models.append(model)
        val_losses.append(entry["best_val_loss"])

    inv_loss = torch.tensor([1.0 / v for v in val_losses])
    weights = inv_loss / inv_loss.sum()
    return models, weights


def combine_predictions(preds, strategy="mean", weights=None):
    """preds: tensor [n_models, n_samples] of un-normalized predictions."""
    if strategy == "mean":
        return preds.mean(dim=0)
    if strategy == "median":
        return preds.median(dim=0).values
    if strategy == "weighted":
        if weights is None:
            raise ValueError("weighted strategy requires weights")
        return (preds * weights.view(-1, 1)).sum(dim=0)
    raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")


@torch.no_grad()
def collect_predictions(loader, models, device=None):
    """Runs every model once over every batch in `loader` -- the shared
    primitive behind both per-model and combined-ensemble evaluation, so
    each model's forward pass only happens once regardless of how many
    strategies get compared afterward.

    Returns (stacked_preds, targets): stacked_preds is a tensor of shape
    [n_models, n_samples]; targets is a tensor, or None if the loader's
    graphs carry no `y`.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = _load_scaler()

    per_model_preds = []
    targets = None
    for model in models:
        batch_preds, batch_targets = [], []
        has_targets = True
        for batch in loader:
            batch = batch.to(device)
            batch_preds.append(model(batch).cpu() * std + mean)
            if getattr(batch, "y", None) is not None:
                batch_targets.append(batch.y.cpu() * std + mean)
            else:
                has_targets = False
        per_model_preds.append(torch.cat(batch_preds))
        if has_targets and targets is None:
            targets = torch.cat(batch_targets)

    stacked = torch.stack(per_model_preds, dim=0)
    return stacked, targets


def predict_loader(loader, models, weights, strategy="mean", device=None):
    """Combines every model's prediction over `loader` per the chosen
    strategy.

    Returns (preds, disagreement, targets) as numpy arrays -- `targets` is
    None if the loader's graphs carry no `y`. `disagreement` is the
    per-molecule std across the ensemble members (a free uncertainty
    signal, independent of which strategy combined the point estimate).
    """
    stacked, targets = collect_predictions(loader, models, device=device)
    combined = combine_predictions(stacked, strategy=strategy, weights=weights)
    disagreement = stacked.std(dim=0)

    return (
        combined.numpy(),
        disagreement.numpy(),
        targets.numpy() if targets is not None else None,
    )


@torch.no_grad()
def predict_smiles(smiles, models=None, weights=None, strategy="mean", device=None):
    """Single-molecule prediction from a raw SMILES string -- the function
    the Streamlit UI calls. Returns None if the SMILES can't be parsed.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if models is None:
        models, weights = load_ensemble(device=device)

    graph = smiles_to_graph(smiles)
    if graph is None:
        return None

    batch = Batch.from_data_list([graph]).to(device)
    mean, std = _load_scaler()

    preds = torch.stack([model(batch).cpu() * std + mean for model in models])  # [n_models, 1]
    combined = combine_predictions(preds, strategy=strategy, weights=weights)

    return {
        "prediction": combined.item(),
        "uncertainty": preds.std().item(),
        "per_model_predictions": preds.squeeze(-1).tolist(),
        "strategy": strategy,
    }

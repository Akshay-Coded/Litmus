import json
import os

import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.architectures.gnn import SolubilityGNN
from src.config.esol_model_config import DROPOUT, HIDDEN_DIM, MODEL_PATH_1, N_LAYERS
from src.data_pipeline.esol_dataset import get_dataloaders

SCALER_PATH = "data/processed/esol_target_scaler.json"
METRICS_PATH = "results/esol_test_metrics.json"


def evaluate(pooling="mean", model_path=None, metrics_path=None):
    model_path = model_path or MODEL_PATH_1
    metrics_path = metrics_path or METRICS_PATH
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, test_loader = get_dataloaders()

    with open(SCALER_PATH, encoding="utf-8") as f:
        scaler = json.load(f)
    mean, std = scaler["mean"], scaler["std"]

    model = SolubilityGNN(
        hidden=HIDDEN_DIM, n_layers=N_LAYERS, dropout=DROPOUT, pooling=pooling
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds_all, targets_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds_norm = model(batch)
            preds_all.append(preds_norm.cpu() * std + mean)
            targets_all.append(batch.y.cpu() * std + mean)

    preds_all = torch.cat(preds_all).numpy()
    targets_all = torch.cat(targets_all).numpy()

    rmse = mean_squared_error(targets_all, preds_all) ** 0.5
    mae = mean_absolute_error(targets_all, preds_all)
    r2 = r2_score(targets_all, preds_all)

    print("Test set performance (original log-solubility units, mol/L):")
    print(f"   - RMSE: {rmse:.3f}")
    print(f"   - MAE:  {mae:.3f}")
    print(f"   - R^2:  {r2:.3f}")

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"rmse": rmse, "mae": mae, "r2": r2}, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    return rmse, mae, r2


if __name__ == "__main__":
    evaluate()

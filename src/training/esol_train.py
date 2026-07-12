import copy
import json
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.architectures.gnn import SolubilityGNN
from src.config.esol_model_config import (
    BATCH_SIZE,
    DROPOUT,
    EARLY_STOP_PATIENCE,
    HIDDEN_DIM,
    LR,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MAX_EPOCHS,
    MIN_LR,
    MODEL_PATH_1,
    N_LAYERS,
    SEED_1,
    WEIGHT_DECAY,
)
from src.data_pipeline.esol_dataset import get_dataloaders

SCALER_PATH = "data/processed/esol_target_scaler.json"


def run_epoch(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)
            loss = criterion(preds, batch.y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def train(pooling="mean", model_path=None, seed=SEED_1):
    model_path = model_path or MODEL_PATH_1

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

    # target is z-scored (see src/data_pipeline/esol_dataset.py); std lets us report
    # RMSE in real log-solubility units during training: sqrt(MSE_norm) * std == RMSE_orig
    with open(SCALER_PATH, encoding="utf-8") as f:
        target_std = json.load(f)["std"]

    model = SolubilityGNN(
        hidden=HIDDEN_DIM, n_layers=N_LAYERS, dropout=DROPOUT, pooling=pooling
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=MIN_LR,
    )
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"SolubilityGNN (pooling={pooling}): {n_params:,} parameters")

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        val_rmse = (val_loss**0.5) * target_std

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        lr_after = optimizer.param_groups[0]["lr"]

        print(
            f"epoch {epoch:3d} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_rmse {val_rmse:.3f} | lr {lr_after:.2e}"
        )
        if lr_after < lr_before:
            print(
                f"   -> val_loss plateaued, LR reduced {lr_before:.2e} -> {lr_after:.2e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(
                    f"Early stopping at epoch {epoch} (no val improvement for {EARLY_STOP_PATIENCE} epochs)"
                )
                break

    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    print(
        f"Best val_loss (normalized): {best_val_loss:.4f}  (RMSE {(best_val_loss**0.5) * target_std:.3f})"
    )
    print(f"Saved best model to {model_path}")

    return best_val_loss


if __name__ == "__main__":
    train()

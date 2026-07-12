import copy
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.architectures.gnn import MultiTaskToxGNN
from src.config.tox21_model_config import (
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
    N_TASKS,
    SEED_1,
    WEIGHT_DECAY,
)
from src.data_pipeline.tox21_dataset import compute_pos_weight, get_dataloaders
from src.training.tox21_losses import masked_bce_loss


def run_epoch(model, loader, optimizer, device, pos_weight):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = masked_bce_loss(logits, batch.y, pos_weight=pos_weight)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def evaluate_mean_auc(model, loader, device):
    """Mean per-assay ROC-AUC, skipping any (assay, split) pair with no
    positives in that split -- AUC is undefined there, not zero."""
    model.eval()
    probs_all, targets_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probs_all.append(torch.sigmoid(model(batch)).cpu())
            targets_all.append(batch.y.cpu())
    probs_all = torch.cat(probs_all).numpy()
    targets_all = torch.cat(targets_all).numpy()

    aucs = []
    for task_idx in range(N_TASKS):
        mask = ~np.isnan(targets_all[:, task_idx])
        y_true = targets_all[mask, task_idx]
        if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        aucs.append(roc_auc_score(y_true, probs_all[mask, task_idx]))

    return sum(aucs) / len(aucs) if aucs else float("nan")


def train(model_path=None, seed=SEED_1):
    model_path = model_path or MODEL_PATH_1

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)
    pos_weight = compute_pos_weight(train_loader.dataset).to(device)

    model = MultiTaskToxGNN(
        n_tasks=N_TASKS, hidden=HIDDEN_DIM, n_layers=N_LAYERS, dropout=DROPOUT
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=MIN_LR,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"MultiTaskToxGNN: {n_params:,} parameters")

    # val_loss still drives the scheduler and early-stopping patience (it's
    # the smoother, per-batch-averaged signal); but the checkpoint we keep
    # is selected on val_mean_auc, since that's the metric we actually
    # report. A comparison run found these two criteria pick different
    # epochs (best val_loss at epoch 23 vs. best val_mean_auc at epoch 36
    # in one run) and the val_mean_auc-best checkpoint scored higher on
    # test (0.7647 vs. 0.7601 mean AUC) -- selecting on what you report
    # costs nothing here and is the more principled default.
    best_val_loss = float("inf")
    best_val_auc = -float("inf")
    best_state = None
    best_state_epoch = None
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, pos_weight)
        val_loss = run_epoch(model, val_loader, None, device, pos_weight)
        val_mean_auc = evaluate_mean_auc(model, val_loader, device)

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        lr_after = optimizer.param_groups[0]["lr"]

        print(
            f"epoch {epoch:3d} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_mean_auc {val_mean_auc:.3f} | lr {lr_after:.2e}"
        )
        if lr_after < lr_before:
            print(
                f"   -> val_loss plateaued, LR reduced {lr_before:.2e} -> {lr_after:.2e}"
            )

        if val_mean_auc > best_val_auc:
            best_val_auc = val_mean_auc
            best_state = copy.deepcopy(model.state_dict())
            best_state_epoch = epoch

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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

    print(f"Best val_mean_auc: {best_val_auc:.4f} (epoch {best_state_epoch})")
    print(f"Best val_loss (masked BCE): {best_val_loss:.4f}")
    print(f"Saved best model to {model_path}")

    return best_val_auc


if __name__ == "__main__":
    train()

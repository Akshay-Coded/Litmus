"""Joins featurized Tox21 graphs with the scaffold split and hands back
training-ready PyTorch Geometric DataLoaders, plus per-assay pos_weight.
"""

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

GRAPH_PATH = "data/processed/tox21_graphs.pt"
SPLIT_PATH = "data/processed/tox21_splits.csv"


def load_split_graphs(graph_path=GRAPH_PATH, split_path=SPLIT_PATH):
    """Loads the saved graphs and buckets them into train/val/test lists."""
    graphs = torch.load(graph_path, weights_only=False)
    split_df = pd.read_csv(split_path)
    smiles_to_split = dict(zip(split_df["smiles"], split_df["split"]))

    buckets = {"train": [], "val": [], "test": []}
    for graph in graphs:
        buckets[smiles_to_split[graph.smiles]].append(graph)

    return buckets["train"], buckets["val"], buckets["test"]


def compute_pos_weight(train_graphs):
    """
    Per-assay pos_weight = n_negative / n_positive, computed from the TRAIN
    split only (fitting on val/test would leak information, same
    discipline as the solubility target scaler). Untested (NaN) labels are
    excluded from both counts.
    """
    targets = torch.cat([g.y for g in train_graphs], dim=0)
    mask = ~torch.isnan(targets)
    targets_safe = torch.nan_to_num(targets, nan=0.0)

    n_pos = (targets_safe * mask).sum(dim=0)
    n_neg = ((1 - targets_safe) * mask).sum(dim=0)
    return n_neg / n_pos.clamp(min=1)


def get_dataloaders(batch_size=64):
    train_graphs, val_graphs, test_graphs = load_split_graphs()

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    print("Dataloaders ready.")
    print(f"   - train: {len(train_loader.dataset)} graphs, {len(train_loader)} batches")
    print(f"   - val:   {len(val_loader.dataset)} graphs, {len(val_loader)} batches")
    print(f"   - test:  {len(test_loader.dataset)} graphs, {len(test_loader)} batches")

    pos_weight = compute_pos_weight(train_loader.dataset)
    print(f"   - pos_weight (train): {[round(w, 2) for w in pos_weight.tolist()]}")

    batch = next(iter(train_loader))
    print(f"   - example batch: {batch}")

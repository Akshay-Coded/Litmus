"""Joins featurized graphs with the scaffold split and hands back
training-ready PyTorch Geometric DataLoaders.
"""

import json
import os

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

GRAPH_PATH = "data/processed/esol_graphs.pt"
SPLIT_PATH = "data/processed/esol_splits.csv"
SCALER_PATH = "data/processed/esol_target_scaler.json"


def load_split_graphs(graph_path=GRAPH_PATH, split_path=SPLIT_PATH):
    """Loads the saved graphs and buckets them into train/val/test lists."""
    graphs = torch.load(graph_path, weights_only=False)
    split_df = pd.read_csv(split_path)
    smiles_to_split = dict(zip(split_df["smiles"], split_df["split"]))

    buckets = {"train": [], "val": [], "test": []}
    for graph in graphs:
        buckets[smiles_to_split[graph.smiles]].append(graph)

    return buckets["train"], buckets["val"], buckets["test"]


def fit_target_scaler(train_graphs, save_path=SCALER_PATH):
    """
    Computes target mean/std from the TRAIN split only (fitting on val/test
    would leak information) and saves it so predictions can be un-normalized
    back to log(mol/L) at inference time -- the Streamlit UI will load this
    same file.
    """
    targets = torch.cat([g.y for g in train_graphs])
    mean, std = targets.mean().item(), targets.std().item()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)

    return mean, std


def apply_target_scaler(graphs, mean, std):
    for graph in graphs:
        graph.y = (graph.y - mean) / std
    return graphs


def get_dataloaders(batch_size=32, normalize_target=True):
    train_graphs, val_graphs, test_graphs = load_split_graphs()

    if normalize_target:
        mean, std = fit_target_scaler(train_graphs)
        apply_target_scaler(train_graphs, mean, std)
        apply_target_scaler(val_graphs, mean, std)
        apply_target_scaler(test_graphs, mean, std)

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

    batch = next(iter(train_loader))
    print(f"   - example batch: {batch}")

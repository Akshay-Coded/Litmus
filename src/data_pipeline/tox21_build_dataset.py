import os

import pandas as pd
import torch

from src.config.tox21_model_config import TASKS
from src.data_pipeline.featurizer import smiles_to_graph


def build_tox21_graph_dataset(input_path, output_path):
    """
    Converts every (smiles, 12 assay labels) row in the cleaned Tox21 CSV
    into a torch_geometric graph and saves the resulting list to disk. Each
    graph's target is a 12-length vector with NaN preserved for assays the
    molecule was never tested on -- masked_bce_loss handles those at
    training time.
    """
    print("Building Tox21 graph dataset...")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Could not find cleaned data at {input_path}. Run tox21_data_processing.py first."
        )

    df = pd.read_csv(input_path, dtype={task: "float64" for task in TASKS})

    graphs = []
    failed_smiles = []
    for smiles, target in zip(df["smiles"], df[TASKS].values.tolist()):
        graph = smiles_to_graph(smiles, target=target)
        if graph is None:
            failed_smiles.append(smiles)
        else:
            graphs.append(graph)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(graphs, output_path)

    print("Done.")
    print(f"   - Input molecules: {len(df)}")
    print(f"   - Converted to graphs: {len(graphs)}")
    print(f"   - Failed to parse: {len(failed_smiles)}")
    if failed_smiles:
        print(f"   - Failed SMILES: {failed_smiles}")
    print(f"   - Saved to: {output_path}")

    return graphs


if __name__ == "__main__":
    build_tox21_graph_dataset(
        input_path="data/processed/tox21_clean.csv",
        output_path="data/processed/tox21_graphs.pt",
    )

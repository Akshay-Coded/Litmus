import os

import pandas as pd
import torch

from src.data_pipeline.featurizer import smiles_to_graph


def build_graph_dataset(input_path, output_path):
    """
    Converts every (smiles, target_solubility) row in the cleaned ESOL CSV
    into a torch_geometric graph and saves the resulting list to disk.
    """
    print("Building graph dataset...")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Could not find cleaned data at {input_path}. Run esol_data_processing.py first."
        )

    df = pd.read_csv(input_path)

    graphs = []
    failed_smiles = []
    for smiles, target in zip(df["smiles"], df["target_solubility"]):
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
    build_graph_dataset(
        input_path="data/processed/esol_clean.csv",
        output_path="data/processed/esol_graphs.pt",
    )

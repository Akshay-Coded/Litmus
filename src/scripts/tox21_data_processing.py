import os

import pandas as pd
from rdkit import Chem

from src.config.tox21_model_config import TASKS
from src.data_pipeline.build_scaffold_split import build_scaffold_split


def clean_tox21_data(input_path, output_path):
    """
    Cleans the Tox21 dataset: validates every SMILES through RDKit, drops
    unparseable rows, canonicalizes and dedupes repeated structures (Tox21
    has known within-dataset redundancy), and keeps the 12 assay columns as
    floats so missing labels stay NaN rather than being coerced to 0.
    """
    print("Starting Tox21 Preprocessing...")

    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Could not find raw data at {input_path}. Ensure the file is placed there."
        )

    # Read the raw CSV file, keeping assay columns as float so NaN survives
    df = pd.read_csv(input_path, dtype={task: "float64" for task in TASKS})
    initial_rows = len(df)

    # 1. Validate SMILES via RDKit, drop unparseable rows
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    smiles_failed = df["mol"].isna().sum()
    df = df[df["mol"].notna()].reset_index(drop=True)

    # 2. Canonicalize and drop duplicate structures, keeping the first occurrence
    df["smiles"] = df["mol"].apply(Chem.MolToSmiles)
    duplicate_mask = df.duplicated(subset="smiles", keep="first")
    duplicates_dropped = duplicate_mask.sum()
    df = df[~duplicate_mask].reset_index(drop=True)

    # 3. Keep only the required columns
    processed_df = df[["smiles", "mol_id"] + TASKS]

    # 4. Ensure target directory exists and save the clean file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)

    print("Preprocessing complete!")
    print(f"   - Original Row Count: {initial_rows}")
    print(f"   - Dropped Rows (unparseable SMILES): {smiles_failed}")
    print(f"   - Dropped Rows (duplicate structures): {duplicates_dropped}")
    print(f"   - Final Processed Dataset: {len(processed_df)} molecules.")
    print(f"   - Saved to: {output_path}")

    return processed_df


if __name__ == "__main__":
    # Standard path routing mapping to your scalable project folder layout
    clean_tox21_data(
        input_path="data/raw/tox21.csv",
        output_path="data/processed/tox21_clean.csv",
    )
    build_scaffold_split(
        input_path="data/processed/tox21_clean.csv",
        output_path="data/processed/tox21_splits.csv",
    )

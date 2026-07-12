import os
import pandas as pd


def clean_esol_data(input_path, output_path):
    """
    Cleans the ESOL dataset by isolating the SMILES text representation
    and the target continuous log solubility value.
    """
    print("🚀 Starting ESOL Preprocessing...")

    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Could not find raw data at {input_path}. Ensure the file is placed there."
        )

    # Read the raw CSV file
    df = pd.read_csv(input_path)

    # 1. Map only the required columns
    required_cols = {
        "smiles": "smiles",
        "measured log solubility in mols per litre": "target_solubility",
    }

    # Extract and rename columns
    processed_df = df[list(required_cols.keys())].rename(columns=required_cols)

    # 2. Drop any rows with missing critical features (SMILES or Target)
    initial_rows = len(processed_df)
    processed_df = processed_df.dropna(subset=["smiles", "target_solubility"])
    dropped_count = initial_rows - len(processed_df)

    # 3. Ensure target directory exists and save the clean file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)

    print("✅ Preprocessing complete!")
    print(f"   - Original Row Count: {initial_rows}")
    print(f"   - Dropped Rows (NaNs): {dropped_count}")
    print(f"   - Final Processed Dataset: {len(processed_df)} molecules.")
    print(f"   - Saved to: {output_path}")


if __name__ == "__main__":
    # Standard path routing mapping to your scalable project folder layout
    clean_esol_data(
        input_path="data/raw/esol.csv",
        output_path="data/processed/esol_clean.csv",
    )

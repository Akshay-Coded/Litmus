import os

import pandas as pd

from src.data_pipeline.scaffold_split import scaffold_split


def build_scaffold_split(
    input_path, output_path, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42
):
    """
    Assigns each molecule in the cleaned ESOL CSV to a train/val/test split
    by scaffold group and saves the assignment as a lookup CSV.
    """
    print("Building scaffold split...")

    df = pd.read_csv(input_path)
    train_idx, val_idx, test_idx = scaffold_split(
        df["smiles"].tolist(),
        frac_train=frac_train,
        frac_val=frac_val,
        frac_test=frac_test,
        seed=seed,
    )

    split = pd.Series(index=df.index, dtype=object)
    split.iloc[train_idx] = "train"
    split.iloc[val_idx] = "val"
    split.iloc[test_idx] = "test"

    out = df[["smiles"]].copy()
    out["split"] = split

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)

    print("Done.")
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        print(f"   - {name}: {len(idx)} molecules ({len(idx) / len(df):.1%})")
    print(f"   - Saved to: {output_path}")

    return out


if __name__ == "__main__":
    build_scaffold_split(
        input_path="data/processed/esol_clean.csv",
        output_path="data/processed/esol_splits.csv",
    )

"""Domain-of-applicability for the Tox21 classifier.

Ran the same analysis as notebooks/error_analysis.ipynb section 5 (bucket
test-set error by Tanimoto similarity to the nearest training molecule),
but for Tox21 the signal is much weaker than it was for solubility:
per-molecule Brier score (averaged over that molecule's tested assays) by
similarity tercile was 0.204 / 0.194 / 0.193 -- essentially flat. The only
real effect shows up at the extreme: molecules below 0.3 similarity to
anything in training scored 0.220 vs ~0.19-0.21 everywhere else. So unlike
solubility, scaffold similarity here only supports one honest flag
("structurally novel"), not a fine-grained green/amber/red badge -- the
per-assay reliability tier (from evaluate_tox21.py's AUC standard error)
and the predicted probability's own distance from 0.5 do most of the real
work for confidence in this tab.
"""

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from src.config.tox21_model_config import TASKS

SPLIT_PATH = "data/processed/tox21_splits.csv"
CLEAN_DATA_PATH = "data/processed/tox21_clean.csv"

SIMILARITY_NOVEL = 0.3

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return _morgan_gen.GetFingerprint(mol)


def load_train_reference():
    """Returns (smiles_list, fingerprints, labels_df) for the training
    split -- the reference set every new molecule is compared against.
    Meant to be loaded once and cached by the caller.
    """
    splits = pd.read_csv(SPLIT_PATH)
    clean = pd.read_csv(CLEAN_DATA_PATH, dtype={t: "float64" for t in TASKS})
    train = splits[splits["split"] == "train"].merge(clean, on="smiles")

    smiles_list = train["smiles"].tolist()
    fingerprints = [fingerprint(s) for s in smiles_list]
    return smiles_list, fingerprints, train[TASKS]


def nearest_train_molecules(smiles, train_reference, k=3):
    """Returns (neighbors, max_similarity). `neighbors` is a list of the k
    most similar training molecules, each a dict with smiles/similarity/
    labels (the tested assays for that neighbor), sorted most-similar first.
    """
    train_smiles, train_fps, train_labels = train_reference
    query_fp = fingerprint(smiles)
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, train_fps)

    order = sorted(range(len(similarities)), key=lambda i: -similarities[i])[:k]
    neighbors = [
        {
            "smiles": train_smiles[i],
            "similarity": similarities[i],
            "labels": train_labels.iloc[i].dropna().to_dict(),
        }
        for i in order
    ]
    return neighbors, max(similarities)


def is_structurally_novel(max_similarity):
    """The one honest scaffold-based flag the data supports for Tox21 --
    see the module docstring for why this isn't a finer-grained badge."""
    return max_similarity < SIMILARITY_NOVEL

"""Domain-of-applicability: how similar is a new molecule to what the model
was actually trained on, and does that plus ensemble disagreement mean the
prediction should be trusted?

Thresholds are grounded in real measurements, not chosen by eye:

- SIMILARITY_RED / SIMILARITY_GREEN come from
  notebooks/esol_error_analysis.ipynb section 5 -- bucketed mean |error| on
  the test set was 0.919 below 0.3 similarity, 0.647-0.652 in the 0.3-0.7
  range, and 0.585 above 0.7. The sharp drop happens crossing 0.3, and
  error is essentially flat above 0.5, so those are the boundaries.
- DISAGREEMENT_GREEN / DISAGREEMENT_RED are the median (0.256) and 75th
  percentile (0.371) of the ensemble's per-molecule prediction std on the
  test set, where mean |error| was 0.673 in the bottom tercile of
  disagreement vs 0.817 in the top tercile (corr = 0.20).
"""

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

SPLIT_PATH = "data/processed/esol_splits.csv"
CLEAN_DATA_PATH = "data/processed/esol_clean.csv"

SIMILARITY_RED = 0.3
SIMILARITY_GREEN = 0.5
DISAGREEMENT_GREEN = 0.25
DISAGREEMENT_RED = 0.37

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return _morgan_gen.GetFingerprint(mol)


def load_train_reference():
    """Returns (smiles_list, fingerprints, true_solubilities) for the
    training split -- the reference set every new molecule is compared
    against. Meant to be loaded once and cached by the caller.
    """
    splits = pd.read_csv(SPLIT_PATH)
    clean = pd.read_csv(CLEAN_DATA_PATH)
    train = splits[splits["split"] == "train"].merge(clean, on="smiles")

    smiles_list = train["smiles"].tolist()
    fingerprints = [fingerprint(s) for s in smiles_list]
    targets = train["target_solubility"].tolist()
    return smiles_list, fingerprints, targets


def nearest_train_molecules(smiles, train_reference, k=3):
    """Returns (neighbors, max_similarity). `neighbors` is a list of the
    k most similar training molecules, each a dict with smiles/similarity/
    true_solubility, sorted most-similar first.
    """
    train_smiles, train_fps, train_targets = train_reference
    query_fp = fingerprint(smiles)
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, train_fps)

    order = sorted(range(len(similarities)), key=lambda i: -similarities[i])[:k]
    neighbors = [
        {
            "smiles": train_smiles[i],
            "similarity": similarities[i],
            "true_solubility": train_targets[i],
        }
        for i in order
    ]
    return neighbors, max(similarities)


def confidence_verdict(max_similarity, disagreement):
    """Combines structural novelty and ensemble disagreement into a single
    green/amber/red verdict with a human-readable reason.
    """
    if max_similarity < SIMILARITY_RED and disagreement > DISAGREEMENT_RED:
        return "red", (
            f"Novel scaffold (similarity {max_similarity:.2f} to nearest training molecule) "
            f"and the 5 ensemble models disagree with each other -- treat this prediction with caution."
        )
    if max_similarity < SIMILARITY_RED:
        return "red", (
            f"This molecule is structurally novel (similarity {max_similarity:.2f} to the nearest "
            f"training molecule, below the 0.3 threshold where test error jumps sharply) -- "
            f"it resembles a class of compound underrepresented in training."
        )
    if disagreement > DISAGREEMENT_RED:
        return "red", (
            f"The 5 ensemble models disagree with each other on this molecule "
            f"(spread {disagreement:.2f}, above the top-quartile threshold) -- treat with caution "
            f"even though the structure itself isn't unusual."
        )
    if max_similarity >= SIMILARITY_GREEN and disagreement <= DISAGREEMENT_GREEN:
        return "green", (
            f"In-domain: structurally similar to training chemistry (similarity {max_similarity:.2f}) "
            f"and the ensemble models agree (spread {disagreement:.2f})."
        )
    return "amber", (
        f"Borderline: similarity to training chemistry is {max_similarity:.2f} and model "
        f"disagreement is {disagreement:.2f} -- neither clearly in- nor out-of-domain."
    )

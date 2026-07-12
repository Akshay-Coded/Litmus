"""Bemis-Murcko scaffold splitting.

EDA (notebooks/esol_eda.ipynb, section 5) found ESOL's scaffolds are heavily
concentrated -- the top 10 scaffolds alone cover 64% of the dataset -- so a
random split would leak near-identical structures between train and test.
This groups molecules by scaffold first and keeps every molecule sharing a
scaffold in the same split.
"""

from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _scaffold_for_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def scaffold_split(smiles_list, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42):
    """
    Splits molecules into train/val/test by scaffold group.

    Groups are shuffled (for reproducible tie-breaking) then placed largest
    first, filling train up to frac_train, then val up to frac_val, with the
    remainder going to test. Returns three lists of indices into `smiles_list`.
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-8

    scaffold_to_indices = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold_to_indices[_scaffold_for_smiles(smiles)].append(idx)

    groups = list(scaffold_to_indices.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    n_total = len(smiles_list)
    n_train_target = frac_train * n_total
    n_val_target = frac_val * n_total

    train_idx, val_idx, test_idx = [], [], []
    for group in groups:
        if len(train_idx) < n_train_target:
            train_idx.extend(group)
        elif len(val_idx) < n_val_target:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    return train_idx, val_idx, test_idx

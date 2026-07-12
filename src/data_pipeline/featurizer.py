"""SMILES -> graph conversion.

smiles_to_graph() is the single reusable entry point: the batch dataset
builder calls it for every row in the training CSV, and the Streamlit UI
will call it directly on whatever SMILES a user types in, so predictions
at inference time use the exact same featurization as training.
"""

import torch
from rdkit import Chem
from torch_geometric.data import Data

from src.config.features import (
    ATOM_DEGREES,
    ATOM_HYBRIDIZATIONS,
    ATOM_SYMBOLS,
    BOND_STEREOS,
    BOND_TYPES,
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
)


def _one_hot(value, choices):
    encoding = [0] * len(choices)
    index = choices.index(value) if value in choices else choices.index("other")
    encoding[index] = 1
    return encoding


def _atom_features(atom):
    symbol = atom.GetSymbol() if atom.GetSymbol() in ATOM_SYMBOLS else "other"
    degree = atom.GetDegree() if atom.GetDegree() in ATOM_DEGREES else "other"
    hybridization = (
        atom.GetHybridization() if atom.GetHybridization() in ATOM_HYBRIDIZATIONS else "other"
    )

    features = (
        _one_hot(symbol, ATOM_SYMBOLS)
        + _one_hot(degree, ATOM_DEGREES)
        + [atom.GetFormalCharge()]
        + [atom.GetTotalNumHs()]
        + _one_hot(hybridization, ATOM_HYBRIDIZATIONS)
        + [int(atom.GetIsAromatic())]
        + [int(atom.IsInRing())]
        + [atom.GetNumRadicalElectrons()]
    )
    return features


def _bond_features(bond):
    stereo = bond.GetStereo() if bond.GetStereo() in BOND_STEREOS else "other"
    features = (
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + [int(bond.GetIsConjugated())]
        + [int(bond.IsInRing())]
        + _one_hot(stereo, BOND_STEREOS)
    )
    return features


def smiles_to_graph(smiles, target=None):
    """Convert a single SMILES string into a torch_geometric Data graph.

    Returns None if RDKit can't parse the SMILES (e.g. malformed user input).
    `target` is optional so this same function works for labeled training
    rows and for unlabeled molecules typed into the UI.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = [_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float).view(-1, NODE_FEATURE_DIM)

    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_feat = _bond_features(bond)
        # undirected bond -> two directed edges, each carrying the same features
        edge_indices += [[i, j], [j, i]]
        edge_features += [bond_feat, bond_feat]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float).view(-1, EDGE_FEATURE_DIM)
    else:
        # single-atom molecules have no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    if target is not None:
        data.y = torch.tensor([target], dtype=torch.float)

    return data

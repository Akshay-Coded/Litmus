"""Feature vocabulary for converting molecules (SMILES) into graphs.

This is the single source of truth for the atom/bond feature encoding.
Both the batch dataset builder and the interactive UI import from here,
so a SMILES from a user's typed input is featurized identically to one
from the training CSV.
"""

from rdkit.Chem import rdchem

# --- Atom (node) features ---

ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "other"]

ATOM_DEGREES = [0, 1, 2, 3, 4, "other"]

ATOM_HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    "other",
]

# one-hot(symbol) + one-hot(degree) + formal_charge + total_num_Hs
# + one-hot(hybridization) + is_aromatic + is_in_ring + num_radical_electrons
NODE_FEATURE_DIM = (
    len(ATOM_SYMBOLS) + len(ATOM_DEGREES) + 1 + 1 + len(ATOM_HYBRIDIZATIONS) + 1 + 1 + 1
)

# --- Bond (edge) features ---

BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]

BOND_STEREOS = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
    "other",
]

# one-hot(bond_type) + is_conjugated + is_in_ring + one-hot(stereo)
EDGE_FEATURE_DIM = len(BOND_TYPES) + 1 + 1 + len(BOND_STEREOS)

"""Tox21 model/training hyperparameters -- single source of truth shared by
build_tox21_dataset.py, tox21_dataset.py, train_tox21.py, and evaluate_tox21.py.
"""

TASKS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]
N_TASKS = len(TASKS)
NR_TASKS = [t for t in TASKS if t.startswith("NR-")]
SR_TASKS = [t for t in TASKS if t.startswith("SR-")]

HIDDEN_DIM = 128
N_LAYERS = 3
DROPOUT = 0.2

LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64  # Tox21 (~7,800 molecules) is ~7x ESOL's size

# ReduceLROnPlateau, same reasoning as the solubility config: reactive
# rather than a fixed schedule, patience kept below EARLY_STOP_PATIENCE
# so the LR gets cut before training stops.
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 8
MIN_LR = 1e-5

MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 15

SEED_1 = 42

MODEL_PATH_1 = "models/tox21_gnn_seed1.pt"

# EDA (notebooks/tox21_eda.ipynb, section 3) found every assay keeps >=20
# test-fold positives after the scaffold split -- no assay needs to be
# dropped or specially flagged for insufficient support.
LOW_COUNT_THRESHOLD = 20

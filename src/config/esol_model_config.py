"""ESOL model/training hyperparameters -- single source of truth shared by
esol_train.py, esol_evaluate.py, and the UI's model-loading code.
"""

HIDDEN_DIM = 128
N_LAYERS = 3
DROPOUT = 0.2

LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32

# ReduceLROnPlateau: reactive rather than a fixed schedule (e.g. cosine),
# since val_loss on this small dataset is noisy, not smoothly decreasing.
# Its patience must stay below EARLY_STOP_PATIENCE so the LR gets cut
# before training stops.
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 8
MIN_LR = 1e-5

MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20

SEED_1 = 42
SEED_2 = 123
SEED_3 = 456
SEED_4 = 789
SEED_5 = 101112

MODEL_PATH_1 = "models/esol_gnn_seed1.pt"
MODEL_PATH_2 = "models/esol_gnn_seed2.pt"
MODEL_PATH_3 = "models/esol_gnn_seed3.pt"
MODEL_PATH_4 = "models/esol_gnn_seed4.pt"
MODEL_PATH_5 = "models/esol_gnn_seed5.pt"

# Records each ensemble member's checkpoint path, pooling, and validation
# performance -- src/inference/esol_predictor.py reads this to load the
# ensemble without hardcoding which seeds/paths exist, and to weight
# members by how well they actually did.
ENSEMBLE_MANIFEST_PATH = "models/esol_ensemble_manifest.json"

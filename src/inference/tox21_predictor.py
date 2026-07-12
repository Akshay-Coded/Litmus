"""Single-molecule inference for the Tox21 multi-task classifier -- the
Streamlit toxicity tab calls predict_smiles() directly, the same shape as
src/inference/predictor.py does for solubility.

Unlike solubility, there's no ensemble here (see MODEL_PATH_1 in
tox21_model_config.py) -- one seed is trained so far -- so the only native
uncertainty signal is the predicted probability itself (near 0.5 = model
genuinely unsure) plus the per-assay reliability tier computed offline in
evaluate_tox21.py from each assay's test-set support.
"""

import json

import torch
from torch_geometric.data import Batch

from src.architectures.gnn import MultiTaskToxGNN
from src.config.tox21_model_config import DROPOUT, HIDDEN_DIM, MODEL_PATH_1, N_LAYERS, N_TASKS, TASKS
from src.data_pipeline.featurizer import smiles_to_graph

METRICS_PATH = "results/tox21_test_metrics.json"


def load_model(model_path=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_path or MODEL_PATH_1

    model = MultiTaskToxGNN(
        n_tasks=N_TASKS, hidden=HIDDEN_DIM, n_layers=N_LAYERS, dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_assay_reliability(metrics_path=None):
    """Per-assay {auc, auprc, n_test, n_pos, reliability} from the offline
    test evaluation -- the UI reads this rather than recomputing it, so the
    reliability badge always matches whatever checkpoint is actually loaded.
    """
    metrics_path = metrics_path or METRICS_PATH
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)["per_assay"]


@torch.no_grad()
def predict_smiles(smiles, model=None, device=None):
    """Single-molecule multi-task prediction from a raw SMILES string.
    Returns None if the SMILES can't be parsed.

    Returns a dict with a "probabilities" mapping of assay -> P(toxic).
    These are relative activity scores, not calibrated likelihoods -- no
    temperature scaling or other calibration has been applied, so a value
    of 0.7 means "more likely active than 0.3", not "70% probability".
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = load_model(device=device)

    graph = smiles_to_graph(smiles)
    if graph is None:
        return None

    batch = Batch.from_data_list([graph]).to(device)
    logits = model(batch).squeeze(0).cpu()
    probs = torch.sigmoid(logits)

    return {
        "probabilities": dict(zip(TASKS, probs.tolist())),
        "logits": dict(zip(TASKS, logits.tolist())),
    }

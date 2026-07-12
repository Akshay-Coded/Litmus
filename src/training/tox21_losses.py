"""Masked BCE loss for multi-task binary classification with sparse labels.

EDA locked masked BCE (notebooks/tox21_eda.ipynb, section 1): missing labels
must contribute nothing to the loss or gradient, not be treated as a
negative. See src/test/test_tox21_masked_loss.py for the mask correctness
check.
"""

import torch
import torch.nn.functional as F


def masked_bce_loss(logits, targets, pos_weight=None):
    """
    logits, targets: [batch, n_tasks]. targets has NaN where that assay was
    never tested for that molecule -- those (molecule, assay) pairs are
    zeroed out before averaging, so they never touch the gradient.
    """
    mask = ~torch.isnan(targets)
    targets_safe = torch.nan_to_num(targets, nan=0.0)

    per_element = F.binary_cross_entropy_with_logits(
        logits, targets_safe, pos_weight=pos_weight, reduction="none"
    )
    per_element = per_element * mask
    return per_element.sum() / mask.sum()

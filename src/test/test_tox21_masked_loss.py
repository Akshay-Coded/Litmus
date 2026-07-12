"""Sanity check for masked_bce_loss -- the single highest-risk bug in the
Tox21 pipeline. A batch with deliberately-inserted NaN labels (mirroring
Tox21's real sparsity) must produce a finite loss, and untested
(molecule, assay) pairs must receive exactly zero gradient. A mask that
leaks would silently corrupt every downstream metric.

Run with: python -m src.test.test_tox21_masked_loss
"""

import torch

from src.training.tox21_losses import masked_bce_loss


def test_masked_loss_ignores_untested_labels():
    torch.manual_seed(0)
    batch_size, n_tasks = 8, 12

    logits = torch.randn(batch_size, n_tasks, requires_grad=True)
    targets = torch.randint(0, 2, (batch_size, n_tasks)).float()

    missing_mask = torch.rand(batch_size, n_tasks) < 0.5
    targets[missing_mask] = float("nan")
    assert missing_mask.any() and (~missing_mask).any(), "need both missing and tested labels"

    loss = masked_bce_loss(logits, targets)
    assert torch.isfinite(loss), "loss must be finite even with NaN labels present"

    loss.backward()
    assert logits.grad is not None
    assert torch.all(logits.grad[missing_mask] == 0), "untested pairs must get zero gradient"
    assert torch.any(logits.grad[~missing_mask] != 0), "tested pairs should get nonzero gradient"

    print("masked_bce_loss: finite loss, untested pairs get zero gradient. OK.")


if __name__ == "__main__":
    test_masked_loss_ignores_untested_labels()
    print("All masked loss checks passed.")

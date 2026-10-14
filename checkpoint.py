from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    step: int,
    config: dict,
) -> None:
    """Save everything needed to rebuild the model or resume training.

    `epoch` is the epoch just completed (0-based), `step` the global step after it.
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": config,
    }
    # Write then rename, so a crash mid-save never leaves a corrupt checkpoint
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)

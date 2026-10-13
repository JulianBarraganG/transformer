import logging
from collections.abc import Callable

from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from telemetry import RunLogger, TrainStepRecord
from transformer import Transformer

logger = logging.getLogger(__name__)

LossFunction = Callable[[Tensor, Tensor], Tensor]

def train_one_epoch_transformer(
    model: Transformer,
    training_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    loss_fn: LossFunction,
    pad_idx: int,
    epoch: int,
    step: int,
    run_logger: RunLogger,
) -> tuple[float, int]:
    """Train for one epoch, starting at global `step`.

    Returns the token-weighted mean loss over the epoch and the global step
    to continue from.
    """
    loss_sum = 0.
    token_sum = 0
    device = next(model.parameters()).device

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        source = data[0].to(device)
        targets = data[1].to(device) # exclude EOS or PAD
        dec_in, labels = targets[:, :-1], targets[:, 1:]

        # Must zero grads before each batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(source, dec_in)

        # Compute the loss and its gradients
        # Reshape out (B, T, V) -> (B*T, V) and flatten labels
        loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
        loss.backward()

        # Gather data to be recorded, between backward() and step()
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), float("inf") # Inf norm -> no clip
        ).item()
        n_tokens = int((labels != pad_idx).sum().item())
        lr = scheduler.get_last_lr()[0]
        batch_loss = loss.item()
        run_logger.log(
            TrainStepRecord(step, epoch, batch_loss, n_tokens, grad_norm, lr)
        )

        # Adjust learning weights
        optimizer.step()
        scheduler.step()
        step += 1

        loss_sum += batch_loss * n_tokens
        token_sum += n_tokens

    return loss_sum / token_sum, step


def train_transformer(
    model: Transformer,
    training_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    loss_fn: LossFunction,
    pad_idx: int,
    epochs: int,
    run_logger: RunLogger,
) -> None:
    step = 0
    for e in range(epochs):
        model.train(True)
        avg_loss, step = train_one_epoch_transformer(
            model, training_loader, optimizer, scheduler, loss_fn, pad_idx, e, step,
            run_logger,
        )

        logger.info(f"Loss at epoch {e + 1}: {avg_loss}")

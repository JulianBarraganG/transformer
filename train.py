import logging

from torch import Tensor, nn, no_grad
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from tracking.checkpoint import save_checkpoint
from tracking.tracking import RunLogger, TrainStepRecord, ValRecord
from transformer import Transformer

logger = logging.getLogger(__name__)


class SeqCrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(**kwargs)

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        return self.ce(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def train_one_epoch_transformer(
    model: Transformer,
    training_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    loss_fn: SeqCrossEntropyLoss,
    pad_idx: int,
    epoch: int,
    step: int,
    run_logger: RunLogger,
) -> tuple[float, int]:
    """Train for one epoch, starting at global `step`.

    Returns a tuple of the token-weighted mean loss over the epoch and the
    global step to continue from.
    """
    loss_sum = 0.0
    token_sum = 0
    device = next(model.parameters()).device

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        source = data[0].to(device)
        targets = data[1].to(device)  # exclude EOS or PAD
        dec_in, labels = targets[:, :-1], targets[:, 1:]

        # Must zero grads before each batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(source, dec_in)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Gather data to be recorded, between backward() and step()
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            float("inf"),  # Inf norm -> no clip
        ).item()
        n_tokens = int((labels != pad_idx).sum().item())
        lr = scheduler.get_last_lr()[0]
        lr = lr.item() if isinstance(lr, Tensor) else lr  # type stuff
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

    return (loss_sum / token_sum), step


@no_grad()
def _validate_transformer(
    model: Transformer,
    val_loader: DataLoader,
    loss_fn: SeqCrossEntropyLoss,
    pad_idx: int,
) -> tuple[float, int]:
    device = next(model.parameters()).device
    model.eval()

    val_loss_sum = 0.0
    token_sum = 0

    for data in val_loader:
        source = data[0].to(device)
        targets = data[1].to(device)  # exclude EOS or PAD
        dec_in, labels = targets[:, :-1], targets[:, 1:]

        outputs = model(source, dec_in)

        n_tokens = int((labels != pad_idx).sum().item())
        token_sum += n_tokens

        batch_val_loss = loss_fn(outputs, labels).item()
        val_loss_sum += batch_val_loss * n_tokens

    model.train(True)

    return (val_loss_sum / token_sum), token_sum


def train_transformer(
    model: Transformer,
    training_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    loss_fn: SeqCrossEntropyLoss,
    pad_idx: int,
    epochs: int,
    run_logger: RunLogger,
) -> None:
    step = 0
    best_val = float("inf")
    ckpt_dir = run_logger.run_dir / "checkpoints"
    for e in range(epochs):
        model.train(True)
        avg_loss, step = train_one_epoch_transformer(
            model,
            training_loader,
            optimizer,
            scheduler,
            loss_fn,
            pad_idx,
            e,
            step,
            run_logger,
        )

        logger.info(f"Train loss at epoch {e + 1}: {avg_loss}")
        val_loss, val_token_sum = _validate_transformer(
            model, val_loader, loss_fn, pad_idx
        )
        run_logger.log(ValRecord(step, e, val_loss, val_token_sum))
        logger.info(f"Validation loss at epoch {e + 1}: {val_loss}")

        save_checkpoint(
            ckpt_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            e,
            step,
            run_logger.config,
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                ckpt_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                e,
                step,
                run_logger.config,
            )

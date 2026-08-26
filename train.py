from collections.abc import Callable
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from transformer import Transformer


LossFunction = Callable[[Tensor, Tensor], Tensor]

def train_one_epoch_transformer(
    model: Transformer,
    training_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    loss_fn: LossFunction,
) -> float:
    running_loss = 0.
    last_loss = 0.
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

        # Adjust learning weights
        optimizer.step()
        scheduler.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            running_loss = 0.
    return last_loss

def train_transformer(
    model: Transformer,
    training_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    loss_fn: LossFunction,
    epochs: int
) -> None:
    for e in range(epochs):
        model.train(True)
        avg_loss = train_one_epoch_transformer(model, training_loader, optimizer, scheduler, loss_fn)

        print(f"Loss at epoch {e + 1}: {avg_loss}")

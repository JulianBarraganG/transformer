import logging

import torch
from torch.nn import CrossEntropyLoss
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from mock_data import PAD, CopyDataset, collate_copy
from telemetry import RunLogger
from train import train_transformer
from transformer import Transformer

TRAIN_PARAMS = {
    "warmup": 40,
    "batch_size": 64,
    "epochs": 2,
    "lr": 1.0,
    "betas": (0.9, 0.98),
    "eps": 1e-9,
}
TRANSFORMER_PARAMS = {
    "max_seq_len": 14,
    "vocab_size": 32,
    "padding_idx": PAD,
    "d_model": 64,
    "d_ff": 252,
    "d_k": 8,
    "d_v": 8,
    "N": 2,
}

logger = logging.getLogger(__name__)

# Prep the data
dataset = CopyDataset(10000, TRANSFORMER_PARAMS.get("vocab_size", 32))
loader = DataLoader(
    dataset, batch_size=TRAIN_PARAMS["batch_size"], shuffle=True, collate_fn=collate_copy
)
#Prep the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    logger.warning("Device is NOT on CUDA. Restart now if unintentional.")
model = Transformer(**TRANSFORMER_PARAMS).to(device)
loss_fn = CrossEntropyLoss(ignore_index=PAD)
optimizer = Adam(
    params=model.parameters(),
    lr=TRAIN_PARAMS["lr"],
    betas=TRAIN_PARAMS["betas"],
    eps=TRAIN_PARAMS["eps"],
)
scheduler = LambdaLR(
    optimizer,
    lambda step: model.d_model**(-0.5) * min(
        max(step, 1)**-0.5, max(step, 1) * TRAIN_PARAMS["warmup"]**-1.5
    ),
)

# Do the training
print("starting training...")
if __name__ == "__main__":
    logging.basicConfig(filename="transformer.log", level=logging.INFO)
    logger.info("Starting training...")
    run_logger = RunLogger(
        "copy", config={"model": TRANSFORMER_PARAMS, "train": TRAIN_PARAMS}
    )
    train_transformer(
        model, loader, optimizer, scheduler, loss_fn, PAD,
        epochs=TRAIN_PARAMS["epochs"], run_logger=run_logger,
    )
    logger.info("Training finished.")

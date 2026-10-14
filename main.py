import logging
from functools import partial
from pathlib import Path

import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dates_data import DatesDataset, collate_dates, load_vocab
from tracking.tracking import RunLogger
from train import SeqCrossEntropyLoss, train_transformer
from transformer import Transformer

DATA_DIR = Path(__file__).parent / "data" / "dates"

TRAIN_PARAMS = {
    "warmup": 400,
    "batch_size": 64,
    "epochs": 20,
    "lr": 1.0,
    "betas": (0.9, 0.98),
    "eps": 1e-9,
}
# vocab_size and padding_idx are filled in from the vocab at startup
TRANSFORMER_PARAMS = {
    "max_seq_len": 31,
    "d_model": 64,
    "d_ff": 252,
    "d_k": 8,
    "d_v": 8,
    "N": 2,
}

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(filename="transformer.log", level=logging.INFO)

    # Prep the data
    vocab = load_vocab(DATA_DIR / "vocab.json")
    collate = partial(collate_dates, vocab=vocab)
    train_ds = DatesDataset(DATA_DIR / "train.jsonl", vocab)
    val_ds = DatesDataset(DATA_DIR / "valid.jsonl", vocab)
    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=TRAIN_PARAMS["batch_size"], shuffle=False, collate_fn=collate
    )
    model_params = TRANSFORMER_PARAMS | {
        "vocab_size": len(vocab),
        "padding_idx": vocab.pad_id,
    }

    # Prep the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("Device is NOT on CUDA. Restart now if unintentional.")
    if device.type == "cuda":
        logger.info("Device moved to CUDA")
    model = Transformer(**model_params).to(device)
    loss_fn = SeqCrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = Adam(
        params=model.parameters(),
        lr=TRAIN_PARAMS["lr"],
        betas=TRAIN_PARAMS["betas"],
        eps=TRAIN_PARAMS["eps"],
    )
    scheduler = LambdaLR(
        optimizer,
        lambda step: model.d_model ** (-0.5)
        * min(max(step, 1) ** -0.5, max(step, 1) * TRAIN_PARAMS["warmup"] ** -1.5),
    )

    # Do the training
    logger.info("Starting training...")
    run_logger = RunLogger(
        "dates", config={"model": model_params, "hyper": TRAIN_PARAMS}
    )
    train_transformer(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        vocab.pad_id,
        epochs=TRAIN_PARAMS["epochs"],
        run_logger=run_logger,
    )
    party_pooper = "\U0001f389"  # celebratory unicode emoji
    logger.info(f"Training finished! {party_pooper}")

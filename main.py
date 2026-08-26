import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from train import train_transformer
from mock_data import PAD, CopyDataset, collate_copy
from transformer import Transformer
from torch.nn import CrossEntropyLoss

WARMUP: int = 40

# Prep the data
vocab_size, batch_size = 32, 64
dataset = CopyDataset(10000, vocab_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_copy)
#Prep the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("WARNING: Model is on CPU")
    raise ValueError("Restart")
model = Transformer(14, vocab_size, 0, d_model=64, d_ff=252, d_k=8, d_v=8, N=2).to(device)
print(model)
#model = Transformer(14, vocab_size, 0).to(device)
loss_fn = CrossEntropyLoss(ignore_index=PAD)
optimizer = Adam(params=model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
scheduler = LambdaLR(
    optimizer,
    lambda step: model.d_model**(-0.5) * min(
        max(step, 1)**-0.5, max(step, 1) * WARMUP**-1.5
    ),
)

# Do the training
print("starting training...")
train_transformer(model, loader, optimizer, scheduler, loss_fn, epochs=4)

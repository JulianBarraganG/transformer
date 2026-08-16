import math

import torch
from torch import nn
from torch.optim import ada

from transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pad_idx, batch_size, seq_len, vocab_size = 10, 32, 128, 1000
m = Transformer(seq_len, vocab_size, pad_idx)
print(m.embedding.emb_layer.weight is m.output_head.weight)
print(sum(p.numel() for p in m.parameters()))
print("test forward")
# rand token ids: (batch, seq) -> embeds to (batch, seq, d_model)
X = torch.randint(0, vocab_size, (batch_size, seq_len))
Y = torch.randint(0, vocab_size, (batch_size, seq_len))
X[0, 124:] = 10

out = m(X, Y)
print(f"Result shape: {tuple(out.shape)}") # expect (32, 128, 1000)
m.eval()  # dropout off, else the number is noisy
with torch.no_grad():
    logits = m(X, Y)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
print(f"initial loss {loss.item():.3f} (want ~{math.log(vocab_size):.3f})")

def train_one_epoch(model: nn.Module):
    # zero grads
    # inference
    # loss
    # backwards
    # step

def train(epochs: int):
    pass

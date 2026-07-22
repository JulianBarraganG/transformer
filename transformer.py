import torch
import torch.nn as nn
import torch.optim as optim


class PositionalEncoding(nn.Module):
    pass

class FeedForward(nn.Module):
    """Position wise feed forward network"""
    pass

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, h: int = 8, d_v: int | None = None) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0, f"d_model={d_model} must be divisible by h={h}"
        self.d_k = d_model // h
        self.d_v = d_v if d_v is not None else self.d_k
        self.W_O = nn.Linear((h * self.d_v), d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        """Scaled dot product attention."""
    pass

class EncoderLayer(nn.Module):
    pass

class DecoderLayer(nn.Module):
    pass

class Transformer(nn.Module):
    pass

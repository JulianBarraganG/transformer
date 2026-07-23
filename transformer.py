import torch
import torch.nn as nn
from torch.nn.functional import softmax
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
        self.W_Q = nn.Linear(d_model, self.d_k)
        self.W_K = nn.Linear(d_model, self.d_k)
        self.W_V = nn.Linear(d_model, self.d_v)

    def scaled_dot_product_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Scaled dot product attention."""
        assert Q.shape[1] == K.shape[1] == self.d_k,(
            f"Expected Q and K to have {self.d_k} i.e. d_k columns"
        )
        assert V.shape[1] == self.d_v, f"Expected V to have {self.d_v} i.e. d_v columns"
        scaled_dot_prods = softmax(torch.matmul(Q, torch.t(K)) / self.d_k)

        return torch.matmul(scaled_dot_prods, V)

    def multi_head(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        pass

class EncoderLayer(nn.Module):
    pass

class DecoderLayer(nn.Module):
    pass

class Transformer(nn.Module):
    pass

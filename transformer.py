import torch
from torch import nn
from torch.nn.functional import softmax


class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class FeedForward(nn.Module):
    """Position wise feed forward network"""
    def __init__(self, d_model: int = 512, d_ff: int = 2048) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.layer_1 = nn.Linear(self.d_model, self.d_ff)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(self.d_ff, self.d_model)

    def forward(self, X: torch.Tensor):
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.layer_2(out)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, h: int = 8, d_v: int | None = None) -> None:
        super().__init__()
        assert d_model % h == 0, f"d_model={d_model} must be divisible by h={h}"
        self.d_k = d_model // h
        self.d_v = d_v if d_v is not None else self.d_k
        self.h = h # number of heads in multi-head attention
        self.W_O = nn.Linear((self.h * self.d_v), d_model)
        self.W_Q = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(self.h)])
        self.W_K = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(self.h)])
        self.W_V = nn.ModuleList([nn.Linear(d_model, self.d_v) for _ in range(self.h)])

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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Calculate h head_i = attention(QW^Q_i,...)
        heads = []
        for i in range(self.h):
            Qi = self.W_Q[i](X)
            Ki = self.W_K[i](X)
            Vi = self.W_V[i](X)
            heads.append(self.scaled_dot_product_attention(Qi, Ki, Vi))
        cat = torch.concat(heads, dim=1)
        return self.W_O(cat)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        h: int = 8,
        d_model: int = 512,
        d_ff: int = 2048,
        d_v: int | None = None,
    ) -> None:
        super().__init__()
        self.mha= MultiHeadAttention(d_model=d_model, h=h, d_v=d_v)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.layer_norm_mha = nn.LayerNorm(d_model)
        self.layer_norm_ff = nn.LayerNorm(d_model)

        def forward(self, X: torch.Tensor) -> None:
            # Sub-layer one (multi-head attention)
            Z = self.mha(X)
            Z = self.layer_norm_mha(X + Z) # residual after X
            # Sub-layer two (feed-forward network)
            R = self.ff(Z)
            R = self.layer_norm_ff(Z + R) # residual after Z
            return R


class DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class Transformer(nn.Module):
    def __init__(
        self, 
        h: int = 8,
        N: int = 6,
        d_model: int = 512,
        d_ff: int = 2048,
        d_v: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer() for _ in range(N)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer() for _ in range(N)]
        )

import torch
from torch import nn
from torch.nn.functional import softmax

QueryKeyPair: type = tuple[torch.Tensor, torch.Tensor]

class PositionalEncoding(nn.Module):
    # probably not an nn.Module
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError()

class FeedForward(nn.Module):
    """Position wise feed forward network"""
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.layer_1 = nn.Linear(self.d_model, self.d_ff)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(self.d_ff, self.d_model)

    def forward(self, X: torch.Tensor):
        out = self.layer_1(X)
        out = self.relu(out)
        out = self.layer_2(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self, h: int, d_model: int, d_v: int | None, masked: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % h == 0, f"d_model={d_model} must be divisible by h={h}"
        self.d_k = d_model // h
        self.d_v = d_v if d_v is not None else self.d_k
        self.h = h # number of heads in multi-head attention
        self.mask_inputs = masked
        self.W_O = nn.Linear((self.h * self.d_v), d_model)
        self.W_Q = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(self.h)])
        self.W_K = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(self.h)])
        self.W_V = nn.ModuleList([nn.Linear(d_model, self.d_v) for _ in range(self.h)])

    def _mask_future_inputs(self, sm_input: torch.Tensor) -> torch.Tensor:
        assert sm_input.shape[0] == sm_input.shape[1],(
            f"Expected square tensor matrix input, got ({in_shape[0], in_shape[1]})."
        )
        seq_len = sm_input.shape[0]
        mask = torch.full(sm_input.shape, False)
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    mask[i, j] = True
                    # TODO: might be a faster way

        return sm_input.masked_fill(mask, -torch.inf)

    def scaled_dot_product_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Scaled dot product attention."""
        assert Q.shape[1] == K.shape[1] == self.d_k,(
            f"Expected Q and K to have {self.d_k} i.e. d_k columns"
        )
        assert V.shape[1] == self.d_v, f"Expected V to have {self.d_v} i.e. d_v columns"
        softmax_input = torch.matmul(Q, torch.t(K)) / self.d_k
        if self.mask_inputs:
            softmax_input = self._mask_future_inputs(softmax_input)
        scaled_dot_prods = softmax(softmax_input, dim=1)

        return torch.matmul(scaled_dot_prods, V)

    def forward(self, X: torch.Tensor, memory: torch.Tensor | None = None) -> torch.Tensor:
        # Calculate h head_i = attention(QW^Q_i,...)
        heads = []
        key_val_source = X if memory is None else memory
        for i in range(self.h):
            Qi = self.W_Q[i](X)
            Ki = self.W_K[i](key_val_source)
            Vi = self.W_V[i](key_val_source)
            heads.append(self.scaled_dot_product_attention(Qi, Ki, Vi))
        cat = torch.concat(heads, dim=1)
        return self.W_O(cat)


class EncoderLayer(nn.Module):
    def __init__(self, h: int, d_model: int, d_ff: int, d_v: int | None) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, h=h, d_v=d_v)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.layer_norm_self = nn.LayerNorm(d_model)
        self.layer_norm_ff = nn.LayerNorm(d_model)

    def forward(self, X: torch.Tensor) -> None:
        # Sub-layer one (self-attention)
        Z = self.self_attention(X)
        Z = self.layer_norm_self(X + Z) # residual after X
        # Sub-layer two (feed-forward network)
        R = self.ff(Z)
        R = self.layer_norm_ff(Z + R) # residual after Z
        return R


class DecoderLayer(nn.Module):
    def __init__(self, h: int, d_model: int, d_ff: int, d_v: int | None) -> None:
        super().__init__()
        self.encdec_attention = MultiHeadAttention(d_model=d_model, h=h, d_v=d_v)
        self.self_attention = MultiHeadAttention(d_model=d_model, h=h, d_v=d_v, masked=True)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.layer_norm_encdec = nn.LayerNorm(d_model)
        self.layer_norm_self = nn.LayerNorm(d_model)
        self.layer_norm_ff = nn.LayerNorm(d_model)

    def forward(self, X: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Decoder layer. The parameter `memory` referes to the
        output of the last Encoder layer in the Encoder stack."""
        # Sub-layer one (masked self-attention)
        Z = self.self_attention(X)
        Z = self.layer_norm_self(X + Z) # residual after X
        # Sub-layer two (encoder-decoder attention)
        Y = self.encdec_attention(X, memory=memory)
        Y = self.layer_norm_encdec(Z + Y)
        # Sub-layer three (feed-forward network)
        R = self.ff(Y)
        R = self.layer_norm_ff(Y + R) # residual after Z
        return R


class Transformer(nn.Module):
    def __init__(
        self, 
        h: int = 8,
        N: int = 6,
        d_model: int = 512,
        d_ff: int = 2048,
        d_v: int | None = None,
    ) -> None:
        """NOT IMPLEMENTED. What's there is just blueprint to keep grid in head,
        albeit wrong atm."""
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(h=h, d_model=d_model, d_ff=d_ff, d_v=d_v)
                for _ in range(N)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(h=h, d_model=d_model, d_ff=d_ff, d_v=d_v)
                for _ in range(N)
            ]
        )

        def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
            # embed and positionally encode X and Y
            # missing ...
            # Pass through encoder stack
            for enc in self.encoder_layers:
                X = enc(X)
            memory = X.clone().detach() # TODO: is this unnecessary?
            # Pass through decoder stack
            for dec in self.decoder_layers:
                Y = dec(Y, memory)

            return Y # PLACEHOLDER

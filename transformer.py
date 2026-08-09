import math

import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_idx: int | None = None) -> None:
        super().__init__()
        self.emb_layer = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.d_model = d_model
        self.padding_idx = pad_idx

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # TODO: consider different inputs than torch tensors
        return self.emb_layer(X) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    pe: torch.Tensor
    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # Calculate the PE matrix
        pe = torch.empty((self.max_seq_len, self.d_model), requires_grad=False)
        pos = torch.arange(0, self.max_seq_len).unsqueeze(1) # shape (max_seq_len, 1)
        two_i = torch.arange(0, self.d_model, 2) # shape (d_model/2,)
        progression_range = 10000
        # exp of log 10000^(-2i / d_model) avoids overflow; div_term.shape is (d_model/2,)
        div_term = torch.exp(-(two_i / d_model) * math.log(progression_range))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term[: self.d_model // 2])
        # the integer div by 2 is a bespoke floor for correct slice in uneven d_model case

        self.register_buffer("pe", pe) # goes to GPU without being model param/trained on

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        seq_len = X.size(-2) # either (batch, N, dim) or (N, dim)
        assert seq_len <= self.max_seq_len, (
            f"Sequence too long: {seq_len} > {self.max_seq_len} = max sequence length"
        )

        return X + self.pe[:seq_len]


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
        self, h: int, d_model: int, d_k: int, d_v: int | None, masked: bool = False,
    ) -> None:
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v if d_v is not None else d_k
        self.h = h # number of heads in multi-head attention
        self.mask_inputs = masked
        self.W_O = nn.Linear((self.d_v * h), d_model)
        self.W_Q = nn.Linear(d_model, d_k * h)
        self.W_K = nn.Linear(d_model, d_k * h)
        self.W_V = nn.Linear(d_model, (self.d_v * h))

    def _mask_future_inputs(self, sm_input: torch.Tensor) -> torch.Tensor:
        assert sm_input.shape[-2] == sm_input.shape[-1], (
            "Expected square score matrix, got "
            f"({sm_input.shape[-2]}, {sm_input.shape[-1]})."
        )
        seq_len = sm_input.shape[-1]
        mask = torch.full((seq_len, seq_len), False, device=sm_input.device)
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    mask[i, j] = True
                    # TODO: rewrite with torch.triu

        return sm_input.masked_fill(mask, -torch.inf)

    def scaled_dot_product_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Scaled dot product attention."""
        assert Q.shape[-1] == K.shape[-1] == self.d_k,(
            f"Expected Q and K to have last dim {self.d_k} i.e. d_k"
        )
        assert V.shape[-1] == self.d_v, f"Expected V to have last dim {self.d_v} i.e. d_v"
        softmax_input = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if self.mask_inputs:
            softmax_input = self._mask_future_inputs(softmax_input)
        scaled_dot_prods = nn.functional.softmax(softmax_input, dim=-1)

        return torch.matmul(scaled_dot_prods, V)

    def forward(self, X: torch.Tensor, memory: torch.Tensor | None = None) -> torch.Tensor:
        # Calculate h head_i = attention(QW^Q_i,...)
        key_val_source = X if memory is None else memory
        seq_q = X.shape[0]
        seq_kv = key_val_source.shape[0]
        Q = self.W_Q(X).reshape(seq_q, self.h, self.d_k).transpose(0, 1) # (h, n, d_k)
        K = self.W_K(key_val_source).reshape(seq_kv, self.h, self.d_k).transpose(0, 1)
        V = self.W_V(key_val_source).reshape(seq_kv, self.h, self.d_v).transpose(0, 1)
        heads = self.scaled_dot_product_attention(Q, K, V) # heads (h, seq_q, d_v)
        cat = heads.transpose(0, 1).reshape(seq_q, self.d_v * self.h)
        return self.W_O(cat)


class EncoderLayer(nn.Module):
    def __init__(
        self, 
        h: int,
        d_model: int,
        d_ff: int,
        d_k: int,
        d_v: int | None,
        p_drop: float | None = 0.1,
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, h=h, d_k=d_k, d_v=d_v)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.layer_norm_self = nn.LayerNorm(d_model)
        self.layer_norm_ff = nn.LayerNorm(d_model)
        self.p_drop = p_drop

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Sub-layer one (self-attention)
        Z = self.self_attention(X)
       
        Z = self.layer_norm_self(X + Z) # residual after X
        # Sub-layer two (feed-forward network)
        R = self.ff(Z)
       
        R = self.layer_norm_ff(Z + R) # residual after Z
        return R


class DecoderLayer(nn.Module):
    def __init__(
        self,
        h: int,
        d_model: int,
        d_ff: int,
        d_k: int,
        d_v: int | None,
        p_drop: float | None = 0.1,
    ) -> None:
        super().__init__()
        self.encdec_attention = MultiHeadAttention(d_model=d_model, h=h, d_k=d_k, d_v=d_v)
        self.self_attention = MultiHeadAttention(
            d_model=d_model, h=h, d_k=d_k, d_v=d_v, masked=True,
        )
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.layer_norm_encdec = nn.LayerNorm(d_model)
        self.layer_norm_self = nn.LayerNorm(d_model)
        self.layer_norm_ff = nn.LayerNorm(d_model)
        self.p_drop = p_drop

    def forward(self, X: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Decoder layer. The parameter `memory` referes to the
        output of the last Encoder layer in the Encoder stack."""
        # Sub-layer one (masked self-attention)
        Z = self.self_attention(X)
       
        Z = self.layer_norm_self(X + Z) # residual after X
        # Sub-layer two (encoder-decoder attention)
        Y = self.encdec_attention(Z, memory=memory)
       
        Y = self.layer_norm_encdec(Z + Y)
        # Sub-layer three (feed-forward network)
        R = self.ff(Y)
       
        R = self.layer_norm_ff(Y + R) # residual after Z
        return R


class Transformer(nn.Module):
    def __init__(
        self, 
        max_seq_len: int,
        vocab_size: int,
        h: int = 8,
        N: int = 6,
        d_ff: int = 2048,
        d_model: int = 512,
        d_k: int = 64,
        d_v: int | None = None,
        p_drop: float | None = 0.1,
    ) -> None:
        """NOT IMPLEMENTED. What's there is just blueprint to keep grid in head,
        albeit wrong atm."""
        super().__init__()
        self.positional_encoding = PositionalEncoding(max_seq_len, d_model)
        self.embedding = Embedding(vocab_size, d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie the weights for embedding and output head as they do in paper
        self.output_head.weight = self.embedding.emb_layer.weight
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(h=h, d_model=d_model, d_ff=d_ff, d_k=d_k, d_v=d_v, p_drop=p_drop)
                for _ in range(N)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(h=h, d_model=d_model, d_ff=d_ff, d_k=d_k, d_v=d_v, p_drop=p_drop)
                for _ in range(N)
            ]
        )
        self.p_drop = p_drop

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # Embed and positionally encode X
        X = self.embedding(X)
        X = self.positional_encoding(X)
        # Same for Y
        Y = self.embedding(Y)
        Y = self.positional_encoding(Y)
        # droupout the sum of embedding and pos enc
       
       
        # Pass through encoder stack
        for enc in self.encoder_layers:
            X = enc(X)
        # Pass through decoder stack
        for dec in self.decoder_layers:
            Y = dec(Y, X)
        # Tied weights from embedding used for output head (transposed)
        Y = self.output_head(Y)

        return Y # Predicted next-token distribution raw logits

if __name__ == "__main__":
    m = Transformer(128, 1000)
    print(m.embedding.emb_layer.weight is m.output_head.weight)
    print(sum(p.numel() for p in m.parameters()))
    print("test forward")
    X = torch.randint(0, 1000, (128,)) # rand ids in vocab size, 128 of these
    Y = torch.randint(0, 1000, (128,))

    print(f"Result: {m(X, Y)}")

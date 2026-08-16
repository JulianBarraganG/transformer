import math

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_idx: int | None = None) -> None:
        super().__init__()
        self.emb_layer = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.d_model = d_model
        self.padding_idx = pad_idx

    def forward(self, X: Tensor) -> Tensor:
        # TODO: consider different inputs than torch tensors
        return self.emb_layer(X) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    pe: Tensor
    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        # Calculate the PE matrix
        pe = torch.empty((max_seq_len, d_model))
        pos = torch.arange(0, max_seq_len).unsqueeze(1) # shape (max_seq_len, 1)
        two_i = torch.arange(0, d_model, 2) # shape (d_model/2,)
        progression_range = 10000
        # exp of log 10000^(-2i / d_model) avoids overflow; div_term.shape is (d_model/2,)
        div_term = torch.exp(-(two_i / d_model) * math.log(progression_range))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term[: d_model // 2])
        # the integer div by 2 is a bespoke floor for correct slice in uneven d_model case

        self.register_buffer("pe", pe) # goes to GPU without being model param/trained on

    def forward(self, X: Tensor) -> Tensor:
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
        self.layer_2 = nn.Linear(self.d_ff, self.d_model)

    def forward(self, X: Tensor):
        X = self.layer_1(X)
        X = F.relu(X)
        X = self.layer_2(X)
        return X


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        h: int,
        d_model: int,
        d_k: int,
        d_v: int,
    ) -> None:
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.h = h # number of heads in multi-head attention
        self.W_O = nn.Linear((self.d_v * h), d_model)
        self.W_Q = nn.Linear(d_model, d_k * h)
        self.W_K = nn.Linear(d_model, d_k * h)
        self.W_V = nn.Linear(d_model, (self.d_v * h))

    def scaled_dot_product_attention(
        self, Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        """Scaled dot product attention. `mask` is True where attention is forbidden."""
        assert Q.shape[-1] == K.shape[-1] == self.d_k,(
            f"Expected Q and K to have last dim {self.d_k} i.e. d_k"
        )
        assert V.shape[-1] == self.d_v, f"Expected V to have last dim {self.d_v} i.e. d_v"

        softmax_input = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            softmax_input.masked_fill_(mask, torch.finfo(softmax_input.dtype).min)
        attention = F.softmax(softmax_input, dim=-1)

        return torch.matmul(attention, V)

    def forward(
        self,
        X: Tensor,
        memory: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        # Calculate h head_i = attention(QW^Q_i,...)
        key_val_source = X if memory is None else memory
        # reshape to (*, h, n, d_k)
        Q = self.W_Q(X).unflatten(-1, (self.h, self.d_k)).transpose(-3, -2) 
        K = self.W_K(key_val_source).unflatten(-1, (self.h, self.d_k)).transpose(-3, -2)
        V = self.W_V(key_val_source).unflatten(-1, (self.h, self.d_v)).transpose(-3, -2)
        heads = self.scaled_dot_product_attention(Q, K, V, mask) # heads shape (*, h, n, d_v)
        cat = heads.transpose(-3, -2).flatten(-2) # merge last 2 axes; shape (*, n, h*d_v)

        return self.W_O(cat)


class EncoderLayer(nn.Module):
    def __init__(
        self, 
        h: int,
        d_model: int,
        d_ff: int,
        d_k: int,
        d_v: int,
        p_drop: float | None = 0.1,
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(h=h, d_model=d_model, d_k=d_k, d_v=d_v)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.layer_norm_self = nn.LayerNorm(d_model)
        self.layer_norm_ff = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop) if p_drop is not None else None

    def forward(self, X: Tensor, pad_mask: Tensor) -> Tensor:
        # Sub-layer one (self-attention)
        Z = self.self_attention(X, mask=pad_mask)
        Z = self.dropout(Z) if self.dropout is not None else Z
        Z = self.layer_norm_self(X + Z) # residual after X
        # Sub-layer two (feed-forward network)
        R = self.ff(Z)
        R = self.dropout(R) if self.dropout is not None else R
        R = self.layer_norm_ff(Z + R) # residual after Z
        return R


class DecoderLayer(nn.Module):
    def __init__(
        self,
        h: int,
        d_model: int,
        d_ff: int,
        d_k: int,
        d_v: int,
        p_drop: float | None = 0.1,
    ) -> None:
        super().__init__()
        self.encdec_attention = MultiHeadAttention(h=h, d_model=d_model, d_k=d_k, d_v=d_v)
        self.self_attention = MultiHeadAttention(h=h, d_model=d_model, d_k=d_k, d_v=d_v)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.layer_norm_encdec = nn.LayerNorm(d_model)
        self.layer_norm_self = nn.LayerNorm(d_model)
        self.layer_norm_ff = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop) if p_drop is not None else None

    def forward(
        self, X: Tensor, memory: Tensor, self_mask: Tensor, pad_mask: Tensor
    ) -> Tensor:
        """Forward pass for the Decoder layer. The parameter `memory` referes to the
        output of the last Encoder layer in the Encoder stack."""
        # Sub-layer one (masked self-attention) -- causality lives here
        Z = self.self_attention(X, mask=self_mask)
        Z = self.dropout(Z) if self.dropout is not None else Z
        Z = self.layer_norm_self(X + Z) # residual after X
        # Sub-layer two (encoder-decoder attention) -- unmasked, decoder sees all of source
        Y = self.encdec_attention(Z, memory=memory, mask=pad_mask)
        Y = self.dropout(Y) if self.dropout is not None else Y
        Y = self.layer_norm_encdec(Z + Y)
        # Sub-layer three (feed-forward network)
        R = self.ff(Y)
        R = self.dropout(R) if self.dropout is not None else R
        R = self.layer_norm_ff(Y + R) # residual after Z

        return R


class Encoder(nn.Module):
    """Stack of N identical encoder layers."""
    def __init__(
        self,
        N: int,
        h: int,
        d_model: int,
        d_ff: int,
        d_k: int,
        d_v: int | None,
        p_drop: float | None = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(h=h, d_model=d_model, d_ff=d_ff, d_k=d_k, d_v=d_v, p_drop=p_drop)
                for _ in range(N)
            ]
        )

    def forward(self, X: Tensor, pad_mask: Tensor) -> Tensor:
        for layer in self.layers:
            X = layer(X, pad_mask)
        return X


class Decoder(nn.Module):
    """Stack of N identical decoder layers. `memory` is the final encoder output."""
    def __init__(
        self,
        N: int,
        h: int,
        d_model: int,
        d_ff: int,
        d_k: int,
        d_v: int | None,
        p_drop: float | None = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(h=h, d_model=d_model, d_ff=d_ff, d_k=d_k, d_v=d_v, p_drop=p_drop)
                for _ in range(N)
            ]
        )

    def forward(
        self, X: Tensor, memory: Tensor, self_mask: Tensor, pad_mask: Tensor
    ) -> Tensor:
        for layer in self.layers:
            X = layer(X, memory, self_mask, pad_mask)
        return X


class Transformer(nn.Module):
    causal_mask: Tensor
    def __init__(
        self, 
        max_seq_len: int,
        vocab_size: int,
        padding_idx: int,
        h: int = 8,
        N: int = 6,
        d_ff: int = 2048,
        d_model: int = 512,
        d_k: int = 64,
        d_v: int | None = None,
        p_drop: float | None = 0.1,
    ) -> None:
        """Transformer Neural Network"""
        super().__init__()
        d_v = d_v if d_v is not None else d_k
        self.dropout = nn.Dropout(p_drop) if p_drop is not None else None
        self.positional_encoding = PositionalEncoding(max_seq_len, d_model)
        self.embedding = Embedding(vocab_size, d_model, pad_idx=padding_idx)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie the weights for embedding and output head as they do in paper
        self.output_head.weight = self.embedding.emb_layer.weight
        self.encoder = Encoder(
            N=N, h=h, d_model=d_model, d_ff=d_ff, d_k=d_k, d_v=d_v, p_drop=p_drop
        )
        self.decoder = Decoder(
            N=N, h=h, d_model=d_model, d_ff=d_ff, d_k=d_k, d_v=d_v, p_drop=p_drop
        )
        self.padding_idx = padding_idx
        self.p_drop = p_drop
        self.d_model = d_model
        self.register_buffer(
            "causal_mask",
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool).triu(1),
            persistent=False,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Init shared weight tensor post others
        nn.init.normal_(self.embedding.emb_layer.weight, mean=0.0, std=self.d_model**-0.5)
        # Finally set padding tokens weight zero i.e. no learning
        with torch.no_grad():
            self.embedding.emb_layer.weight[self.padding_idx].zero_()


    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        assert src.dim() == 2, (
            f"Encoder input must be rank 2 (batch, seq_len), not rank {src.dim()} input"
        )
        assert tgt.dim() == 2, (
            f"Decoder input must be rank 2 (batch, seq_len), got rank {tgt.dim()} input"
        )
        # Calculate the combined causal and pad mask
        tgt_len = tgt.shape[-1]
        # Padding mask expect (B,1,1,S) for broadcasting over query/key-val len and heads
        src_pad_mask = (src == self.padding_idx)[:, None, None, :] # (B,S) -> (B,1,1,S)
        tgt_pad_mask = (tgt == self.padding_idx)[:, None, None, :] # (B,T) -> (B,1,1,T)
        causal_mask = self.causal_mask[:tgt_len, :tgt_len]
        self_attn_mask = causal_mask | tgt_pad_mask # cheap guard for left padding
        # Embed and positionally encode source tensor
        src = self.embedding(src)
        src = self.positional_encoding(src)
        # Same for target tensor
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        # Droupout the sum of embedding and pos enc
        if self.dropout is not None:
            src = self.dropout(src)
            tgt = self.dropout(tgt)
        # Pass through encoder stack; `memory` is what cross-attention keys/values come from
        memory = self.encoder(src, src_pad_mask)
        # Pass through decoder stack; cross-attention masks the SOURCE pads, not the target's
        tgt = self.decoder(tgt, memory, self_attn_mask, src_pad_mask)
        # Tied weights from embedding used for output head (transposed)
        tgt = self.output_head(tgt)

        return tgt # Predicted next-token distribution raw logits

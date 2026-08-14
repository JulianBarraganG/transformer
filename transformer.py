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
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # Calculate the PE matrix
        pe = torch.empty((self.max_seq_len, self.d_model))
        pos = torch.arange(0, self.max_seq_len).unsqueeze(1) # shape (max_seq_len, 1)
        two_i = torch.arange(0, self.d_model, 2) # shape (d_model/2,)
        progression_range = 10000
        # exp of log 10000^(-2i / d_model) avoids overflow; div_term.shape is (d_model/2,)
        div_term = torch.exp(-(two_i / d_model) * math.log(progression_range))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term[: self.d_model // 2])
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
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(self.d_ff, self.d_model)

    def forward(self, X: Tensor):
        X = self.layer_1(X)
        X = self.relu(X)
        X = self.layer_2(X)
        return X


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        h: int,
        d_model: int,
        d_k: int,
        d_v: int | None,
    ) -> None:
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v if d_v is not None else d_k
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
            softmax_input.masked_fill_(mask, -torch.finfo(softmax_input.dtype).min)
        attention = F.softmax(softmax_input, dim=-1)

        return torch.matmul(attention, V)

    def forward(
        self,
        X: Tensor,
        memory: Tensor | None = None,
        mask: Tensor | None = None
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
        d_v: int | None,
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
        Z = self.self_attention(X, pad_mask)
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
        d_v: int | None,
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

    def forward(self, X: Tensor, memory: Tensor, mask: Tensor | None) -> Tensor:
        """Forward pass for the Decoder layer. The parameter `memory` referes to the
        output of the last Encoder layer in the Encoder stack."""
        # Sub-layer one (masked self-attention) -- causality lives here
        Z = self.self_attention(X, mask=mask)
        Z = self.dropout(Z) if self.dropout is not None else Z
        Z = self.layer_norm_self(X + Z) # residual after X
        # Sub-layer two (encoder-decoder attention) -- unmasked, decoder sees all of source
        Y = self.encdec_attention(Z, memory=memory)
        Y = self.dropout(Y) if self.dropout is not None else Y
        Y = self.layer_norm_encdec(Z + Y)
        # Sub-layer three (feed-forward network)
        R = self.ff(Y)
        R = self.dropout(R) if self.dropout is not None else R
        R = self.layer_norm_ff(Y + R) # residual after Z

        return R


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
        self.dropout = nn.Dropout(p_drop) if p_drop is not None else None
        self.positional_encoding = PositionalEncoding(max_seq_len, d_model)
        self.embedding = Embedding(vocab_size, d_model, pad_idx=padding_idx)
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
        # init shared weight tensor post others
        nn.init.normal_(self.embedding.emb_layer.weight, mean=0.0, std=self.d_model**-0.5)


    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        # Calculate the combined causal and pad mask
        tgt_len = Y.shape[-2]
        # TODO: Mask tokens between seq len and batch_max_seq_len
        pad_mask = None # NOT IMPLEMENTED
        causal_mask = self.causal_mask[:tgt_len, :tgt_len]
        # Embed and positionally encode X
        X = self.embedding(X)
        X = self.positional_encoding(X)
        # Same for Y
        Y = self.embedding(Y)
        Y = self.positional_encoding(Y)
        # droupout the sum of embedding and pos enc
        if self.dropout is not None:
            X = self.dropout(X)
            Y = self.dropout(Y)
        # Pass through encoder stack
        for enc in self.encoder_layers:
            X = enc(X, pad_mask)
        # Pass through decoder stack; slice the cached mask to this batch's target length
        for dec in self.decoder_layers:
            Y = dec(Y, X, causal_mask)
        # Tied weights from embedding used for output head (transposed)
        Y = self.output_head(Y)

        return Y # Predicted next-token distribution raw logits

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = Transformer(128, 1000, 10)
    print(m.embedding.emb_layer.weight is m.output_head.weight)
    print(sum(p.numel() for p in m.parameters()))
    print("test forward")
    batch_size, seq_len, vocab_size = 32, 128, 1000
    # rand token ids: (batch, seq) -> embeds to (batch, seq, d_model)
    X = torch.randint(0, vocab_size, (batch_size, seq_len))
    Y = torch.randint(0, vocab_size, (batch_size, seq_len))

    out = m(X, Y)
    print(f"Result shape: {tuple(out.shape)}") # expect (32, 128, 1000)
    m.eval()  # dropout off, else the number is noisy
    with torch.no_grad():
        logits = m(X, Y)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    print(f"initial loss {loss.item():.3f} (want ~{math.log(vocab_size):.3f})")

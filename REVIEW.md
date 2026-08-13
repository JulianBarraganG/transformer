# Transformer Review — pre-training pass

Review of `transformer.py` before writing the training loop.
Checkbox state is edited in place by `checklist.py` (or by hand — it's just markdown).

Suggested order: **A1/B1 → C5 → C2 → C1+C3 → training loop with C4.**
A1/B1 first because nothing downstream is measurable until initial loss reads ~6.9.

## A. Immediate — bugs and wrong-shape class

- [ ] **A1. Embedding init will break training** (`transformer.py:11`, `:220`)
      `nn.Embedding` defaults to `N(0, 1)` per component, then `forward` multiplies by
      `sqrt(512) ~= 22.6`. Tied to `output_head`, so at step 0 the decoder's post-LayerNorm
      output (components ~std 1) dots against weight rows of std 1 over 512 dims, giving
      logits with std ~= 22.6. Softmax over that is numerically one-hot. Initial loss lands
      in the hundreds instead of `ln(1000) ~= 6.9`, and gradients for every non-argmax class
      are ~0. Will present as a learning-rate problem. Fix is B1.

- [x] **A2. `padding_idx` is dead** (`transformer.py:9-11`, `:217`)
      `Embedding.__init__` accepts `pad_idx`; `Transformer.__init__` never passes it.
      Non-obvious interaction once wired: `nn.Embedding`'s `padding_idx` zeroes that row and
      blocks its gradient *through the embedding path only*. The weight is tied, so
      `output_head` still accumulates gradient into the pad row via the logit path. The pad
      row will not stay zero.

- [x] **A3. Wrong shape comment** (`transformer.py:122`)
      `flatten(-2)` yields `(..., seq_q, h*d_v)`, not `(..., h*d_v)`. Cosmetic, but this is
      exactly the comment that gets trusted at 2am.

- [x] **A4. Stale docstring** (`transformer.py:212-213`)
      "NOT IMPLEMENTED ... albeit wrong atm". It is implemented. Delete.

- [x] **A5. Dead attribute `self.p_drop`** (`transformer.py:233`)
      Nothing reads it; the `nn.Dropout` modules own that state now.

- [x] **A6. Misleading `requires_grad=False`** (`transformer.py:26`)
      Already the default, and `register_buffer` is what actually determines the semantics.
      Reads like it is doing something.

## B. Init — the syntactic sugar not being leveraged

- [x] **B1. Add `_init_weights`, call at end of `__init__` AFTER the tying line**
      `nn.Module.modules()` / `.apply(fn)` walk the whole submodule tree — that is the idiom.

      ```python
      def _init_weights(self) -> None:
          for module in self.modules():
              if isinstance(module, nn.Linear):
                  nn.init.xavier_uniform_(module.weight)
                  if module.bias is not None:
                      nn.init.zeros_(module.bias)
          # after the Linear sweep, so it wins on the tied weight
          nn.init.normal_(self.embedding.emb_layer.weight, mean=0.0, std=self.d_model**-0.5)
      ```

      Three things to understand rather than copy:

      1. **Order is load-bearing, because of tying.** `output_head.weight` *is*
         `emb_layer.weight` — one tensor, two names. A generic `isinstance(m, nn.Linear)`
         sweep hits it as `output_head` and xavier-inits it. Xavier on `(1000, 512)` gives
         std ~= 0.032, then x`sqrt(512)` -> std 0.72 into LayerNorm. Not catastrophic, but
         not what §3.4 specifies. Doing the embedding init last makes intent unambiguous.
      2. **`d_model**-0.5` is a std, not a variance.** std ~= 0.0442, i.e. `N(0, 1/512)`.
         After x`sqrt(512)` the embeddings land at std ~= 1 — matched to what LayerNorm and
         the residual stream expect. That is the entire point of the `sqrt(d_model)` scale.
      3. **Why xavier over PyTorch's default.** `nn.Linear` defaults to
         `kaiming_uniform_(a=sqrt(5))`, which works out to `U(-1/sqrt(fan_in), 1/sqrt(fan_in))`,
         std ~= `1/sqrt(3*fan_in)`. Legacy default tuned for a different era. Xavier/Glorot
         uses `fan_in + fan_out`, which is what the reference implementations (tensor2tensor,
         Annotated Transformer) use. For `W_Q` at 512->512 that is std 0.026 vs 0.044; it
         matters more at the 512->2048 FFN boundary where fan_in and fan_out diverge.

- [x] **B2. Sanity check: untrained forward should give loss ~= 6.9**
      `F.cross_entropy(logits.flatten(0, 1), targets.flatten())` on an untrained model with
      `vocab_size=1000` should print ~= `ln(1000) = 6.9`. If it does not, init is wrong —
      and that needs to be known *before* a training loop exists to blame.

## C. Architectural / what actually blocks training

- [ ] **C1. No padding mask — this is the blocker, not a TODO** (`transformer.py:71`, `:77`)
      Real batches have variable-length sequences. Without a pad mask every pad token
      participates in attention as a real key, and the encoder output is contaminated.
      Ranked #1 because everything else in the training loop is downstream of the interface
      change it forces: `masked: bool` in `__init__` has to become `mask: Tensor | None` in
      `forward`. A constructor-time flag cannot express a per-batch mask. Do this refactor
      *before* the training loop — it touches every call site.

- [x] **C2. Causal mask rebuilt per call with a Python double loop** (`transformer.py:88-96`)
      `128^2 = 16384` Python iterations x 6 decoder layers x every step, both forward passes.
      Not a micro-optimization — it will dominate step time by orders of magnitude.
      `torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)` is one call, and it belongs
      in a `register_buffer` at `max_seq_len` sliced to `[:n, :n]`, not recomputed. Same
      reasoning as the `pe` buffer — the instinct was already there once.

- [ ] **C3. `-torch.inf` will bite in mixed precision** (`transformer.py:96`)
      Fine today: a causal row always has >=1 unmasked entry. Once padding masks land, a
      fully-masked row gives `softmax([-inf, ...]) = NaN`. Standard fix is
      `torch.finfo(scores.dtype).min`. Do it when C1 lands.

- [ ] **C4. Post-LN requires warmup — not optional** (`transformer.py:150`, `:187` etc.)
      The paper is implemented faithfully (LayerNorm *after* the residual add), which is
      correct. Consequence: gradients through the residual path get rescaled at every layer,
      and a 6-layer post-LN stack diverges without the §5.3 warmup schedule. Budget for
      `lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)` with `warmup=4000` as part of
      getting it to train at all, not as tuning. (Pre-LN — norm *inside* the residual branch —
      trains without warmup and is what modern models use. Not a bug; just know the choice
      has a string attached.)

- [ ] **C5. `forward(self, X, Y)` naming will cause a real bug** (`transformer.py:235`)
      `Y` here is the decoder *input* — targets shifted right. The training loop will also
      have a `Y` meaning labels. Two different tensors, offset by one position, same name.
      Failure mode is a model that trivially learns to copy its input and shows suspiciously
      low loss. Rename to `src` / `tgt` now. Cheap.

- [ ] **C6. Missing input contract** (`transformer.py:235`)
      `assert src.dim() == 2` in `Transformer.forward`. Strict at the boundary, polymorphic
      in the core — the MHA internals stay tail-relative because rank changes mid-computation
      anyway, but the public API should not shape-sniff.

- [ ] **C7. Resolve `d_v` once** (`transformer.py:209`, `:75`)
      `Transformer` defaults `d_v=None` and threads it down; `MultiHeadAttention` resolves
      `None -> d_k` six levels deep, per layer, 12 times. Resolve it once in
      `Transformer.__init__` so `d_v` is a concrete int everywhere below. Makes `state_dict`
      shapes predictable and removes the need to remember which layer resolved what.

## D. Polish

- [ ] **D1. `scaled_dot_product_attention` shadows `F.scaled_dot_product_attention`**
      Assumed deliberate. Once the hand-rolled version is verified correct, swapping in the
      fused kernel is a one-line A/B that gives FlashAttention. Keep the name.

- [ ] **D2. `nn.ReLU()` module vs `F.relu`** (`transformer.py:54`)
      Both idiomatic. Module form is right if the activation should ever be swappable via
      config; `F.relu` is more common for a stateless op. Non-issue.

- [ ] **D3. No `Encoder` / `Decoder` wrapper classes around the `ModuleList`s**
      Fine at this size. Worth adding if an encoder-only variant is ever wanted.

- [ ] **D4. `PositionalEncoding` stores unused `self.d_model`** (`transformer.py:23`)
      Only `max_seq_len` is read. Harmless.

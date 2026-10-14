# Tracking

Decisions for training observability. Local only, no external tracking tools.

## Run directory

```
runs/<YYYY-MM-DD_HHMM>_<run-name>/
  config.json
  metrics.jsonl
  checkpoints/
    epoch_<NNN>.pt
    last.pt
    best.pt        # lowest val loss
```

`runs/` is gitignored. Timestamp is Europe/Paris local time.

## config.json

Written once by `RunLogger`, before training starts.

```
{"model": {...}, "hyper": {...}}
```

- `model`: `Transformer` kwargs, incl. `vocab_size`/`padding_idx` from the vocab
- `train`: warmup, batch size, epochs, lr, betas, eps

Later: seed, git commit hash + dirty flag, torch version, device, dataset.

## metrics.jsonl

- One JSON object per line, file opened in append mode per write
- Every record keyed by `step` (global step = optimizer updates, never reset per epoch)
- `epoch` is 0-based; +1 only in console output
- Every record has a `kind` field, a defaulted field on the record dataclass

### kind: `train_step` (every step)

| field         | note                                                     |
|---------------|----------------------------------------------------------|
| `step`        | global step                                              |
| `epoch`       |                                                          |
| `batch_loss`  | batch loss                                               |
| `n_tokens`    | non-PAD target tokens, for token-weighted epoch means    |
| `grad_norm`   | total L2 norm, after `backward()`, before `step()`       |
| `lr`          | `scheduler.get_last_lr()`, read before `scheduler.step()`|

Epoch-level summaries are derived from step records at analysis time.

### kind: `val` (end of every epoch)

| field         | note                                                     |
|---------------|----------------------------------------------------------|
| `step`        | global step after the epoch                              |
| `epoch`       |                                                          |
| `val_loss`    | token-weighted mean over the whole val split             |
| `n_tokens`    | non-PAD target tokens in the val split                   |

## Checkpoints

 Saved at each epoch boundary. A dict containing:

- `model`, `optimizer`, `scheduler` state dicts
- `epoch`, `step`
- RNG states (torch, cuda, python `random`)
- `config`

`epoch_<NNN>.pt` kept for every epoch (0-based, matches `epoch`). `last.pt` overwritten each epoch. `best.pt` on lowest val loss.

Never `torch.save(model)`. Rebuild from config, then load state dict.

## Reproducibility

Not implemented yet.

- Seed everything
- Explicit `torch.Generator` passed to the `DataLoader`
- Full GPU determinism off by default

## Recording architecture

- Record dataclasses and `RunLogger` live in `tracking.py`
- `RunLogger` is passed explicitly into the training functions; it owns the run directory and writes `config.json` and `metrics.jsonl`
- Training loop produces `TrainStepRecord`s; `_validate_transformer` returns numbers, `train_transformer` logs the `ValRecord`
- Stdlib `logging` is for human-readable text only
- Checkpointing is separate from metric logging

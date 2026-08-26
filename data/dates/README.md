# Date normalization (synthetic)

Sequence-to-sequence: a date in an arbitrary human surface form → ISO 8601.

```
"Wednesday, March 3, 2021"  →  "2021-03-03"
"03.03.2021"                →  "2021-03-03"
"3 Mar 2021"                →  "2021-03-03"
```

Character-level, so no tokenizer is needed. Regenerate with `python data/make_dates.py`
(deterministic given `--seed`, default 0).

## Files

| file | rows | contents |
|---|---|---|
| `train.jsonl` | 50,000 | 11 surface formats |
| `valid.jsonl` | 5,000 | same formats, unseen dates |
| `test.jsonl` | 5,000 | same formats, unseen dates |
| `test_unseen_format.jsonl` | 5,000 | **2 held-out formats**, unseen dates |
| `vocab.json` | — | char → id, specials at 0–3 |
| `meta.json` | — | vocab size, max lengths, split counts |

One JSON object per line:

```json
{"src": "Thursday, July 11, 2024", "tgt": "2024-07-11", "format": "weekday_full"}
```

`format` is metadata for error analysis — which surface forms the model gets wrong.
Not an input.

## Splits

**Disjoint by calendar date, not by row.** A date in `train` appears in no other split,
in any surface form. Random row splitting would leak: the model could memorize
`3 March 2021 → 2021-03-03` and then score well on `March 3, 2021` without having
generalized at all.

**`test_unseen_format.jsonl`** uses `spelled_out` (`"14th of December, 1990"`) and
`dashed_abbr` (`"22-Aug-1983"`), both absent from training. Regular `test` measures
*new dates, familiar formats*; this one measures *new dates, unfamiliar formats*.
Expect a large gap. The second number is the interesting one.

## Conventions

Numeric formats fix a convention so the task is well-posed: **slashes are US**
(`MM/DD/YYYY`), **dots are European** (`DD.MM.YYYY`). Without that rule `03/04/2021`
has two correct answers and the model can only ever learn the prior.

Years are 1950–2049, always 4 digits. Two-digit years would reintroduce ambiguity.

`iso` is included as an input format, so some examples are identity mappings.

## Numbers you need

```
vocab_size            50   (46 chars + <pad> <bos> <eos> <unk>)
max src chars         29   ("Wednesday, September 30, 2021")
max tgt chars         10
suggested max_seq_len 31   (max chars + <bos> + <eos>)
```

Special token ids: `<pad>=0`, `<bos>=1`, `<eos>=2`, `<unk>=3`. Pad at 0 by convention;
pass it as `padding_idx` to `Transformer`.

## Difficulty knobs

Task is easy by design — it should train in minutes. To make it harder: widen
`--start-year`/`--end-year`, add noisy surface forms (typos, `"Sept"` vs `"Sep"`,
extra whitespace), or add two-digit years with a pivot rule.

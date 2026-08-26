"""Generate a synthetic date-normalization dataset.

Task: map a date written in an arbitrary human surface form to ISO 8601.

    "Wednesday, March 3, 2021"  ->  "2021-03-03"
    "03.03.2021"                ->  "2021-03-03"
    "3 Mar 2021"                ->  "2021-03-03"

Character-level, so no tokenizer is needed -- the vocabulary is just the set of
characters that appear, plus four special tokens.

Output layout (JSONL, one example per line, mirroring how HuggingFace ships
translation corpora):

    dates/train.jsonl
    dates/valid.jsonl
    dates/test.jsonl
    dates/test_unseen_format.jsonl
    dates/vocab.json
    dates/meta.json

Two properties worth knowing about, both deliberate:

1. Splits are disjoint by *calendar date*, not by row. A date that appears in
   train never appears in valid or test, in any surface form. Random row
   splitting would leak -- the model could memorize "3 March 2021 -> 2021-03-03"
   and score well on "March 3, 2021" without generalizing.

2. `test_unseen_format.jsonl` uses surface formats held out of training
   entirely. Normal test measures "new dates, familiar formats"; this one
   measures "new dates, unfamiliar formats", which is the harder and more
   interesting number.

Numeric formats fix a convention so the task stays well-posed: slashes are
US (MM/DD/YYYY), dots are European (DD.MM.YYYY). Without that rule "03/04/2021"
has two correct answers and the model can only ever learn the prior.
"""

import argparse
import calendar
import json
import random
from datetime import date, timedelta
from pathlib import Path

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]  # ids 0..3; pad at 0 by convention


def ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 11 -> '11th', 23 -> '23rd'."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# Each entry renders a date into one human surface form. Keep these pure and
# deterministic -- all the randomness lives in which format gets picked.
FORMATS = {
    "month_day_year": lambda d: f"{calendar.month_name[d.month]} {d.day}, {d.year}",
    "month_day_year_lower": lambda d: f"{calendar.month_name[d.month].lower()} {d.day}, {d.year}",
    "day_month_year": lambda d: f"{d.day} {calendar.month_name[d.month]} {d.year}",
    "abbr_day_year": lambda d: f"{calendar.month_abbr[d.month]} {d.day}, {d.year}",
    "day_abbr_year": lambda d: f"{d.day} {calendar.month_abbr[d.month]} {d.year}",
    "weekday_full": lambda d: (
        f"{calendar.day_name[d.weekday()]}, {calendar.month_name[d.month]} {d.day}, {d.year}"
    ),
    "weekday_abbr": lambda d: (
        f"{calendar.day_abbr[d.weekday()]}, {calendar.month_abbr[d.month]} {d.day}, {d.year}"
    ),
    "slash_us": lambda d: f"{d.month:02d}/{d.day:02d}/{d.year}",
    "dot_eu": lambda d: f"{d.day:02d}.{d.month:02d}.{d.year}",
    "iso": lambda d: f"{d.year}-{d.month:02d}-{d.day:02d}",  # identity case
    "month_ordinal_year": lambda d: f"{calendar.month_name[d.month]} {ordinal(d.day)}, {d.year}",
    "spelled_out": lambda d: f"{ordinal(d.day)} of {calendar.month_name[d.month]}, {d.year}",
    "dashed_abbr": lambda d: f"{d.day}-{calendar.month_abbr[d.month]}-{d.year}",
}

# test_unseen_format.jsonl. Chosen to be structurally unlike the rest
HELDOUT_FORMATS = ["spelled_out", "dashed_abbr"]

TRAIN_FORMATS = [k for k in FORMATS if k not in HELDOUT_FORMATS]


def iso(d: date) -> str:
    return d.isoformat()


def all_dates(start_year: int, end_year: int) -> list[date]:
    start, end = date(start_year, 1, 1), date(end_year, 12, 31)
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def make_examples(
    rng: random.Random, dates: list[date], format_keys: list[str], n: int
) -> list[dict]:
    """Sample n (surface form -> ISO) pairs, drawing dates from `dates` only.

    Dates repeat across examples under different surface forms -- that is the
    point of the task, and it keeps the split boundary at the date level.
    """
    seen: set[str] = set()
    out: list[dict] = []
    while len(out) < n:
        d = rng.choice(dates)
        key = rng.choice(format_keys)
        src = FORMATS[key](d)
        # Dedup on the rendered string, not (format, date): distinct formats can
        # collide. calendar.month_abbr[5] == month_name[5] == "May", so every
        # May date renders identically under the full-name and abbreviated forms.
        if src in seen:
            continue
        seen.add(src)
        out.append({"src": src, "tgt": iso(d), "format": key})
    rng.shuffle(out)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "dates")
    p.add_argument("--train", type=int, default=50_000)
    p.add_argument("--valid", type=int, default=5_000)
    p.add_argument("--test", type=int, default=5_000)
    p.add_argument("--unseen-format", type=int, default=5_000)
    p.add_argument("--start-year", type=int, default=1950)
    p.add_argument("--end-year", type=int, default=2049)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # Partition the calendar itself, so no date crosses a split boundary.
    pool = all_dates(args.start_year, args.end_year)
    rng.shuffle(pool)
    n_valid = int(0.1 * len(pool))
    n_test = int(0.1 * len(pool))
    valid_dates = pool[:n_valid]
    test_dates = pool[n_valid : n_valid + n_test]
    train_dates = pool[n_valid + n_test :]

    splits = {
        "train": make_examples(rng, train_dates, TRAIN_FORMATS, args.train),
        "valid": make_examples(rng, valid_dates, TRAIN_FORMATS, args.valid),
        "test": make_examples(rng, test_dates, TRAIN_FORMATS, args.test),
        "test_unseen_format": make_examples(
            rng, test_dates, HELDOUT_FORMATS, args.unseen_format
        ),
    }

    for name, rows in splits.items():
        write_jsonl(args.out / f"{name}.jsonl", rows)

    # Vocabulary over every character the corpus can produce. Built from all
    # splits so the test sets never hit <unk> -- with a 100-year range and
    # 13 formats the character inventory is closed anyway.
    chars = sorted({c for rows in splits.values() for r in rows for c in r["src"] + r["tgt"]})
    stoi = {tok: i for i, tok in enumerate(SPECIALS + chars)}
    (args.out / "vocab.json").write_text(
        json.dumps(
            {
                "pad_token": PAD, "pad_id": stoi[PAD],
                "bos_token": BOS, "bos_id": stoi[BOS],
                "eos_token": EOS, "eos_id": stoi[EOS],
                "unk_token": UNK, "unk_id": stoi[UNK],
                "stoi": stoi,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Lengths here are raw character counts. Add 2 before choosing max_seq_len:
    # the target gets <bos> and <eos>.
    max_src = max(len(r["src"]) for rows in splits.values() for r in rows)
    max_tgt = max(len(r["tgt"]) for rows in splits.values() for r in rows)
    meta = {
        "vocab_size": len(stoi),
        "max_src_chars": max_src,
        "max_tgt_chars": max_tgt,
        "suggested_max_seq_len": max(max_src, max_tgt) + 2,
        "counts": {k: len(v) for k, v in splits.items()},
        "train_formats": TRAIN_FORMATS,
        "heldout_formats": HELDOUT_FORMATS,
        "seed": args.seed,
        "year_range": [args.start_year, args.end_year],
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"wrote {args.out}")
    for k, v in splits.items():
        print(f"  {k:<20} {len(v):>6} rows")
    print(f"  vocab_size           {len(stoi):>6}")
    print(f"  max src / tgt chars  {max_src:>6} / {max_tgt}")
    print(f"  suggested max_seq_len{meta['suggested_max_seq_len']:>6}")
    print("\nsample:")
    for row in splits["train"][:5]:
        print(f"  {row['src']:<32} -> {row['tgt']}   [{row['format']}]")


if __name__ == "__main__":
    main()

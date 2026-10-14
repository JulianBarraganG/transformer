import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Vocab:
    stoi: dict[str, int]
    itos: list[str]
    pad_id: int
    bos_id: int
    eos_id: int
    unk_id: int

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> Tensor:
        """Map characters to ids. Characters outside the vocab become `<unk>`."""
        return torch.tensor(
            [self.stoi.get(c, self.unk_id) for c in text], dtype=torch.long
        )

    def decode(self, ids: Tensor) -> str:
        """Map ids back to text, special tokens included (e.g. `<bos>`)."""
        return "".join(self.itos[i] for i in ids.tolist())


def load_vocab(path: Path) -> Vocab:
    raw = json.loads(path.read_text())
    stoi: dict[str, int] = raw["stoi"]
    itos = [""] * len(stoi)
    for token, i in stoi.items():
        itos[i] = token
    return Vocab(
        stoi=stoi,
        itos=itos,
        pad_id=raw["pad_id"],
        bos_id=raw["bos_id"],
        eos_id=raw["eos_id"],
        unk_id=raw["unk_id"],
    )


class DatesDataset(Dataset):
    """One split of the dates data as `(src, tgt)` pairs of raw char ids.

    No special tokens and no padding, both are the collate function's job.
    """

    def __init__(self, path: Path, vocab: Vocab) -> None:
        self.pairs: list[tuple[Tensor, Tensor]] = []
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                self.pairs.append((vocab.encode(row["src"]), vocab.encode(row["tgt"])))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.pairs[idx]


def collate_dates(
    batch: list[tuple[Tensor, Tensor]], vocab: Vocab
) -> tuple[Tensor, Tensor]:
    """Pad to the longest sequence *in this batch* and apply teacher forcing.

    Bind `vocab` with `functools.partial` before passing to `DataLoader`.
    Returns `(src, tgt)`:

        src(source):   a b c <eos>
        tgt(target):   <bos> x y <eos>
    """
    src_len = max(len(src) for src, _ in batch) + 1  # + <eos>
    tgt_len = max(len(tgt) for _, tgt in batch) + 2  # + <bos> & <eos>

    src_out = torch.full((len(batch), src_len), vocab.pad_id, dtype=torch.long)
    tgt_out = torch.full((len(batch), tgt_len), vocab.pad_id, dtype=torch.long)

    for i, (src, tgt) in enumerate(batch):
        n, m = len(src), len(tgt)
        src_out[i, :n] = src
        src_out[i, n] = vocab.eos_id
        tgt_out[i, 0] = vocab.bos_id
        tgt_out[i, 1 : (m + 1)] = tgt
        tgt_out[i, m + 1] = vocab.eos_id

    return src_out, tgt_out

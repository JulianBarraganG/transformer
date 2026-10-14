import torch
from torch import Tensor
from torch.utils.data import Dataset

PAD, BOS, EOS = 0, 1, 2
SEQ_START = 3  # sequence range from (3, max)


class CopyDataset(Dataset):
    """Random token sequences of varying length. The target is the source.

    Yields raw token ids with no special tokens and no padding -- both are the
    collate function's job, since padding length depends on the whole batch.
    """

    def __init__(
        self,
        n_rows: int,
        vocab_size: int,
        min_len: int = 4,
        max_len: int = 12,
        seed: int = 0,
    ) -> None:
        g = torch.Generator().manual_seed(seed)
        lengths = torch.randint(min_len, max_len + 1, (n_rows,), generator=g)
        # Sample above the specials so no "real" token collides with PAD/BOS/EOS
        self.rows = [
            torch.randint(SEQ_START, vocab_size, (int(n),), generator=g)
            for n in lengths
        ]

    def __len__(self) -> int:
        return len(self.rows)
        # TODO: is self.n_rows better or worse?
        # In case of multiple workers, dataset is copied. Memory of object grows.
        # Opting to keep object size minimal, and increase len calc overhead.

    def __getitem__(self, idx: int) -> Tensor:
        return self.rows[idx]


def collate_copy(batch: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Pad to the longest sequence *in this batch* and apply teacher forcing.

    Returns `(src, tgt)`:

        src(source):   a b c
        tgt(target):   <bos> a b c <eos>
    """
    src_len = max(len(row) for row in batch) + 1  # + <eos>
    tgt_len = max(len(row) for row in batch) + 2  # + <bos> & <eos>

    src = torch.full((len(batch), src_len), PAD, dtype=torch.long)
    tgt = torch.full((len(batch), tgt_len), PAD, dtype=torch.long)

    for i, row in enumerate(batch):
        n = len(row)
        src[i, :n] = row
        src[i, n] = EOS
        tgt[i, 0] = BOS
        tgt[i, 1 : (n + 1)] = row
        tgt[i, n + 1] = EOS

    return src, tgt

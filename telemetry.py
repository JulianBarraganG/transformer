import json
from dataclasses import (
    asdict,
    dataclass,
)
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class ValRecord:
    step: int
    epoch: int
    val_loss: float
    n_tokens: int
    kind: str = "val"

@dataclass(frozen=True)
class TrainStepRecord:
    step: int
    epoch: int
    batch_loss: float
    n_tokens: int
    grad_norm: float
    lr: float
    kind: str = "train_step"

class RunLogger:
    def __init__(self, name: str, config: dict, root: Path = Path("runs")) -> None:
        stamp = datetime.now(tz=ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d_%H%M")
        self.config = config
        self.run_dir = root / f"{stamp}_{name}"
        (self.run_dir / "checkpoints").mkdir(exist_ok=True, parents=True)
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2))
        self._metrics_path = self.run_dir / "metrics.jsonl"

    def log(self, record: TrainStepRecord | ValRecord) -> None:
        with self._metrics_path.open("a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

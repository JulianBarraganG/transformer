from dataclasses import dataclass, fields
from logging import getLogger
from pathlib import Path

import polars as pl
import polars.selectors as cs

from tracking.tracking import TrainStepRecord, ValRecord

logger = getLogger(__name__)


@dataclass(frozen=True)
class Run:
    name: str
    config: pl.DataFrame
    train_metrics: pl.DataFrame
    val_metrics: pl.DataFrame


_DTYPES = {int: pl.Int64, float: pl.Float64, str: pl.String}
# An explicit schema, because inference fails on val-only columns: the first
# lines are all train steps, and runs from before validation have no val rows.
_METRICS_SCHEMA = {
    f.name: _DTYPES[f.type]
    for record_cls in (TrainStepRecord, ValRecord)
    for f in fields(record_cls)
}


def _columns(record_cls: type[TrainStepRecord | ValRecord]) -> list[str]:
    return [f.name for f in fields(record_cls) if f.name != "kind"]


def list_runs(root: Path) -> list[Path]:
    run_dirs = sorted(run for run in root.iterdir() if (run / "metrics.jsonl").exists())
    if run_dirs == []:
        logger.warning(
            "`list_runs()` resulted in empty list. Verify correct input root path "
            "and try again."
        )

    return run_dirs


def load_run(run_dir: Path) -> Run:
    """Load a run directory written by `RunLogger`.

    `config` is a single row with one column per setting, prefixed by section
    (e.g. `model.d_model`, `train.lr`).
    """
    config = pl.read_json(run_dir / "config.json").unnest(cs.struct(), separator=".")
    metrics = pl.read_ndjson(run_dir / "metrics.jsonl", schema=_METRICS_SCHEMA)
    train_metrics = metrics.filter(pl.col("kind") == "train_step").select(
        _columns(TrainStepRecord)
    )
    val_metrics = metrics.filter(pl.col("kind") == "val").select(_columns(ValRecord))

    return Run(run_dir.name, config, train_metrics, val_metrics)

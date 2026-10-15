from pathlib import Path

import altair as alt
import polars as pl
import streamlit as st

from tracking.loading import Run, list_runs, load_run

RUNS_ROOT = Path(__file__).parent.resolve() / "runs"
SMOOTH_WINDOW = 50
TRAIN_LABEL = f"Train ({SMOOTH_WINDOW}-step mean)"
VAL_LABEL = "Validation (end of epoch)"
COLORS = {TRAIN_LABEL: "#2a78d6", VAL_LABEL: "#e8603c"}


# Streamlit reruns this whole script on every widget interaction. The mtime
# argument is only part of the cache key: an unchanged file is served from
# cache, a run that is still training gets re-read.
@st.cache_data(max_entries=10)
def cached_load_run(run_dir: Path, mtime: float) -> Run:
    return load_run(run_dir)


def get_steps_per_epoch(run: Run) -> int:
    if "data.steps_per_epoch" in run.config.columns:
        return run.config["data.steps_per_epoch"].item()
    # Runs logged before steps_per_epoch was in config.json are all finished, so
    # the epoch-0 row count is the full epoch.
    return run.train_metrics.filter(pl.col("epoch") == 0).height


def loss_chart(run: Run) -> alt.LayerChart:
    steps_per_epoch = get_steps_per_epoch(run)
    train = run.train_metrics.select(
        x=pl.col("step") / steps_per_epoch,
        loss=pl.col("batch_loss"),
        mean=pl.col("batch_loss").rolling_mean(SMOOTH_WINDOW, min_samples=1),
        series=pl.lit(TRAIN_LABEL),
    )
    val = run.val_metrics.select(
        x=pl.col("step") / steps_per_epoch,
        epoch=pl.col("epoch") + 1,
        loss=pl.col("val_loss"),
        series=pl.lit(VAL_LABEL),
    )

    x = alt.X("x:Q", title=f"Epoch ({steps_per_epoch} steps each)")
    y_scale = alt.Scale(type="log")
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
        legend=alt.Legend(title=None, orient="top-right"),
    )

    raw = (
        alt.Chart(train)
        .mark_line(color=COLORS[TRAIN_LABEL], opacity=0.25, strokeWidth=1)
        .encode(x=x, y=alt.Y("loss:Q", scale=y_scale, title="Loss (log)"))
    )
    smoothed = (
        alt.Chart(train)
        .mark_line(strokeWidth=2)
        .encode(x=x, y=alt.Y("mean:Q", scale=y_scale), color=color)
    )
    validation = (
        alt.Chart(val)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=50))
        .encode(
            x=x,
            y=alt.Y("loss:Q", scale=y_scale),
            color=color,
            tooltip=["epoch:Q", alt.Tooltip("loss:Q", format=".2e")],
        )
    )
    return (raw + smoothed + validation).properties(height=400).interactive()


def grad_norm_chart(run: Run) -> alt.LayerChart:
    steps_per_epoch = get_steps_per_epoch(run)
    train = run.train_metrics.select(
        x=pl.col("step") / steps_per_epoch,
        grad_norm=pl.col("grad_norm"),
        mean=pl.col("grad_norm").rolling_mean(SMOOTH_WINDOW, min_samples=1),
    )

    x = alt.X("x:Q", title=f"Epoch ({steps_per_epoch} steps each)")
    y_scale = alt.Scale(type="log")
    color = COLORS[TRAIN_LABEL]

    raw = (
        alt.Chart(train)
        .mark_line(color=color, opacity=0.25, strokeWidth=1)
        .encode(x=x, y=alt.Y("grad_norm:Q", scale=y_scale, title="Grad norm (log)"))
    )
    smoothed = (
        alt.Chart(train)
        .mark_line(color=color, strokeWidth=2)
        .encode(x=x, y=alt.Y("mean:Q", scale=y_scale))
    )
    return (raw + smoothed).properties(height=250).interactive()


def lr_chart(run: Run) -> alt.Chart:
    steps_per_epoch = get_steps_per_epoch(run)
    train = run.train_metrics.select(x=pl.col("step") / steps_per_epoch, lr="lr")

    return (
        alt.Chart(train)
        .mark_line(color=COLORS[TRAIN_LABEL], strokeWidth=2)
        .encode(
            x=alt.X("x:Q", title=f"Epoch ({steps_per_epoch} steps each)"),
            y=alt.Y("lr:Q", title="LR", axis=alt.Axis(format=".0e")),
        )
        .properties(height=200)
        .interactive()
    )


st.set_page_config(page_title="Training runs", layout="wide")
st.title("Training runs")

run_dirs = list_runs(RUNS_ROOT)
if not run_dirs:
    st.info(f"No runs found in {RUNS_ROOT}")
    st.stop()

latest_run_dir = st.sidebar.selectbox(
    "Run", run_dirs, index=len(run_dirs) - 1, format_func=lambda d: d.name
)


@st.fragment(run_every="5s")
def live_metrics(run_dir: Path):
    run = cached_load_run(run_dir, (run_dir / "metrics.jsonl").stat().st_mtime)

    st.subheader("Loss per step")
    if run.val_metrics.is_empty():
        st.caption("This run has no validation records.")
    st.altair_chart(loss_chart(run), width="stretch")

    st.subheader("Gradient norm (total L2, before step)")
    st.caption(f"Faint: every step. Solid: {SMOOTH_WINDOW}-step mean.")
    st.altair_chart(grad_norm_chart(run), width="stretch")

    st.subheader("Learning rate")
    st.altair_chart(lr_chart(run), width="stretch")


live_metrics(latest_run_dir)

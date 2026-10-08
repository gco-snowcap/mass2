"""Experiment states for live data: the DASTARD `*_experiment_state.txt` format, written and read while it grows.

The file is plain text, one state change per line, appended by the DAQ as each change happens:

    # unix time in nanoseconds, state label
    1722087044504155611, START
    1722087044757970883, CAL2

A record's state is the last state change at or before its timestamp, the same backward as-of join that
`mass2.Channel.with_experiment_state_df` performs offline.
"""

from pathlib import Path

import polars as pl

from .histogram import NO_STATE

STATE_SCHEMA: dict[str, pl.DataType] = {"timestamp": pl.Datetime("us", "UTC"), "state_label": pl.String()}


def parse_state_text(text: str) -> pl.DataFrame:
    """Parse state-file text into (timestamp, state_label), ignoring comments and an unfinished last line."""
    times, labels = [], []
    for line in text.splitlines(keepends=True):
        if not line.endswith("\n") or line.startswith("#") or not line.strip():
            continue
        unixnano, label = line.split(",", 1)
        times.append(int(unixnano) // 1000)
        labels.append(label.strip())
    return pl.DataFrame(
        {"timestamp": times, "state_label": labels}, schema={"timestamp": pl.Int64, "state_label": pl.String}
    ).with_columns(pl.col("timestamp").cast(STATE_SCHEMA["timestamp"]))


class StateFileWriter:
    """Append state changes to a state file, flushing each line so a follower sees it at once."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", encoding="utf-8")
        self._file.write("# unix time in nanoseconds, state label\n")
        self._file.flush()

    def write(self, unixnano: int, label: str) -> None:
        self._file.write(f"{unixnano}, {label}\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class StateFileFollower:
    """Re-reads a growing state file whenever it changes. `path=None` means there are no states."""

    def __init__(self, path: str | Path | None):
        self.path = None if path is None else Path(path)
        self._size = -1
        self.states = pl.DataFrame(schema=STATE_SCHEMA)

    def poll(self) -> pl.DataFrame:
        """The state changes known so far, oldest first."""
        if self.path is not None and self.path.exists():
            size = self.path.stat().st_size
            if size != self._size:
                self._size = size
                self.states = parse_state_text(self.path.read_text(encoding="utf-8"))
        return self.states

    def label(self, df: pl.DataFrame) -> pl.DataFrame:
        """`df` with a `state_label` column, in its original row order. Records before any state get NO_STATE."""
        states = self.poll().sort("timestamp")
        labeled = (
            df
            .drop("state_label", strict=False)
            .with_row_index("_row")
            .sort("timestamp")
            .join_asof(states, on="timestamp", strategy="backward")
            .sort("_row")
            .drop("_row")
        )
        return labeled.with_columns(pl.col("state_label").fill_null(NO_STATE))

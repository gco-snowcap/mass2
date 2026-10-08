"""Energy histograms of good pulses, per channel and experiment state, accumulated in fixed time slices.

Slices are aligned to multiples of `slice_s` since the Unix epoch. A slice is "open" until the data have
advanced `grace_s` past its end; then it is finalized and returned by `SlicedHistogrammer.add`. Records that
arrive for an already-finalized slice are counted in `late_records` and otherwise dropped.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from numpy.typing import NDArray


@dataclass(frozen=True)
class HistogramSpec:
    """Energy binning and slice length shared by every histogram."""

    e_lo: float = 0.0
    e_hi: float = 1200.0
    bin_width: float = 1.0
    slice_s: float = 10.0

    @property
    def nbins(self) -> int:
        return int(round((self.e_hi - self.e_lo) / self.bin_width))

    @property
    def slice_us(self) -> int:
        return int(round(self.slice_s * 1e6))

    def to_dict(self) -> dict:
        return {"e_lo": self.e_lo, "e_hi": self.e_hi, "bin_width": self.bin_width, "slice_s": self.slice_s, "nbins": self.nbins}


NO_STATE = "(none)"  # state label for records with no experiment state (no state file, or before its first line)


@dataclass
class HistogramSlice:
    """Counts per (channel, state) for one time slice starting at `start_us` (µs since the epoch, UTC)."""

    start_us: int
    nbins: int
    counts: dict[tuple[int, str], NDArray[np.int64]] = field(default_factory=dict)

    def get(self, ch_num: int, state: str) -> NDArray[np.int64]:
        """The counts for one channel and state, created as zeros if absent."""
        key = (ch_num, state)
        if key not in self.counts:
            self.counts[key] = np.zeros(self.nbins, dtype=np.int64)
        return self.counts[key]

    def total(self) -> int:
        return int(sum(c.sum() for c in self.counts.values()))


class SlicedHistogrammer:
    """Accumulate good-pulse energies into per-channel histograms, one per time slice."""

    def __init__(
        self, spec: HistogramSpec, energy_col: str, good_col: str = "good", state_col: str = "state_label", grace_s: float = 1.0
    ):
        self.spec = spec
        self.energy_col = energy_col
        self.good_col = good_col
        self.state_col = state_col
        self.grace_us = int(grace_s * 1e6)
        self._open: dict[int, HistogramSlice] = {}  # slice start (us) -> slice
        self._finalized_before_us: int | None = None  # every slice starting earlier than this is finalized
        self.latest_us: int | None = None
        self.first_us: int | None = None
        self.late_records = 0
        self.counted = 0  # good, in-range records histogrammed so far

    def add(self, df: pl.DataFrame) -> list[HistogramSlice]:
        """Count the good, in-range records of `df`. Return the slices this finalized, oldest first."""
        if len(df) == 0:
            return []
        spec = self.spec
        t_us = pl.col("timestamp").dt.epoch("us")
        latest, first = df.select(t_us.max(), t_us.min().alias("first")).row(0)
        self.first_us = first if self.first_us is None else min(self.first_us, first)
        self.latest_us = latest if self.latest_us is None else max(self.latest_us, latest)

        state = pl.col(self.state_col).cast(pl.String) if self.state_col in df.columns else pl.lit(None, pl.String)
        binned = (
            df
            .lazy()
            .filter(pl.col(self.good_col), pl.col(self.energy_col).is_finite())
            .select(
                "ch_num",
                state=state.fill_null(NO_STATE),
                slice_start=(t_us // spec.slice_us) * spec.slice_us,
                bin=((pl.col(self.energy_col) - spec.e_lo) / spec.bin_width).floor().cast(pl.Int64),
            )
            .filter(pl.col("bin").is_between(0, spec.nbins - 1))
            .group_by("slice_start", "ch_num", "state", "bin")
            .len()
            .collect()
        )
        if self._finalized_before_us is not None:
            late = binned["slice_start"] < self._finalized_before_us
            self.late_records += int(binned.filter(late)["len"].sum())
            binned = binned.filter(~late)

        self.counted += int(binned["len"].sum())
        for (start, ch_num, state_label), group in binned.group_by("slice_start", "ch_num", "state"):
            hist_slice = self._open.setdefault(start, HistogramSlice(start, spec.nbins))
            np.add.at(hist_slice.get(ch_num, state_label), group["bin"].to_numpy(), group["len"].to_numpy())

        # Make sure the slice holding the newest data exists, even if it has no good counts yet.
        latest_us = self.latest_us
        newest = (latest_us // spec.slice_us) * spec.slice_us
        self._open.setdefault(newest, HistogramSlice(newest, spec.nbins))
        return self._finalize(lambda start: start + spec.slice_us + self.grace_us <= latest_us)

    def flush(self) -> list[HistogramSlice]:
        """Finalize and return every open slice (use at end of stream)."""
        return self._finalize(lambda start: True)

    def open_slices(self) -> list[HistogramSlice]:
        """The slices still accumulating, oldest first. Normally just the current one."""
        return [self._open[k] for k in sorted(self._open)]

    def _finalize(self, is_done: Callable[[int], bool]) -> list[HistogramSlice]:
        done = [self._open.pop(k) for k in sorted(self._open) if is_done(k)]
        if done:
            self._finalized_before_us = done[-1].start_us + self.spec.slice_us
        return done


def slices_to_df(slices: list[HistogramSlice]) -> pl.DataFrame:
    """One row per (slice, channel, state): slice_start, ch_num, state_label, counts (Array(UInt32, nbins))."""
    if not slices:
        return pl.DataFrame(
            schema={
                "slice_start": pl.Datetime("us", "UTC"),
                "ch_num": pl.Int64,
                "state_label": pl.String,
                "counts": pl.List(pl.UInt32),
            }
        )
    starts, chans, states, counts = [], [], [], []
    for s in slices:
        for ch_num, state in sorted(s.counts):
            starts.append(s.start_us)
            chans.append(ch_num)
            states.append(state)
            counts.append(s.counts[ch_num, state])
    nbins = slices[0].nbins
    return pl.DataFrame({
        "slice_start": pl.Series(starts, dtype=pl.Int64).cast(pl.Datetime("us", "UTC")),
        "ch_num": pl.Series(chans, dtype=pl.Int64),
        "state_label": pl.Series(states, dtype=pl.String),
        "counts": np.stack(counts).astype(np.uint32) if counts else np.zeros((0, nbins), np.uint32),
    })


def df_to_slices(df: pl.DataFrame, spec: HistogramSpec) -> list[HistogramSlice]:
    """Inverse of `slices_to_df`, oldest slice first."""
    slices: dict[int, HistogramSlice] = {}
    if len(df) == 0:
        return []
    counts = np.asarray(df["counts"].to_list(), dtype=np.int64)
    keys = zip(df["slice_start"].dt.epoch("us").to_list(), df["ch_num"].to_list(), df["state_label"].to_list())
    for row, (start, ch_num, state) in enumerate(keys):
        slices.setdefault(start, HistogramSlice(start, spec.nbins)).counts[ch_num, state] = counts[row]
    return [slices[k] for k in sorted(slices)]


def sparse_counts(counts: dict[tuple[int, str], NDArray]) -> dict[str, dict[str, list[int]]]:
    """{state: {channel: [bin gap, count, bin gap, count, ...]}}: nonzero bins only, bin numbers delta-coded.

    This is how slices travel to the viewer: with bins as fine as the resolution fits use (up to ~20,000),
    nearly all bins of a short slice are empty.
    """
    out: dict[str, dict[str, list[int]]] = {}
    for (ch, state), c in counts.items():
        nz = np.flatnonzero(c)
        if len(nz):
            gaps = np.diff(nz, prepend=0)
            out.setdefault(state, {})[str(ch)] = np.column_stack([gaps, c[nz]]).ravel().tolist()
    return out

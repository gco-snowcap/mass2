"""Simulate a live data acquisition by replaying LJH pulse records into a growing Arrow IPC stream file.

All channels go into one stream, merged in time order, `chunk_size` records per record batch. The columns are

    chunk          UInt64   sequence number of the record batch, starting at 0
    ch_num         Int64    channel number
    timestamp      Datetime(us, UTC)
    subframecount  UInt64
    pulse          Array(UInt16, n_samples)

Records keep their original timestamps, and are written at a playback speed (a multiple of real time) that
can be changed while running through --speed-file. With `repeats` the data are replayed again after the end,
continuing the timeline. If the data have an experiment-state
file, its state changes are replayed on the same timeline into OUT_experiment_state.txt, each line appended
just before the first chunk that follows it, as DASTARD would. A `ScaledChannel` adds a fake channel
whose pulses are a source channel's pulses with the signal multiplied by a constant, as a detector of a
different gain would record them; it needs a recipe of its own to come out at the right energies.

Command line:  mass2-live-sim OUT.arrows [--ljh-folder DIR] [--repeats N] [--speed X] [--scale 4219:14219:1.03]
               [--no-states]
"""

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl
import pulsedata
from numpy.typing import NDArray

import mass2
from mass2.core import ljhutil
from ..parent import exit_with_parent
from ..arrow_stream import ArrowStreamWriter
from ..states import StateFileWriter, parse_state_text


@dataclass(frozen=True)
class SimSource:
    """The raw records of one channel, held in memory for replay.

    A gain-scaled copy shares its source's `pulses` array and is scaled chunk by chunk as it is written, so
    an array of many copies costs no extra memory.
    """

    ch_num: int
    pulses: NDArray[np.uint16]  # shape (npulses, nsamples)
    timestamp_us: NDArray[np.int64]
    subframecount: NDArray[np.uint64]
    n_presamples: int
    gain: float = 1.0
    scale_baseline: bool = False

    def records(self, rows: NDArray) -> NDArray[np.uint16]:
        """The pulse records at `rows`, with this channel's gain applied."""
        if self.gain == 1.0:
            return self.pulses[rows]
        return scale_pulses(self.pulses[rows], self.gain, self.n_presamples, self.scale_baseline)


@dataclass(frozen=True)
class ScaledChannel:
    """A fake channel `new_ch` made from channel `source_ch` with its signal multiplied by `factor`."""

    source_ch: int
    new_ch: int
    factor: float

    @classmethod
    def parse(cls, text: str) -> "ScaledChannel":
        """Parse 'SOURCE:NEW:FACTOR', e.g. '4219:14219:1.03'."""
        source, new, factor = text.split(":")
        return cls(int(source), int(new), float(factor))


def load_ljh_sources(pulse_folder: str | Path, max_pulses: int | None = None) -> dict[int, SimSource]:
    """Load every channel's pulses, timestamps and subframecounts from the LJH files in `pulse_folder`."""
    data = mass2.Channels.from_ljh_folder(pulse_folder)
    sources = {}
    for ch_num, ch in data.channels.items():
        assert ch.pulseframer is not None, f"channel {ch_num} has no raw pulses"
        n = ch.npulses if max_pulses is None else min(max_pulses, ch.npulses)
        sources[ch_num] = SimSource(
            ch_num=ch_num,
            pulses=ch.pulseframer.load_raw_chunk(0, n)["pulse"].to_numpy(),
            timestamp_us=ch.df["timestamp"].head(n).dt.epoch("us").to_numpy(),
            subframecount=ch.df["subframecount"].head(n).to_numpy(),
            n_presamples=ch.header.n_presamples,
        )
    return sources


def load_experiment_states(pulse_folder: str | Path) -> pl.DataFrame | None:
    """The experiment-state changes recorded alongside the LJH files in `pulse_folder`, or None if there is no file."""
    ljh = next(Path(pulse_folder).glob("*_chan*.ljh"))
    path = Path(ljhutil.experiment_state_path_from_ljh_path(ljh))
    return parse_state_text(path.read_text(encoding="utf-8")) if path.exists() else None


def state_file_path(stream_path: str | Path) -> Path:
    """Where the simulator writes the state file for the stream at `stream_path`."""
    stream_path = Path(stream_path)
    return stream_path.with_name(f"{stream_path.stem}_experiment_state.txt")


def scale_pulses(pulses: NDArray, factor: float, n_presamples: int, scale_baseline: bool = False) -> NDArray[np.uint16]:
    """Multiply pulse records by `factor`, rounding and clipping to the uint16 range.

    By default only the signal is scaled: each record's pretrigger baseline is held fixed and its deviations from
    the baseline are multiplied. Multiplying the raw samples (`scale_baseline=True`) also moves the pretrigger
    mean by (factor-1)*baseline, which a drift correction will read as a large drift, so the resulting gain error
    would not be a pure constant.
    """
    x = pulses.astype(np.float32)
    if scale_baseline:
        x *= factor
    else:
        baseline = x[:, :n_presamples].mean(axis=1, keepdims=True)
        x = baseline + factor * (x - baseline)
    return np.clip(np.rint(x), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def simulate(
    sources: dict[int, SimSource],
    out_path: str | Path,
    *,
    chunk_size: int = 100,
    repeats: int = 1,
    scaled: Sequence[ScaledChannel] = (),
    speed: float = 5.0,
    speed_file: str | Path | None = None,
    pace: bool = True,
    scale_baseline: bool = False,
    start_time_us: int | None = None,
    states: pl.DataFrame | None = None,
) -> int:
    """Write the sources (plus any scaled channels) to an Arrow IPC stream; return the number of chunks written.

    Records keep their ORIGINAL timestamps (shifted to begin at `start_time_us` if given), so every time and
    rate downstream is a real one. With `pace`, chunks are written at `speed` times real time: a chunk goes
    out once the wall clock reaches its last record's time, scaled by the speed. `speed_file`, if given, is a
    JSON file {"speed": x} re-read before each chunk, so the speed can change while running. Without `pace`,
    everything is written as fast as possible. Repeats continue the timeline after the end of the data.
    `states` (timestamp, state_label) are replayed into `state_file_path(out_path)`.
    """
    sources = dict(sources)
    for s in scaled:
        src = sources[s.source_ch]
        sources[s.new_ch] = replace(src, ch_num=s.new_ch, gain=s.factor, scale_baseline=scale_baseline)

    # One replay cycle, merged across channels in time order (stable, so ties keep channel order).
    ch_nums = np.concatenate([np.full(len(s.timestamp_us), ch) for ch, s in sources.items()])
    rows = np.concatenate([np.arange(len(s.timestamp_us)) for s in sources.values()])
    t_us = np.concatenate([s.timestamp_us for s in sources.values()])
    sfc = np.concatenate([s.subframecount for s in sources.values()])
    order = np.argsort(t_us, kind="stable")
    ch_nums, rows, t_us, sfc = ch_nums[order], rows[order], t_us[order], sfc[order]

    timeline = _Timeline(
        t0_us=int(t_us[0]),
        cycle_us=int(t_us[-1] - t_us[0]) + 1000,  # one cycle's duration, plus 1 ms so cycles don't overlap
        start_us=int(t_us[0]) if start_time_us is None else start_time_us,
        rate=1.0,  # original time: one second of data is one second of timestamp
    )
    cycle_sfc = int(sfc.max() - sfc.min()) + 1
    state_replay = _StateReplay(states, timeline, state_file_path(out_path)) if states is not None and len(states) else None
    pacer = _Pacer(speed, speed_file)

    nchunks = 0
    cycle = 0
    with ArrowStreamWriter(out_path) as writer:
        while repeats <= 0 or cycle < repeats:
            sim_t_us = timeline.to_sim(t_us - timeline.t0_us, cycle)
            sim_sfc = sfc + np.uint64(cycle * cycle_sfc)
            for lo in range(0, len(t_us), chunk_size):
                hi = min(lo + chunk_size, len(t_us))
                if pace:
                    pacer.wait_for(float(sim_t_us[hi - 1]))
                if state_replay is not None:
                    state_replay.write_until(sim_t_us[hi - 1], cycle)  # a state change precedes the records after it
                writer.write(_chunk_frame(nchunks, sources, ch_nums[lo:hi], rows[lo:hi], sim_t_us[lo:hi], sim_sfc[lo:hi]))
                nchunks += 1
            cycle += 1
    if state_replay is not None:
        state_replay.close()
    return nchunks


class _Pacer:
    """Holds back each chunk until the wall clock catches up with its data time at the current speed."""

    def __init__(self, speed: float, speed_file: str | Path | None):
        self.speed = float(speed)
        self.speed_file = None if speed_file is None else Path(speed_file)
        self._file_mtime = 0.0
        self.data_anchor_us: float | None = None
        self.wall_anchor = 0.0

    def _read_speed(self) -> float:
        if self.speed_file is not None and self.speed_file.exists():
            mtime = self.speed_file.stat().st_mtime
            if mtime != self._file_mtime:
                self._file_mtime = mtime
                try:
                    return float(json.loads(self.speed_file.read_text())["speed"])
                except (ValueError, KeyError, json.JSONDecodeError):
                    pass  # a half-written file; keep the current speed
        return self.speed

    def wait_for(self, data_us: float) -> None:
        if self.data_anchor_us is None:
            self.data_anchor_us, self.wall_anchor = data_us, time.time()
        new_speed = self._read_speed()
        if new_speed != self.speed and new_speed > 0:
            # Re-anchor at the present position, so changing speed never jumps or stalls the data.
            now = time.time()
            elapsed_data = (now - self.wall_anchor) * self.speed * 1e6
            self.data_anchor_us, self.wall_anchor, self.speed = self.data_anchor_us + elapsed_data, now, new_speed
        _sleep_until(self.wall_anchor + (data_us - self.data_anchor_us) / 1e6 / self.speed)


@dataclass(frozen=True)
class _Timeline:
    """Maps offsets into one cycle of the original data onto the written timestamps (later cycles follow on)."""

    t0_us: int
    cycle_us: int
    start_us: int
    rate: float

    def to_sim(self, offset_us: NDArray, cycle: int) -> NDArray:
        return self.start_us + (offset_us + cycle * self.cycle_us) / self.rate


class _StateReplay:
    """Writes each cycle's state changes on the simulated clock, never ahead of the records."""

    def __init__(self, states: pl.DataFrame, timeline: _Timeline, path: Path):
        # Offsets into the cycle, clipped to it so one cycle's last state never follows the next cycle's first.
        offsets = states["timestamp"].dt.epoch("us").to_numpy() - timeline.t0_us
        self.offsets = np.clip(offsets, 0, timeline.cycle_us - 1)
        self.labels = states["state_label"].to_list()
        self.timeline = timeline
        self.writer = StateFileWriter(path)
        self.cycle = 0
        self.next = 0

    def write_until(self, sim_us: float, cycle: int) -> None:
        if cycle != self.cycle:
            self.cycle, self.next = cycle, 0
        times = self.timeline.to_sim(self.offsets, cycle)
        while self.next < len(self.labels) and times[self.next] <= sim_us:
            self.writer.write(int(times[self.next] * 1000), self.labels[self.next])
            self.next += 1

    def close(self) -> None:
        self.writer.close()


def _chunk_frame(
    chunk: int, sources: dict[int, SimSource], ch_nums: NDArray, rows: NDArray, t_us: NDArray, sfc: NDArray
) -> pl.DataFrame:
    """Build one chunk's DataFrame, gathering each row's pulse from its channel's array."""
    nsamples = next(iter(sources.values())).pulses.shape[1]
    pulses = np.empty((len(rows), nsamples), dtype=np.uint16)
    for ch in np.unique(ch_nums):
        here = ch_nums == ch
        pulses[here] = sources[int(ch)].records(rows[here])
    return pl.DataFrame({
        "chunk": pl.Series(np.full(len(rows), chunk, dtype=np.uint64)),
        "ch_num": pl.Series(ch_nums.astype(np.int64)),
        "timestamp": pl.Series(t_us.astype(np.int64)).cast(pl.Datetime("us", "UTC")),
        "subframecount": pl.Series(sfc.astype(np.uint64)),
        "pulse": pulses,
    })


def _sleep_until(t_s: float) -> None:
    delay = t_s - time.time()
    if delay > 0:
        time.sleep(delay)


def default_pulse_folder() -> Path:
    """The BESSY 2024-07-27 pulse data from the `pulsedata` package, also used by the mass2 tests."""
    return pulsedata.pulse_noise_ljh_pairs["bessy_20240727"].pulse_folder


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `mass2-live-sim`."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, help="Arrow IPC stream file to write (conventionally *.arrows)")
    p.add_argument("--ljh-folder", type=Path, default=None, help="folder of LJH pulse files (default: pulsedata bessy_20240727)")
    p.add_argument("--max-pulses", type=int, default=None, help="use at most this many records per channel")
    p.add_argument("--chunk-size", type=int, default=100, help="records per record batch (default 100)")
    p.add_argument("--repeats", type=int, default=1, help="replay the data this many times; 0 = forever (default 1)")
    p.add_argument("--speed", type=float, default=5.0, help="playback speed, in multiples of real time (default 5)")
    p.add_argument("--speed-file", type=Path, default=None, help='JSON {"speed": x}, re-read before each chunk to change speed live')
    p.add_argument("--as-fast-as-possible", action="store_true", help="write without pacing (timestamps are original either way)")
    p.add_argument(
        "--scale", action="append", default=[], type=ScaledChannel.parse, metavar="SRC:NEW:FACTOR",
        help="add channel NEW made from channel SRC with its signal times FACTOR; may repeat",
    )  # fmt: skip
    p.add_argument("--scale-baseline", action="store_true", help="--scale multiplies raw samples, baseline included")
    p.add_argument("--no-states", action="store_true", help="do not replay the experiment-state file")
    args = p.parse_args(argv)
    exit_with_parent()

    folder = args.ljh_folder or default_pulse_folder()
    sources = load_ljh_sources(folder, args.max_pulses)
    states = None if args.no_states else load_experiment_states(folder)
    print(f"mass2-live-sim: channels {sorted(sources)} + scaled {[s.new_ch for s in args.scale]} -> {args.out}", flush=True)
    try:
        n = simulate(
            sources, args.out, chunk_size=args.chunk_size, repeats=args.repeats, scaled=args.scale,
            speed=args.speed, speed_file=args.speed_file, pace=not args.as_fast_as_possible, scale_baseline=args.scale_baseline,
            states=states,
        )  # fmt: skip
        print(f"mass2-live-sim: finished, wrote {n} chunks", flush=True)
    except KeyboardInterrupt:
        print("mass2-live-sim: interrupted; stream closed", flush=True)


if __name__ == "__main__":
    main()

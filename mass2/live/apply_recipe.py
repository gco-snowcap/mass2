"""Apply a saved mass2 recipe to a live (or finished) Arrow IPC stream of pulse records.

`run_live` is the whole loop. Every `poll_s` seconds it reads the complete record batches appended to the
input stream, tags each record with its experiment state, runs each channel's recipe, and appends one output
batch per input batch (same rows, same order, same `chunk` and `ch_num`) to the output stream. Good pulses
are histogrammed in energy per channel and state, in fixed time slices. It stops after the input's
end-of-stream marker, or on Ctrl-C.

Output columns: every input column except `pulse`, every recipe output, `good` (the recipe's final
good-pulse expression), and `state_label`. In HIST_DIR:

    histograms_meta.json       binning, slice length, energy column, recipe steps (plus anything in --meta)
    histograms.arrows          finished slices, appended as they complete (slice_start, ch_num, state_label, counts)
    histograms_current.arrows  the slice still filling, rewritten after each poll
    states.json, status.json   state changes seen so far, and running counters, rewritten after each poll

Command line:  mass2-live-apply RECIPE.pkl IN.arrows OUT.arrows HIST_DIR [--state-file F] [--alias 14219=4219] ...
"""

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import polars as pl

import mass2
from mass2.core.misc import PulseDataFromNumpy
from mass2.core.recipe import Recipe
from .parent import exit_with_parent
from .arrow_stream import ArrowStreamTailer, ArrowStreamWriter, write_stream_atomically, write_text_atomically
from .histogram import HistogramSpec, SlicedHistogrammer, slices_to_df
from .states import StateFileFollower


MAX_RECORDS = 2000  # records given to the recipe at once; its filters hold a few float64 copies of them


def run_live(
    recipe_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    hist_dir: str | Path,
    *,
    spec: HistogramSpec = HistogramSpec(),
    energy_col: str = "energy_5lagy_best",
    aliases: dict[int, int] | None = None,
    state_path: str | Path | None = None,
    meta: dict | None = None,
    poll_s: float = 0.5,
    grace_s: float = 1.0,
    max_records: int = MAX_RECORDS,
) -> None:
    """Follow `input_path` until its stream ends, writing results and histograms. See the module docstring."""
    hist_dir = Path(hist_dir)
    hist_dir.mkdir(parents=True, exist_ok=True)
    recipes = mass2.misc.unpickle_object(recipe_path)
    applier = LiveRecipeApplier(recipes, aliases)
    states = StateFileFollower(state_path)
    histogrammer = SlicedHistogrammer(spec, energy_col=energy_col, grace_s=grace_s)
    reader = ArrowStreamTailer(input_path, max_read_bytes=16 * 1024 * 1024)
    write_meta(hist_dir, spec, energy_col, recipes, poll_s, meta)
    counts = {"records": 0, "chunks": 0, "polls": 0}

    # Leaving this block for any reason (end of stream, Ctrl-C) writes both streams' end-of-stream markers.
    with ArrowStreamWriter(output_path) as results, ArrowStreamWriter(hist_dir / "histograms.arrows") as finished:
        while not reader.ended:
            t_poll = time.time()
            batches = reader.poll()  # every complete batch appended since the last poll
            for group in _groups(batches, max_records):  # the recipe sees at most max_records at once: bounded memory
                result = applier.process(states.label(pl.concat(group, how="vertical_relaxed")))
                start = 0
                for batch in group:  # one output batch per input batch
                    results.write(result.slice(start, len(batch)))
                    start += len(batch)
                if done := histogrammer.add(result):
                    finished.write(slices_to_df(done))
            if reader.ended and (done := histogrammer.flush()):
                finished.write(slices_to_df(done))
            counts["records"] += sum(len(b) for b in batches)
            counts["chunks"] += len(batches)
            counts["polls"] += 1
            publish(hist_dir, histogrammer, states, applier, reader, results, counts, len(batches))
            if not reader.ended:
                time.sleep(max(0.0, poll_s - (time.time() - t_poll)))


def _groups(batches: list[pl.DataFrame], max_records: int) -> list[list[pl.DataFrame]]:
    """Consecutive batches, grouped so each group holds at most `max_records` records (a bigger batch goes alone)."""
    groups: list[list[pl.DataFrame]] = []
    n = max_records
    for b in batches:
        if n + len(b) > max_records:
            groups.append([])
            n = 0
        groups[-1].append(b)
        n += len(b)
    return groups


class LiveRecipeApplier:
    """Apply per-channel recipes to multi-channel DataFrames of raw records.

    `aliases` maps a channel number to a channel whose recipe it borrows when it has no recipe of its own.
    Rows of channels with no recipe at all pass through with null outputs and `good = False`.
    """

    def __init__(self, recipes: dict[int, Recipe], aliases: dict[int, int] | None = None):
        self.recipes = recipes
        self.aliases = aliases or {}
        self.channels_without_recipe: set[int] = set()

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return the recipe outputs for `df`, row for row, without the `pulse` column."""
        df = df.with_row_index("_row")
        parts = []
        for (ch_num,), df_ch in df.partition_by("ch_num", as_dict=True, maintain_order=True).items():
            recipe = self.recipes.get(ch_num, self.recipes.get(self.aliases.get(ch_num, ch_num)))
            if recipe is None:
                self.channels_without_recipe.add(ch_num)
                parts.append(df_ch.drop("pulse").with_columns(good=pl.lit(False)))
                continue
            out = recipe.calc_from_df(df_ch.drop("pulse"), PulseDataFromNumpy(df_ch["pulse"].to_numpy()))
            parts.append(out.with_columns(good=recipe[-1].good_expr.fill_null(False)))
        return pl.concat(parts, how="diagonal_relaxed").sort("_row").drop("_row")


# ---- what the viewer reads: metadata once, then a snapshot after every poll


def write_meta(hist_dir: Path, spec: HistogramSpec, energy_col: str, recipes: dict, poll_s: float, extra: dict | None) -> None:
    """histograms_meta.json: how to read the histograms, and what made them. `extra` is merged in (e.g. a layout)."""
    meta = spec.to_dict() | {
        "energy_col": energy_col,
        "poll_s": poll_s,
        "recipe_steps": [type(step).__name__ for step in next(iter(recipes.values()))],
        "recipe_channels": sorted(recipes),
    }
    (hist_dir / "histograms_meta.json").write_text(json.dumps(meta | (extra or {}), indent=1))


def publish(
    hist_dir: Path,
    histogrammer: SlicedHistogrammer,
    states: StateFileFollower,
    applier: LiveRecipeApplier,
    reader: ArrowStreamTailer,
    results: ArrowStreamWriter,
    counts: dict,
    new_chunks: int,
) -> None:
    """Rewrite the open slice, the state timeline, and the running counters."""
    write_stream_atomically(slices_to_df(histogrammer.open_slices()), hist_dir / "histograms_current.arrows")
    known = states.poll()
    timeline = [[t / 1e6, label] for t, label in zip(known["timestamp"].dt.epoch("us").to_list(), known["state_label"].to_list())]
    write_text_atomically(json.dumps(timeline), hist_dir / "states.json")
    status = counts | {
        "good_pulses": histogrammer.counted,
        "late_records": histogrammer.late_records,
        "first_data_us": histogrammer.first_us,
        "latest_data_us": histogrammer.latest_us,
        "channels_without_recipe": sorted(applier.channels_without_recipe),
        "input_bytes": reader.bytes_read,
        "output_bytes": results.bytes_written,
        "state_changes": len(timeline),
        "stream_ended": reader.ended,
        "last_poll": {"new_chunks": new_chunks},
        "updated_unix_s": time.time(),
    }
    write_text_atomically(json.dumps(status), hist_dir / "status.json")


def parse_alias(text: str) -> tuple[int, int]:
    """Parse 'NEW=SOURCE', e.g. '14219=4219'."""
    new, source = text.split("=")
    return int(new), int(source)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `mass2-live-apply`."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recipe", type=Path, help="recipe pickle from Channels.save_recipes")
    p.add_argument("input", type=Path, help="Arrow IPC stream of raw records (may still be growing, or not exist yet)")
    p.add_argument("output", type=Path, help="Arrow IPC stream to write (overwritten)")
    p.add_argument("hist_dir", type=Path, help="directory for histogram files")
    p.add_argument("--energy-col", default="energy_5lagy_best", help="recipe output to histogram (default energy_5lagy_best)")
    p.add_argument("--state-file", type=Path, default=None, help="experiment-state file to follow (default: none)")
    p.add_argument("--alias", action="append", default=[], type=parse_alias, metavar="NEW=SRC", help="use SRC's recipe for NEW")
    p.add_argument("--meta", type=Path, default=None, help="JSON merged into histograms_meta.json, e.g. a detector layout")
    p.add_argument("--poll", type=float, default=0.5, help="seconds between polls (default 0.5)")
    p.add_argument("--slice", type=float, default=10.0, help="histogram time slice, seconds of data time (default 10)")
    p.add_argument("--e-lo", type=float, default=0.0)
    p.add_argument("--e-hi", type=float, default=1200.0)
    p.add_argument("--bin", type=float, default=1.0, help="energy bin width, eV (default 1)")
    args = p.parse_args(argv)
    exit_with_parent()

    print(f"mass2-live-apply: following {args.input}", flush=True)
    try:
        run_live(
        args.recipe, args.input, args.output, args.hist_dir,
        spec=HistogramSpec(e_lo=args.e_lo, e_hi=args.e_hi, bin_width=args.bin, slice_s=args.slice),
        energy_col=args.energy_col, aliases=dict(args.alias), state_path=args.state_file, poll_s=args.poll,
        meta=None if args.meta is None else json.loads(args.meta.read_text()),
        )  # fmt: skip
    except KeyboardInterrupt:
        pass
    print("mass2-live-apply: done", flush=True)


if __name__ == "__main__":
    main()

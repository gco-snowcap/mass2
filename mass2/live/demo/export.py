"""Record the real pipeline on each demo dataset and write a standalone page that replays it.

For every dataset this runs the real simulator and the real `run_live` loop (with the saved recipe) over
one pass of the data, as fast as possible, keeping the ORIGINAL timestamps. Every time in the recording is
a real time of the original run. It records exactly what the live viewer reads: the histogram slices the
loop finished, with the records and chunks behind each, the state changes, and a refit of the dataset's
line every 30 slices. The page plays the recording at a chosen multiple of real time, with no server.

Command line:  mass2-live-export OUT.html [--datasets bessy_20240727 20230626 ...] [--workdir DIR]
"""

import argparse
import base64
import json
import tempfile
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

import polars as pl

from ..apply_recipe import run_live
from ..fit import LiveFitter
from ..arrow_stream import ArrowStreamTailer
from ..histogram import df_to_slices, slices_to_df, sparse_counts
from .datasets import DATASETS, DemoDataset
from .launcher import viewer_meta
from .simulate import load_experiment_states, load_ljh_sources, simulate, state_file_path

FIT_EVERY_SLICES = 30


def record(dataset: DemoDataset, workdir: Path) -> dict:
    """Run one pass of `dataset` through the real pipeline and return the recording for the page.

    The raw pulse stream is deleted once processed (it is by far the largest file); its size is kept.
    """
    stream, hist = workdir / "pulses.arrows", workdir / "hist"
    states = load_experiment_states(dataset.pulse_folder)
    simulate(load_ljh_sources(dataset.pulse_folder), stream, pace=False, scaled=dataset.scales, states=states)
    run_live(
        dataset.recipe_path, stream, workdir / "analyzed.arrows", hist, spec=dataset.spec, energy_col=dataset.energy_col,
        aliases={s.new_ch: s.source_ch for s in dataset.scales}, state_path=state_file_path(stream),
        meta=viewer_meta(dataset), poll_s=0, grace_s=0,
    )  # fmt: skip
    input_bytes = stream.stat().st_size
    stream.unlink()

    meta = json.loads((hist / "histograms_meta.json").read_text())
    status = json.loads((hist / "status.json").read_text())
    sparse, fits, t0_us = _read_slices_and_refit(dataset, hist / "histograms.arrows")
    records, chunks = _arrivals_per_slice(workdir / "analyzed.arrows", t0_us, dataset.spec.slice_us)
    n = max([*sparse, *records]) + 1  # every slice of data time, including ones with no good pulses in range
    nchunks = status["chunks"]
    return {
        "key": dataset.key,
        "title": dataset.title,
        "meta": meta | {
            "t0": t0_us / 1e6,  # absolute time (s) of offset 0 in this recording
            "bytes_per_chunk": {"input": input_bytes / nchunks, "output": status["output_bytes"] / nchunks},
        },
        "states": [[round(t - t0_us / 1e6, 3), label] for t, label in json.loads((hist / "states.json").read_text())],
        "slices": [[k * dataset.spec.slice_s, records.get(k, 0), chunks.get(k, 0), sparse.get(k, {})] for k in range(n)],
        "fits": fits,
    }  # fmt: skip


def _read_slices_and_refit(dataset: DemoDataset, path: Path) -> tuple[dict, dict, int]:
    """Read the finished slices batch by batch (a full-resolution copy of them all would not fit in memory),
    keeping each one sparse, and refit the line as mass2-live-fit does after every FIT_EVERY_SLICES slices."""
    spec, sparse, fitter, out, t0_us = dataset.spec, {}, LiveFitter(dataset.spec, dataset.roi), [], None
    reader = ArrowStreamTailer(path)
    while not reader.ended:
        for df in reader.poll():
            for s in df_to_slices(df, spec):
                t0_us = s.start_us if t0_us is None else t0_us
                sparse[(s.start_us - t0_us) // spec.slice_us] = sparse_counts(s.counts)
                fitter.add(slices_to_df([s]))
                if fitter.slices % FIT_EVERY_SLICES == 0 and (made := fitter.fit()) is not None:
                    entry, png = made
                    entry["t"] = round(entry["t"] - t0_us / 1e6, 3)  # seconds into the recording
                    entry["png"] = "data:image/png;base64," + base64.b64encode(png).decode()
                    out.append(entry)
    roi = dataset.roi
    fits = {"roi": {"label": roi.label, "source": roi.source, "dlo": roi.dlo, "dhi": roi.dhi},
            "every_s": FIT_EVERY_SLICES * spec.slice_s, "fits": out}  # fmt: skip
    assert t0_us is not None, "the recording has no histogram slices"
    return sparse, fits, t0_us


def _arrivals_per_slice(analyzed_path: Path, t0_us: int, slice_us: int) -> tuple[dict, dict]:
    """Records, and completed chunks, whose data time falls in each slice (by slice index)."""
    df = pl.read_ipc_stream(analyzed_path, columns=["timestamp", "chunk"])
    k = ((pl.col("timestamp").dt.epoch("us") - t0_us) // slice_us).alias("k")
    records = dict(df.group_by(k).len().iter_rows())
    chunks = dict(df.group_by("chunk").agg(k.max()).group_by("k").len().iter_rows())  # a chunk counts when complete
    return records, chunks


def write_replay_page(recordings: list[dict], out: Path) -> None:
    """The viewer page with `recordings` embedded; it plays them instead of asking a server."""
    page = resources.files("mass2.live.viewer").joinpath("viewer.html").read_text(encoding="utf-8")
    data = json.dumps({"datasets": recordings}, separators=(",", ":")).replace("</", "<\\/")
    marker = "<!-- replay data -->"
    assert marker in page
    out.write_text(page.replace(marker, f'<script id="replay">window.MASS2_REPLAY = {data};</script>'), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `mass2-live-export`."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, help="HTML file to write")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS), help="datasets to include (default: all)")
    p.add_argument("--workdir", type=Path, default=None, help="keep the recorded results here (default: a temporary directory)")
    args = p.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        workdir = args.workdir or Path(tmp)
        recordings = []
        for key in args.datasets:
            print(f"mass2-live-export: recording {key}", flush=True)
            recordings.append(record(DATASETS[key], workdir / key))
    write_replay_page(recordings, args.out)
    print(f"mass2-live-export: wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()

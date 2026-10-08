"""Tests for the core of mass2.live: what a real instrument would run.

1. Stream files are standard Arrow IPC, and can be followed while they are written.
2. Applying a saved recipe live gives exactly what mass2 gives offline.
3. Every good pulse lands in exactly one time slice, and output chunks match input chunks.
4. The live line fit finds a line the way Channel.linefit would.
"""

import json
import threading

import numpy as np
import polars as pl
import pulsedata
import pytest

import mass2
from mass2.live import arrow_stream, histogram, states
from mass2.live.apply_recipe import LiveRecipeApplier, run_live
from mass2.live.demo import simulate
from mass2.live.demo.datasets import DATASETS
from mass2.live.fit import LiveFitter
from mass2.live.histogram import NO_STATE, HistogramSpec

BESSY = DATASETS["bessy_20240727"]
PULSE_FOLDER = pulsedata.pulse_noise_ljh_pairs["bessy_20240727"].pulse_folder


@pytest.fixture(scope="module")
def bessy():
    """The first 2000 pulses per channel of the BESSY run, its saved recipes, and its experiment states."""
    sources = simulate.load_ljh_sources(PULSE_FOLDER, max_pulses=2000)
    return sources, mass2.misc.unpickle_object(BESSY.recipe_path), simulate.load_experiment_states(PULSE_FOLDER)


def test_1_stream_files_are_standard_and_can_be_read_while_growing(tmp_path):
    path = tmp_path / "s.arrows"
    batch = pl.DataFrame({"i": [1, 2, 3], "pulse": np.ones((3, 4), dtype=np.uint16)})
    reader = arrow_stream.ArrowStreamTailer(path)
    assert reader.poll() == []  # the file does not exist yet: nothing to read, no error

    writer = arrow_stream.ArrowStreamWriter(path)
    writer.write(batch)
    writer.write(batch)
    complete = path.read_bytes()
    path.write_bytes(complete[:-5])  # pretend we caught the writer halfway through its second batch
    assert len(reader.poll()) == 1  # only the complete batch is returned
    path.write_bytes(complete)
    assert len(reader.poll()) == 1  # and the rest arrives on the next poll
    writer.close()
    assert reader.poll() == [] and reader.ended

    assert len(pl.read_ipc_stream(path)) == 6  # any ordinary Arrow reader accepts the finished file


def test_2_live_recipe_matches_offline_mass2(tmp_path, bessy):
    """The core promise: chunk-by-chunk live results equal offline `Channels.load_recipes` results."""
    sources, recipes, state_log = bessy
    path = tmp_path / "pulses.arrows"
    simulate.simulate(sources, path, pace=False, states=state_log)
    labeled = states.StateFileFollower(simulate.state_file_path(path)).label(pl.read_ipc_stream(path))
    live = LiveRecipeApplier(recipes).process(labeled)

    offline = mass2.Channels.from_ljh_folder(PULSE_FOLDER).with_experiment_state_by_path().load_recipes(str(BESSY.recipe_path))
    for ch_num, ch in offline.channels.items():
        expect = ch.df.head(2000)
        found = live.filter(pl.col("ch_num") == ch_num).sort("subframecount")
        assert np.allclose(found["energy_5lagy_best"], expect["energy_5lagy_best"], equal_nan=True)
        assert (found["good"] == expect.select(ch.good_expr.alias("good"))["good"]).all()
        assert (found["state_label"] == expect["state_label"].cast(pl.String).fill_null(NO_STATE)).all()
        assert (found["timestamp"].dt.epoch("us") == expect["timestamp"].dt.epoch("us")).all()  # original timestamps


def test_3_every_good_pulse_is_histogrammed_once_and_chunks_match(tmp_path, bessy):
    sources, _, state_log = bessy
    stream, out, hist = tmp_path / "pulses.arrows", tmp_path / "out.arrows", tmp_path / "hist"
    spec = HistogramSpec(e_lo=0, e_hi=1200, bin_width=0.25, slice_s=10)
    writer = threading.Thread(target=simulate.simulate, args=(sources, stream), kwargs=dict(repeats=2, speed=400, states=state_log))
    writer.start()  # the loop follows the file while it is being written
    run_live(BESSY.recipe_path, stream, out, hist, spec=spec, state_path=simulate.state_file_path(stream), poll_s=0.05, grace_s=0.1)
    writer.join()

    raw, results = pl.read_ipc_stream(stream), pl.read_ipc_stream(out)
    assert results["chunk"].to_list() == raw["chunk"].to_list()  # same rows, same order, same chunk numbers
    assert results["ch_num"].to_list() == raw["ch_num"].to_list()

    slices = histogram.df_to_slices(pl.read_ipc_stream(hist / "histograms.arrows"), spec)
    good = results.filter(pl.col("good"), pl.col("energy_5lagy_best").is_between(0, 1200, closed="left"))
    assert sum(s.total() for s in slices) == len(good)
    status = json.loads((hist / "status.json").read_text())
    assert status["records"] == len(raw) and status["good_pulses"] == len(good) and status["stream_ended"]


def test_4_line_fit_finds_the_line():
    """Given a histogram with a 6 eV wide line at 600 eV, the live fitter fits it like Channel.linefit would."""
    spec = HistogramSpec(e_lo=0, e_hi=1200, bin_width=0.25, slice_s=10)
    energies = np.random.default_rng(0).normal(600, 6 / 2.355, 20000)
    counts = np.histogram(energies, bins=spec.nbins, range=(0, 1200))[0]
    fitter = LiveFitter(spec, BESSY.roi)
    fitter.add(histogram.slices_to_df([histogram.HistogramSlice(0, spec.nbins, {(1, "CAL2"): counts})]))
    entry, png = fitter.fit()
    assert entry["peak"][0] == pytest.approx(600, abs=0.1)
    assert entry["fwhm"][0] == pytest.approx(6, rel=0.05)
    assert png.startswith(b"\x89PNG")  # drawn by LineModelResult.plotm()

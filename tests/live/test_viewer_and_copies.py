"""Tests for what the viewer is sent, the shared state selection, and the demo's gain-shifted copies.

8. A copy recorded at a different gain comes out at its source channel's energies, through its own recipe.
9. A page starting from a base sees exactly what a page that followed every slice sees.
10. The histograms survive the trip through Arrow IPC.
11. The line fit follows the states selected in the viewer, and the server checks what it is sent.
12. The demo refuses playback speeds outside the viewer's range.
13. The page is told the resident memory of each process involved.
14. Each visitor gets a run of their own, up to a limit, and a run nobody views is ended.
15. A run whose tool fails starts over, and a tool exits when its demo is gone.
"""

import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import polars as pl
import pulsedata
import pytest

import mass2
from mass2.live import arrow_stream, histogram, states
from mass2.live.apply_recipe import LiveRecipeApplier
from mass2.live.demo import launcher, simulate
from mass2.live.demo.datasets import DATASETS
from mass2.live.fit import LiveFitter, read_selected_states
from mass2.live.histogram import HistogramSlice, HistogramSpec
from mass2.live.viewer import server

MN = DATASETS["20230626"]


def test_8_a_copy_at_another_gain_matches_its_source_through_its_own_recipe(tmp_path):
    copy = next(p for p in MN.pixels if p.source_ch is not None and p.gain != 1.0)
    sources = simulate.load_ljh_sources(MN.pulse_folder, max_pulses=1500)
    path = tmp_path / "pulses.arrows"
    simulate.simulate(sources, path, pace=False, scaled=[simulate.ScaledChannel(copy.source_ch, copy.ch_num, copy.gain)])
    raw = states.StateFileFollower(None).label(pl.read_ipc_stream(path))
    recipes = mass2.misc.unpickle_object(MN.recipe_path)
    assert copy.ch_num in recipes  # learned from the copy's own scaled pulses

    def energies(applier: LiveRecipeApplier, ch: int) -> np.ndarray:
        out = applier.process(raw).filter(pl.col("ch_num") == ch).sort("subframecount")
        return out[MN.energy_col].to_numpy()

    source = energies(LiveRecipeApplier(recipes), copy.source_ch)
    own = energies(LiveRecipeApplier(recipes, {copy.ch_num: copy.source_ch}), copy.ch_num)  # its own recipe wins over the alias
    borrowed = energies(LiveRecipeApplier({copy.source_ch: recipes[copy.source_ch]}, {copy.ch_num: copy.source_ch}), copy.ch_num)
    near_mn = (source > 5800) & (source < 6000)
    assert near_mn.sum() > 100
    assert np.median(np.abs(own[near_mn] - source[near_mn])) < 1.0  # eV
    assert abs(np.median(borrowed[near_mn] / source[near_mn]) - 1) > 0.01  # the source's recipe would be off by about the gain


def _store_with_slices(tmp_path, spec: HistogramSpec, n: int) -> tuple[server.HistogramStore, list[HistogramSlice]]:
    """A histogram directory holding `n` random slices in two states and three channels, and a store following it."""
    rng = np.random.default_rng(1)
    hist = tmp_path / "hist"
    hist.mkdir()
    (hist / "histograms_meta.json").write_text(json.dumps(spec.to_dict()))
    slices = []
    for k in range(n):
        state = "CAL" if (k // 7) % 2 else "SCAN"
        counts = {(ch, state): rng.poisson(0.05, spec.nbins).astype(np.int64) for ch in (1, 2, 3)}
        slices.append(HistogramSlice(1_700_000_000_000_000 + k * spec.slice_us, spec.nbins, counts))
    with arrow_stream.ArrowStreamWriter(hist / "histograms.arrows") as w:
        for k in range(0, n, 5):
            w.write(histogram.slices_to_df(slices[k : k + 5]))
    return server.HistogramStore(hist), slices


def _decode(rows: pl.DataFrame, nbins: int) -> dict[tuple, np.ndarray]:
    out = {}
    for r in rows.iter_rows(named=True):
        a = np.zeros(nbins, np.int64)
        np.add.at(a, np.asarray(r["bins"], dtype=np.int64), np.asarray(r["counts"], dtype=np.int64))
        key = (r["kind"], r["t"], r["state"], r["ch_num"])
        out[key] = out.get(key, 0) + a
    return out


@pytest.mark.parametrize("row_slices", [1, 4, 25])
def test_9_a_base_shows_what_following_every_slice_shows(tmp_path, row_slices):
    spec = HistogramSpec(e_lo=0, e_hi=200, bin_width=0.5, slice_s=10)
    store, slices = _store_with_slices(tmp_path, spec, 203)
    info, rows = store.state(0, row_s=row_slices * spec.slice_s)
    assert info["base"] and info["n_slices"] == 203 and info["row_s"] == row_slices * spec.slice_s
    got = _decode(pl.from_arrow(server.from_arrow_ipc(server.to_arrow_ipc(info, rows))[1]), spec.nbins)

    for (ch, state) in {k for s in slices for k in s.counts}:  # totals: every slice, by channel and state
        assert np.array_equal(got[("total", None, state, ch)], sum(s.counts.get((ch, state), 0) for s in slices))
    row_us = row_slices * spec.slice_us
    by_row: dict[tuple, np.ndarray] = {}
    for s in slices:
        for (_, state), c in s.counts.items():
            key = (s.start_us // row_us, state)
            by_row[key] = by_row.get(key, 0) + c
    open_row = (slices[-1].start_us + spec.slice_us) // row_us
    for (r, state), c in by_row.items():  # complete rows summed over channels; the open row as its slices
        if open_row - server.ROWS <= r < open_row:
            assert np.array_equal(got[("row", r * row_us / 1e6, state, 0)], c)
        elif r < open_row - server.ROWS:
            assert ("row", r * row_us / 1e6, state, 0) not in got  # beyond what the page shows: never sent
    for s in slices:
        if s.start_us // row_us >= open_row:
            for (_, state), c in s.counts.items():
                assert ("slice", s.start_us / 1e6, state, 0) in got

    info, rows = store.state(200, row_s=row_slices * spec.slice_s)  # a page keeping up: the newest slices, by channel
    assert not info["base"]
    got = _decode(pl.from_arrow(server.from_arrow_ipc(server.to_arrow_ipc(info, rows))[1]), spec.nbins)
    assert set(got) == {("slice", s.start_us / 1e6, st, ch) for s in slices[200:] for ch, st in s.counts}
    for s in slices[200:]:
        for (ch, state), c in s.counts.items():
            assert np.array_equal(got[("slice", s.start_us / 1e6, state, ch)], c)


def test_10_histograms_survive_arrow_ipc():
    rows = [
        server.CountRow.dense("total", None, "CAL", 4219, np.array([0, 3, 0, 70000, 1])),
        server.CountRow.dense("slice", 1722087040.0, "SCAN3", 0, np.zeros(5, np.int64)),
    ]
    info = {"run": 3, "status": {"records": 12}, "selected_states": ["CAL"]}
    back_info, table = server.from_arrow_ipc(server.to_arrow_ipc(info, rows))
    assert back_info == info
    t = pl.from_arrow(table)
    assert t["kind"].cast(pl.String).to_list() == ["total", "slice"] and t["t"].to_list() == [None, 1722087040.0]
    assert t["bins"].to_list() == [[1, 3, 4], []] and t["counts"].to_list() == [[3, 70000, 1], []]
    assert table.schema.field("bins").type.value_type.bit_width == 16


def test_11_the_fit_follows_the_selected_states(tmp_path):
    spec = HistogramSpec(e_lo=0, e_hi=1200, bin_width=0.25, slice_s=10)
    rng = np.random.default_rng(0)
    line = lambda e0: np.histogram(rng.normal(e0, 6 / 2.355, 20000), bins=spec.nbins, range=(0, 1200))[0]  # noqa: E731
    fitter = LiveFitter(spec, DATASETS["bessy_20240727"].roi)
    fitter.add(histogram.slices_to_df([HistogramSlice(0, spec.nbins, {(1, "CAL2"): line(600), (1, "SCAN3"): line(603)})]))
    fitter.fit()
    assert fitter.select(["CAL2"]) and fitter.fits == []  # a new selection starts a new series of fits
    entry, _ = fitter.fit()
    assert entry["states"] == ["CAL2"] and entry["peak"][0] == pytest.approx(600, abs=0.1)
    assert not fitter.select(["CAL2"])
    fitter.select(["SCAN3"])
    assert fitter.fit()[0]["peak"][0] == pytest.approx(603, abs=0.1)

    store = server.HistogramStore(tmp_path / "hist")
    store.select_states(["SCAN3", "CAL2"])
    assert read_selected_states(tmp_path / "hist") == ["CAL2", "SCAN3"] and store.state()[0]["selected_states"] == ["CAL2", "SCAN3"]
    store.select_states(None)
    assert read_selected_states(tmp_path / "hist") is None
    for bad in ["CAL2", [1, 2], {"a": 1}, ["x" * 65], ["s"] * 65]:
        with pytest.raises(ValueError):
            store.select_states(bad)


def test_12_the_demo_refuses_speeds_outside_the_viewers_range(tmp_path):
    controller = launcher.DemoController(tmp_path, server.HistogramStore(tmp_path / "hist"))
    for ok in [1, 5, 600]:
        controller.set_speed(ok)
        assert controller.speed == ok
    for bad in [0, 0.5, -5, 601, 1e308, math.inf, math.nan]:
        with pytest.raises(ValueError):
            controller.set_speed(bad)
    assert controller.speed == 600


def test_13_the_page_is_told_each_process_memory():
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    child = subprocess.Popen([sys.executable, "-c", "import time; x = bytearray(80_000_000); time.sleep(30)"])
    try:
        for _ in range(50):  # until the child has allocated its 80 MB
            mem = {p["name"]: p for p in server.process_memory({"viewer": os.getpid(), "child": child.pid, "gone": gone.pid})}
            if mem.get("child", {}).get("rss_bytes", 0) > 80e6:
                break
            time.sleep(0.1)
        assert set(mem) == {"viewer", "child"} and mem["child"]["pid"] == child.pid  # a process no longer running is left out
        assert mem["child"]["rss_bytes"] > 80e6 and mem["viewer"]["rss_bytes"] > 10e6
    finally:
        child.kill()


class _FakeController:
    """Stands in for DemoController: records what it is asked to do, starts no processes."""

    def __init__(self, run_dir, store):
        self.run_dir, self.store, self.speed, self.active, self.stopped, self.maintained = run_dir, store, 5.0, None, False, 0

    def switch(self, key):
        self.active = key

    def describe(self):
        return {"active": self.active}

    def pids(self):
        return {}

    def maintain(self, max_gb):
        self.maintained += 1

    def stop(self):
        self.stopped = True


def test_14_each_visitor_gets_a_run_of_their_own(tmp_path):
    from mass2.live.demo.visitors import VisitorRuns

    runs = VisitorRuns(tmp_path, max_runs=2, dataset="bessy_20240727", idle_s=60, new_controller=_FakeController)
    a, b = runs.route("/").redirect, runs.route("/").redirect
    assert a != b and a.startswith("/r/") and a.endswith("/")
    assert runs.route("/").busy  # a third visitor waits
    site_a, site_b = runs.route(a + "api/state").site, runs.route(b).site
    assert site_a is not site_b and site_a.controller.active == "bessy_20240727"
    assert runs.route(a + "api/state").path == "/api/state" and runs.route(a + "arrow.js").path == "/arrow.js"
    assert runs.route(a.rstrip("/")).redirect == a and runs.route("/r/nothex/").site is None

    runs.runs[b.split("/")[2]].last_seen -= 120  # b's page has been closed for two minutes
    runs.tick(max_gb=4)
    assert site_b.controller.stopped and not site_a.controller.stopped and site_a.controller.maintained == 1
    assert runs.route("/").redirect  # a slot is free again
    revived = runs.route(b + "api/state").site  # b's old address now starts a new run (or waits when all are in use)
    assert revived is None or revived is not site_b
    runs.stop_all()
    assert site_a.controller.stopped and runs.runs == {}


def test_15_a_failed_tool_starts_the_run_over_and_tools_exit_with_their_demo(tmp_path, monkeypatch):
    controller = launcher.DemoController(tmp_path, server.HistogramStore(tmp_path / "hist"))
    starts = []
    monkeypatch.setattr(controller, "switch", lambda key: starts.append(key))
    controller.active = "20230626"
    monkeypatch.setattr(controller, "failed", lambda: ["mass2-live-apply"])
    for _ in range(5):
        controller.maintain(max_gb=4)
    assert starts == ["20230626"] * 3  # three quick restarts, then at most one a minute

    script = "import os, sys, time; from mass2.live.parent import exit_with_parent; exit_with_parent(0.05); time.sleep(30)"
    parent = subprocess.Popen(  # a stand-in demo that starts one tool and is then killed
        [sys.executable, "-c", f"import os, subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', {script!r}], env=os.environ | {{'MASS2_LIVE_PARENT_PID': str(os.getpid())}}); print(p.pid, flush=True); time.sleep(30)"],
        stdout=subprocess.PIPE, text=True,
    )
    child = int(parent.stdout.readline())
    parent.kill()
    parent.wait()
    for _ in range(100):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(child, 9)
        raise AssertionError("the tool outlived its demo")

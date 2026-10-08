"""Tests for the viewer server: what a page is sent.

7. A page is sent exactly what it draws: the run's totals and the newest time-plot rows.
8. The histograms survive the trip through Arrow IPC.
9. The server keeps the selected states for the fit, and refuses a bad selection.
10. The page is told the resident memory of each process involved.
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import polars as pl
import pytest

from mass2.live import arrow_stream, histogram
from mass2.live.fit import read_selected_states
from mass2.live.histogram import HistogramSlice, HistogramSpec
from mass2.live.viewer import server


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
def test_7_the_page_is_sent_exactly_what_it_draws(tmp_path, row_slices):
    spec = HistogramSpec(e_lo=0, e_hi=200, bin_width=0.5, slice_s=10)
    store, slices = _store_with_slices(tmp_path, spec, 203)
    opening = HistogramSlice(slices[-1].start_us + spec.slice_us, spec.nbins, {(2, "CAL"): np.arange(spec.nbins)})  # still filling
    arrow_stream.write_stream_atomically(histogram.slices_to_df([opening]), tmp_path / "hist" / "histograms_current.arrows")
    every = [*slices, opening]
    info, rows = store.state(row_s=row_slices * spec.slice_s)
    assert info["n_slices"] == 203 and info["row_s"] == row_slices * spec.slice_s and info["open_good"] == opening.total()
    got = _decode(pl.from_arrow(server.from_arrow_ipc(server.to_arrow_ipc(info, rows))[1]), spec.nbins)

    for ch, state in {k for s in every for k in s.counts}:  # totals: the whole run, by channel and state
        assert np.array_equal(got[("total", None, state, ch)], sum(s.counts.get((ch, state), 0) for s in every))
    row_us = row_slices * spec.slice_us
    by_row: dict[tuple, np.ndarray] = {}
    for s in every:
        for (_, state), c in s.counts.items():
            by_row[(s.start_us // row_us, state)] = by_row.get((s.start_us // row_us, state), 0) + c
    newest = opening.start_us // row_us
    for (r, state), c in by_row.items():  # the newest rows, summed over channels; older ones are never sent
        key = ("row", r * row_us / 1e6, state, 0)
        assert np.array_equal(got[key], c) if r > newest - server.ROWS else key not in got


def test_8_histograms_survive_arrow_ipc():
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


def test_9_the_server_keeps_the_selected_states_and_refuses_bad_ones(tmp_path):
    store = server.HistogramStore(tmp_path / "hist")
    store.select_states(["SCAN3", "CAL2"])
    assert read_selected_states(tmp_path / "hist") == ["CAL2", "SCAN3"] and store.state()[0]["selected_states"] == ["CAL2", "SCAN3"]
    store.select_states(None)
    assert read_selected_states(tmp_path / "hist") is None
    for bad in ["CAL2", [1, 2], {"a": 1}, ["x" * 65], ["s"] * 65]:
        with pytest.raises(ValueError):
            store.select_states(bad)


def test_10_the_page_is_told_each_process_memory():
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

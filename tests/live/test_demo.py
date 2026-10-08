"""Tests for the demo environment and the viewer, which sit on top of the core.

5. The simulator writes 100-record chunks with all channels and their original timestamps, replays the
   experiment states, scales copies, and follows a speed change while running.
6. The viewer serves what the tools wrote; the demo switches datasets and changes speed from the page.
7. The shareable page embeds an exact recording of the real pipeline.
"""

import json
import threading
import time
import urllib.request

import numpy as np
import polars as pl
import pulsedata
import pytest

from mass2.live.demo import export, launcher, simulate
from mass2.live.demo.datasets import DATASETS
from mass2.live.viewer import server

PULSE_FOLDER = pulsedata.pulse_noise_ljh_pairs["bessy_20240727"].pulse_folder


def test_5_simulator_chunks_states_copies_and_speed_changes(tmp_path):
    sources = simulate.load_ljh_sources(PULSE_FOLDER, max_pulses=2000)
    path, speed_file = tmp_path / "pulses.arrows", tmp_path / "speed.json"
    speed_file.write_text(json.dumps({"speed": 1}))  # real time: 2000 pulses would take minutes...
    run = threading.Thread(
        target=simulate.simulate, args=(sources, path),
        kwargs=dict(scaled=[simulate.ScaledChannel(4219, 99, 1.05)], speed_file=speed_file,
                    states=simulate.load_experiment_states(PULSE_FOLDER)),
    )  # fmt: skip
    run.start()
    time.sleep(1.0)
    speed_file.write_text(json.dumps({"speed": 1000}))  # ...so speed it up while it runs
    run.join(timeout=60)
    assert not run.is_alive()

    df = pl.read_ipc_stream(path)
    assert len(df) == 3 * 2000 and set(df["ch_num"]) == {4219, 4220, 99}
    assert (df.group_by("chunk").len()["len"] == 100).all()
    first = sources[4219].timestamp_us.min()
    assert df.filter(pl.col("ch_num") == 4219)["timestamp"].dt.epoch("us").min() == first  # original timestamps
    assert "CAL2" in simulate.state_file_path(path).read_text()
    original = np.stack(df.filter(pl.col("ch_num") == 4219)["pulse"].to_numpy()).astype(float)
    copy = np.stack(df.filter(pl.col("ch_num") == 99)["pulse"].to_numpy()).astype(float)
    height = lambda p: p.max(axis=1) - p[:, :200].mean(axis=1)  # noqa: E731
    assert np.median(height(copy) / height(original)) == pytest.approx(1.05, abs=0.002)


def test_6_viewer_and_demo_switch_dataset_and_speed(tmp_path):
    store = server.HistogramStore(tmp_path / "bessy_20240727" / "hist")
    controller = launcher.DemoController(tmp_path, store, sim_extra=["--max-pulses", "3000"])
    srv, port = server.start_server(store, port=0, controller=controller)
    url = f"http://127.0.0.1:{port}"

    def post(path, body):
        urllib.request.urlopen(urllib.request.Request(url + path, data=json.dumps(body).encode(), method="POST"))

    def state_when(ok, timeout=120):
        t0 = time.time()
        while time.time() - t0 < timeout:
            s, _ = server.from_arrow_ipc(urllib.request.urlopen(f"{url}/api/state?since=0").read())
            if ok(s):
                return s
            assert not controller.failed()
            time.sleep(0.5)
        raise AssertionError("timed out")

    try:
        controller.switch("bessy_20240727")
        post("/api/speed", {"speed": 60})
        assert json.loads((tmp_path / "bessy_20240727" / "speed.json").read_text()) == {"speed": 60.0}
        s = state_when(lambda s: s["status"] and s["status"]["records"] > 2000)
        assert len(s["channels"]) == 16 and s["meta"]["layout"]["4224"]["gain"] == 1.03
        assert s["status"]["channels_without_recipe"] == []  # every simulated copy borrows a real recipe
        assert s["controller"]["speed"] == 60
        assert "<canvas" in urllib.request.urlopen(url + "/").read().decode()

        post("/api/dataset", {"key": "20230626"})
        s = state_when(lambda s: s["meta"] and s["meta"]["e_hi"] == 10000 and s["status"] and s["status"]["records"] > 0)
        assert s["controller"]["active"] == "20230626"
    finally:
        controller.stop()
        srv.shutdown()


def test_7_shareable_page_holds_an_exact_recording(tmp_path):
    dataset = DATASETS["gamma_20241005"]  # the smallest dataset
    rec = export.record(dataset, tmp_path / "gamma")
    results = pl.read_ipc_stream(tmp_path / "gamma" / "analyzed.arrows")
    good = results.filter(pl.col("good"), pl.col(dataset.energy_col).is_between(dataset.spec.e_lo, dataset.spec.e_hi, closed="left"))
    recorded = sum(sum(pairs[1::2]) for *_, sparse in rec["slices"] for chans in sparse.values() for pairs in chans.values())
    assert recorded == len(good)
    assert sum(r for _, r, _, _ in rec["slices"]) == len(results)
    assert rec["fits"]["fits"] and rec["fits"]["fits"][-1]["png"].startswith("data:image/png;base64,")

    page = tmp_path / "replay.html"
    export.write_replay_page([rec], page)
    embedded = page.read_text().split("window.MASS2_REPLAY = ", 1)[1].split(";</script>", 1)[0]
    assert json.loads(embedded)["datasets"][0]["key"] == "gamma_20241005"

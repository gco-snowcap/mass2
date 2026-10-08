"""Run the whole live pipeline on `pulsedata` datasets, a run for each visitor, switchable from the viewer.

Each visitor gets a run of their own (see `visitors.py`). For the run's dataset (see `datasets.py`):
1. Start `mass2-live-sim`, replaying the LJH data (and its experiment states) into a growing stream, plus
   gain-shifted fake channels.
2. Start `mass2-live-apply` on that stream with the dataset's saved recipe (mass2/live/recipes/<key>.pkl).
   Fake channels at a different gain have recipes of their own in it; exact copies borrow their source channel's.
3. Start `mass2-live-fit`, refitting the line the original analysis fitted, on all channels summed.
The viewer is served from this process. Picking another dataset in the page stops the run's tools and starts
them on it. The data are replayed again and again, continuing the timeline, so a run goes on across passes
and a change of playback speed never interrupts it; only when its files reach `--max-gb` does it start over.

Each pipeline tool runs as its own process, exactly as it would from the command line, and exits if the demo
is gone. A run whose tool fails is started over. Ctrl-C stops them all.

Command line:  mass2-live-demo [WORKDIR] [--dataset bessy_20240727] [--speed 5] [--max-gb 4] [--max-runs 8]
                               [--idle 180] [--lan] [--port 8765] [--public]

Everything is local by default. With --public it is also served on the internet through a Cloudflare quick
tunnel, if cloudflared is installed, and the public address is printed.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Sequence
from typing import IO
from pathlib import Path

from .datasets import DATASETS, DemoDataset
from .simulate import state_file_path
from ..fit import read_selected_states, write_selected_states
from ..parent import PARENT_ENV
from ..viewer.server import HistogramStore, start_router_server, viewer_urls
from .visitors import VisitorRuns

MIN_SPEED, MAX_SPEED = 1.0, 600.0  # the viewer's playback-speed slider runs 1x to 600x real time


class DemoController:
    """Owns the simulator and applier processes for one dataset at a time."""

    def __init__(self, workdir: Path, store: HistogramStore, repeats: int = 0, sim_extra: Sequence[str] = (), speed: float = 5.0):
        self.workdir = workdir
        self.store = store
        self.repeats = repeats
        self.sim_extra = list(sim_extra)  # e.g. ["--max-pulses", "2000"] in tests
        self.active: str | None = None
        self.speed = speed  # playback speed, multiples of real time; the simulator re-reads it before every chunk
        self.phase = "starting"
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._restarts: list[float] = []  # times the run was restarted after a failure
        self._checked = 0.0

    def describe(self) -> dict:
        return {
            "datasets": [{"key": d.key, "title": d.title} for d in DATASETS.values()],
            "active": self.active,
            "phase": self.phase,
            "speed": self.speed,
            "source_bytes": source_bytes(DATASETS[self.active]) if self.active else {},
        }

    def set_speed(self, speed: float) -> None:
        """Change the playback speed of the running simulator (it picks this up before its next chunk).
        Raises ValueError outside the viewer's own range, so no request can make the simulator write without pause."""
        if not MIN_SPEED <= float(speed) <= MAX_SPEED:
            raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
        self.speed = float(speed)
        if self.active is not None:
            self._write_speed(self.workdir / self.active)

    def _write_speed(self, run_dir: Path) -> None:
        tmp = run_dir / "speed.json.tmp"
        tmp.write_text(json.dumps({"speed": self.speed}))
        tmp.replace(run_dir / "speed.json")

    def switch(self, key: str) -> None:
        """Stop the current pipeline and start `key`'s. Raises KeyError if unknown. Starting the same dataset
        over keeps the viewer's selection of states."""
        dataset = DATASETS[key]
        with self._lock:
            self._stop_procs()
            run_dir = self.workdir / key
            keep = read_selected_states(run_dir / "hist") if key == self.active else None
            self.active = key
            run_dir.mkdir(parents=True, exist_ok=True)
            for stale in [run_dir / "pulses.arrows", state_file_path(run_dir / "pulses.arrows"), *(run_dir / "hist").rglob("*.*")]:
                stale.unlink(missing_ok=True)  # never let the applier or viewer pick up a previous run
            self.store.reset(run_dir / "hist")
            if keep is not None:
                write_selected_states(run_dir / "hist", keep)
            self._write_speed(run_dir)
            self._procs = pipeline_processes(dataset, run_dir, self.repeats, self.sim_extra)
            self.phase = "running"

    def pass_done(self) -> bool:
        """True once mass2-live-apply has finished the stream: the simulator's passes over the data are all written."""
        with self._lock:
            proc = self._procs.get("mass2-live-apply")
            return proc is not None and proc.poll() == 0

    def pids(self) -> dict[str, int]:
        """The running pipeline processes, by name."""
        with self._lock:
            return {name: proc.pid for name, proc in self._procs.items() if proc.poll() is None}

    def run_bytes(self) -> int:
        """Bytes in the running dataset's files."""
        if self.active is None:
            return 0
        return sum(p.stat().st_size for p in (self.workdir / self.active).rglob("*") if p.is_file())

    def maintain(self, max_gb: float) -> None:
        """Keep the run going; call it often. Starts the run over when a tool has failed (at most 3 times in 5
        minutes, then once a minute), when every pass asked for is done, or when its files reach `max_gb`."""
        if self.active is None:
            return
        now = time.time()
        if failed := self.failed():
            self._restarts = [t for t in self._restarts if now - t < 300]
            if len(self._restarts) >= 3 and now - self._restarts[-1] < 60:
                return
            print(f"mass2-live-demo: {', '.join(failed)} failed; starting the run over", flush=True)
            self._restarts.append(now)
            self.switch(self.active)
        elif self.pass_done():
            self.switch(self.active)  # every pass asked for is done: start over
        elif now - self._checked > 10:
            self._checked = now
            if self.run_bytes() > max_gb * 1e9:
                self.switch(self.active)  # start over, so the run's files never outgrow max_gb

    def failed(self) -> list[str]:
        """Names of pipeline tools that exited with an error."""
        with self._lock:
            return [name for name, proc in self._procs.items() if proc.poll() not in {None, 0}]

    def stop(self) -> None:
        with self._lock:
            self._stop_procs()

    def _stop_procs(self) -> None:
        for proc in self._procs.values():
            proc.terminate()
        for proc in self._procs.values():
            proc.wait()
        self._procs = {}


def source_bytes(dataset: DemoDataset) -> dict[str, int]:
    """Sizes of the files a run starts from: the LJH files and the saved recipe."""
    files = [*Path(dataset.pulse_folder).glob("*_chan*.ljh"), dataset.recipe_path]
    return {p.name: p.stat().st_size for p in files}


def viewer_meta(dataset: DemoDataset) -> dict:
    """What the viewer shows about a demo dataset, passed through mass2-live-apply's --meta."""
    return {
        "layout": {str(ch): pos for ch, pos in dataset.layout.items()},
        "aliases": {str(s.new_ch): s.source_ch for s in dataset.scales},
        "bin_source": dataset.bin_source,
        "ljh_files": sorted(p.name for p in Path(dataset.pulse_folder).glob("*_chan*.ljh")),
        "files": {"recipe": str(dataset.recipe_path.name), "input": "pulses.arrows", "states": "pulses_experiment_state.txt",
                  "output": "analyzed.arrows", "hist_dir": "hist/"},
    }  # fmt: skip


def pipeline_processes(dataset: DemoDataset, run_dir: Path, repeats: int, sim_extra: Sequence[str]) -> dict[str, subprocess.Popen]:
    """Start the applier, the line fitter and the simulator for `dataset`, all writing under `run_dir`."""
    stream, hist, spec, roi = run_dir / "pulses.arrows", run_dir / "hist", dataset.spec, dataset.roi
    (run_dir / "viewer_meta.json").write_text(json.dumps(viewer_meta(dataset)))
    apply = [sys.executable, "-m", "mass2.live.apply_recipe", str(dataset.recipe_path), str(stream), str(run_dir / "analyzed.arrows")]
    apply += [str(hist), "--state-file", str(state_file_path(stream)), "--energy-col", dataset.energy_col]
    apply += ["--slice", str(spec.slice_s), "--e-lo", str(spec.e_lo), "--e-hi", str(spec.e_hi), "--bin", str(spec.bin_width)]
    apply += ["--meta", str(run_dir / "viewer_meta.json")]
    sim = [sys.executable, "-m", "mass2.live.demo.simulate", str(stream), "--ljh-folder", str(dataset.pulse_folder)]
    sim += ["--speed-file", str(run_dir / "speed.json"), "--repeats", str(repeats), *sim_extra]
    for s in dataset.scales:
        apply += ["--alias", f"{s.new_ch}={s.source_ch}"]
        sim += ["--scale", f"{s.source_ch}:{s.new_ch}:{s.factor}"]
    fit = [sys.executable, "-m", "mass2.live.fit", str(hist), "--line", str(roi.line), "--dlo", str(roi.dlo), "--dhi", str(roi.dhi)]
    fit += ["--source", roi.source]
    env = os.environ | {PARENT_ENV: str(os.getpid())}  # each tool exits if this process is gone
    return {
        "mass2-live-apply": subprocess.Popen(apply, env=env),
        "mass2-live-sim": subprocess.Popen(sim, env=env),
        "mass2-live-fit": subprocess.Popen(fit, env=env),
    }


def public_tunnel(port: int) -> subprocess.Popen:
    """Start a Cloudflare quick tunnel to the local viewer and print its public address (a new one each time)."""
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", f"http://127.0.0.1:{port}"],
        stderr=subprocess.PIPE,
        text=True,
    )

    def follow(stream: IO[str]) -> None:  # print the address once, and keep reading so cloudflared never blocks
        shown = False
        for line in stream:
            if not shown and (m := re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)):
                print(f"mass2-live-demo: public address {m.group(0)}", flush=True)
                shown = True

    assert proc.stderr is not None
    threading.Thread(target=follow, args=(proc.stderr,), daemon=True).start()
    return proc


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `mass2-live-demo`."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "workdir", type=Path, nargs="?", default=Path("mass2_live_demo"), help="output directory (default ./mass2_live_demo)"
    )
    p.add_argument("--dataset", choices=sorted(DATASETS), default="bessy_20240727", help="dataset a new run starts with")
    p.add_argument("--speed", type=float, default=5.0, help="playback speed a new run starts at, multiples of real time (default 5)")
    p.add_argument("--repeats", type=int, default=0, help="passes over the data in a run before it starts over (default 0: no end)")
    p.add_argument("--max-gb", type=float, default=4.0, help="start a run over when its files reach this size, GB (default 4)")
    p.add_argument("--max-runs", type=int, default=8, help="runs at once, one per visitor (default 8)")
    p.add_argument("--idle", type=float, default=180, help="stop a run nobody has viewed for this long, seconds (default 180)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--lan", action="store_true", help="serve the viewer to other devices on this network, e.g. a phone")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument(
        "--public", action="store_true", help="also serve it on the internet: a Cloudflare quick tunnel (needs cloudflared)"
    )
    args = p.parse_args(argv)
    if args.public and shutil.which("cloudflared") is None:
        p.error(
            "--public needs cloudflared on the PATH (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)"
        )

    args.workdir.mkdir(parents=True, exist_ok=True)
    runs = VisitorRuns(
        args.workdir, lambda run_dir, store: DemoController(run_dir, store, args.repeats, speed=args.speed),
        max_runs=args.max_runs, dataset=args.dataset, idle_s=args.idle,
    )  # fmt: skip
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    server, port = start_router_server(runs, args.port, host)
    urls = viewer_urls(host, port)
    print(f"mass2-live-demo: viewer at {'  '.join(urls)}  (Ctrl-C to stop)", flush=True)
    tunnel = public_tunnel(port) if args.public else None
    if not args.no_browser:
        webbrowser.open(urls[-1])

    # SIGTERM (e.g. from `kill` or `timeout`) must clean up the children exactly as Ctrl-C does.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        while True:
            runs.tick(args.max_gb)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if tunnel is not None:
            tunnel.terminate()
        runs.stop_all()
        server.shutdown()


if __name__ == "__main__":
    main()

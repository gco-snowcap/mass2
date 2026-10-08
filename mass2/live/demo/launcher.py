"""Run the whole live pipeline on `pulsedata` datasets, switchable from the viewer.

For the chosen dataset (see `datasets.py`):
1. Start `mass2-live-sim`, replaying the LJH data (and its experiment states) into a growing stream, plus
   gain-shifted fake channels.
2. Start `mass2-live-apply` on that stream with the dataset's saved recipe (mass2/live/recipes/<key>.pkl).
   Fake channels at a different gain have recipes of their own in it; exact copies borrow their source channel's.
3. Start `mass2-live-fit`, refitting the line the original analysis fitted, on all channels summed.
The viewer is served from this process. Picking another dataset in the page stops 1 and 2 and restarts them.
The data are replayed again and again, continuing the timeline, so the run goes on across passes and a change
of playback speed never interrupts it. Only when the run's files reach `--max-gb` does it start over from
empty files, so the disk never holds more than that.

Each pipeline tool runs as its own process, exactly as it would from the command line. Ctrl-C stops them all.

Command line:  mass2-live-demo [WORKDIR] [--dataset bessy_20240727] [--speed 5] [--max-gb 20] [--lan] [--port 8765]
"""

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Sequence
from pathlib import Path

from .datasets import DATASETS, DemoDataset
from .simulate import state_file_path
from ..fit import read_selected_states, write_selected_states
from ..viewer.server import HistogramStore, start_server, viewer_urls

MIN_SPEED, MAX_SPEED = 1.0, 600.0  # the viewer's playback-speed slider runs 1x to 600x real time


class DemoController:
    """Owns the simulator and applier processes for one dataset at a time."""

    def __init__(self, workdir: Path, store: HistogramStore, repeats: int = 0, sim_extra: Sequence[str] = ()):
        self.workdir = workdir
        self.store = store
        self.repeats = repeats
        self.sim_extra = list(sim_extra)  # e.g. ["--max-pulses", "2000"] in tests
        self.active: str | None = None
        self.speed = 5.0  # playback speed, multiples of real time; the simulator re-reads it before every chunk
        self.phase = "starting"
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def describe(self) -> dict:
        return {
            "datasets": [{"key": d.key, "title": d.title} for d in DATASETS.values()],
            "active": self.active,
            "phase": self.phase,
            "speed": self.speed,
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
    return {
        "mass2-live-apply": subprocess.Popen(apply),
        "mass2-live-sim": subprocess.Popen(sim),
        "mass2-live-fit": subprocess.Popen(fit),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `mass2-live-demo`."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "workdir", type=Path, nargs="?", default=Path("mass2_live_demo"), help="output directory (default ./mass2_live_demo)"
    )
    p.add_argument("--dataset", choices=sorted(DATASETS), default="bessy_20240727", help="dataset to start with")
    p.add_argument(
        "--repeats", type=int, default=0,
        help="passes over the data in one run, after which the run starts over (default 0: replay without end)",
    )
    p.add_argument("--max-gb", type=float, default=20.0, help="start the run over when its files reach this size, GB (default 20)")
    p.add_argument(
        "--speed", type=float, default=5.0, help="starting playback speed, multiples of real time (default 5); change it in the page"
    )
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--lan", action="store_true", help="serve the viewer to other devices on this network, e.g. a phone")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)

    args.workdir.mkdir(parents=True, exist_ok=True)
    store = HistogramStore(args.workdir / args.dataset / "hist")
    controller = DemoController(args.workdir, store, args.repeats)
    controller.speed = args.speed
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    server, port = start_server(store, args.port, host, controller)
    controller.switch(args.dataset)

    urls = viewer_urls(host, port)
    print(f"mass2-live-demo: viewer at {'  '.join(urls)}  (Ctrl-C to stop)", flush=True)
    if args.lan:
        print("mass2-live-demo: open the first address on a phone on the same network. Anyone on it can view and switch.", flush=True)
    if not args.no_browser:
        webbrowser.open(urls[-1])

    # SIGTERM (e.g. from `kill` or `timeout`) must clean up the children exactly as Ctrl-C does.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        checked = 0.0
        while not (failed := controller.failed()):
            if controller.active is not None and controller.pass_done():
                controller.switch(controller.active)  # every pass asked for is done: start over
            if time.time() - checked > 10:
                checked = time.time()
                if controller.active is not None and controller.run_bytes() > args.max_gb * 1e9:
                    controller.switch(controller.active)  # start over, so the run's files never outgrow --max-gb
            time.sleep(0.5)
        print(f"mass2-live-demo: {', '.join(failed)} failed; stopping everything", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        server.shutdown()


if __name__ == "__main__":
    main()

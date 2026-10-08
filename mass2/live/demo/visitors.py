"""Give each visitor a run of their own.

Opening the viewer's address starts a new run (its own simulator, applier and line fit, in its own folder)
and sends the visitor to the run's own address, /r/<id>/. Everything the page does there (dataset, playback
speed, states) affects only that run; sending someone the address shows them the same run. A run nobody has
viewed for `idle_s` seconds is stopped and its folder deleted. When `max_runs` are already running, a new
visitor gets a page that retries every few seconds until one is free.
"""

import re
import secrets
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..viewer.server import Controller, HistogramStore, MemorySampler, Route, Site

RUN_PATH = re.compile(r"^/r/([0-9a-f]{12})(/.*)?$")


@dataclass
class VisitorRun:
    site: Site
    controller: Controller
    run_dir: Path
    last_seen: float = field(default_factory=time.time)


class VisitorRuns:
    """The runs, by id; a `Router` for the viewer server."""

    def __init__(
        self,
        workdir: Path,
        new_controller: Callable[[Path, HistogramStore], Controller],
        *,
        max_runs: int,
        dataset: str,
        idle_s: float = 180.0,
    ):
        """`new_controller(run_dir, store)` makes what runs a new run's pipeline (a `DemoController`)."""
        self.workdir, self.new_controller = Path(workdir), new_controller
        self.max_runs, self.dataset, self.idle_s = max_runs, dataset, idle_s
        self.runs: dict[str, VisitorRun] = {}
        self._lock = threading.Lock()
        shutil.rmtree(self.workdir / "runs", ignore_errors=True)  # runs from an earlier server are nobody's now

    def route(self, path: str) -> Route:
        if path == "/":
            run_id = secrets.token_hex(6)
            return Route(None, path, redirect=f"/r/{run_id}/") if self._run(run_id) else Route(None, path, busy=True)
        if path.endswith("/arrow.js"):
            return Route(None, "/arrow.js")
        m = RUN_PATH.match(path)
        if not m:
            return Route(None, path)
        run_id, rest = m.groups()
        if rest is None:
            return Route(None, path, redirect=f"/r/{run_id}/")
        run = self._run(run_id)  # an address whose run has ended starts a new one under the same address
        if run is None:
            return Route(None, rest, busy=True)
        run.last_seen = time.time()
        return Route(run.site, rest)

    def _run(self, run_id: str) -> VisitorRun | None:
        """The run `run_id`, started if need be; None when every run is in use."""
        with self._lock:
            if run_id in self.runs:
                return self.runs[run_id]
            if len(self.runs) >= self.max_runs:
                return None
            run_dir = self.workdir / "runs" / run_id
            store = HistogramStore(run_dir / self.dataset / "hist")
            controller = self.new_controller(run_dir, store)
            memory = MemorySampler("mass2-live-demo (serves every run)", controller)
            run = VisitorRun(Site(store, controller, memory, extra=lambda: {"visitor": self.describe()}), controller, run_dir)
            self.runs[run_id] = run
        controller.switch(self.dataset)
        return run

    def describe(self) -> dict:
        return {"runs": len(self.runs), "max_runs": self.max_runs, "idle_s": self.idle_s}

    def tick(self, max_gb: float) -> None:
        """Keep every run going (see `DemoController.maintain`) and end the runs nobody is viewing."""
        now = time.time()
        with self._lock:
            idle = [rid for rid, run in self.runs.items() if now - run.last_seen > self.idle_s]
            ended = [self.runs.pop(rid) for rid in idle]
            running = list(self.runs.values())
        for run in ended:
            _end(run)
        for run in running:
            run.controller.maintain(max_gb)

    def stop_all(self) -> None:
        with self._lock:
            ended, self.runs = list(self.runs.values()), {}
        for run in ended:
            _end(run)


def _end(run: VisitorRun) -> None:
    """Stop a run's tools and delete its folder."""
    run.controller.stop()
    shutil.rmtree(run.run_dir, ignore_errors=True)

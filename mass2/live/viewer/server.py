"""Serve a live web view of the histograms written by `mass2-live-apply`.

GET  /                     the viewer page
GET  /api/state?since=K&row_s=L   an Arrow IPC stream: the histograms (finalized slices with index >= K or, for a
                           page starting or far behind, a base of totals and time-plot rows L s long, and the open
                           slice), with meta, status, channels, states and fits as JSON in its schema metadata
GET  /arrow.js             Apache Arrow's JavaScript library, which the page reads the stream with
GET  /fits/latest.png      the latest line fit drawn by mass2-live-fit
POST /api/states           {"states": [LABEL, ...] or null} the states the spectrum and the line fit include, for every viewer
POST /api/dataset          {"key": NAME} switch dataset   } only when run by mass2-live-demo,
POST /api/speed            {"speed": X} playback speed  } which owns the simulator

A page starts from a `base`: the totals and the newest time-plot rows, the same size however long the run
has been going. It then asks only for newer slices, so each refresh costs roughly one slice's worth of data.
The finished slices themselves stay in this process and in histograms.arrows. `run` changes whenever the
histogram directory is reset (e.g. a dataset switch), telling the browser to drop what it holds.

Command line:  mass2-live-view HIST_DIR [--port 8765] [--lan]
"""

import argparse
import gzip
import json
import os
import socket
import subprocess
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from collections.abc import Callable
from typing import Any, Protocol, TypeVar
from urllib.parse import parse_qs, urlparse

import numpy as np
import polars as pl
import pyarrow as pa
from numpy.typing import NDArray

from ..arrow_stream import ArrowStreamTailer
from ..fit import read_selected_states, write_selected_states
from ..histogram import HistogramSlice, HistogramSpec, df_to_slices


ARROW_JS = "apache-arrow-21.2.0.es2015.min.js"  # Apache Arrow's JavaScript build, from npm, served at /arrow.js
RUN_FILES = [
    "pulses.arrows", "pulses_experiment_state.txt", "analyzed.arrows", "hist/histograms.arrows", "hist/histograms_current.arrows",
    "hist/states.json", "hist/status.json", "hist/fits/fits.json", "hist/fits/latest.png",
]  # fmt: skip
ROWS = 120  # time-plot rows a page is sent at the start: more than its canvas shows (at most ~105)
CATCH_UP_SLICES = 60  # a page further behind than this is sent a fresh `base` instead of the slices it missed
KEEP_SLICES = 20_000  # slices kept, summed over channels, for time-plot rows: a few hundred MB at most


class Controller(Protocol):
    """What the viewer needs from whoever runs the pipeline, to offer a dataset switcher."""

    def describe(self) -> dict: ...  # {"datasets": [{"key", "title"}], "active": key, "phase": text}

    def switch(self, key: str) -> None: ...

    def set_speed(self, speed: float) -> None: ...

    def pids(self) -> dict[str, int]: ...  # the pipeline processes it runs, by name


class HistogramStore:
    """Follows the histogram files in `hist_dir`. It keeps the totals of every finished slice, the newest
    `CATCH_UP_SLICES` slices in full (for pages keeping up), and for the time plot the newest `KEEP_SLICES`
    slices summed over channels, nonempty bins only. The full record of every slice is histograms.arrows."""

    def __init__(self, hist_dir: str | Path):
        self._lock = threading.Lock()
        self.run = 0
        self.reset(hist_dir)

    def reset(self, hist_dir: str | Path) -> None:
        """Forget everything and follow `hist_dir` from the start."""
        with self._lock:
            self.hist_dir = Path(hist_dir)
            self._tailer = ArrowStreamTailer(self.hist_dir / "histograms.arrows")
            self.meta: dict | None = None
            self.n_slices = 0  # finished slices so far
            self.recent: deque[HistogramSlice] = deque(maxlen=CATCH_UP_SLICES)
            self.summed: deque[tuple[int, dict[str, Sparse]]] = deque(maxlen=KEEP_SLICES)  # (start µs, by state)
            self.totals: dict[tuple[int, str], NDArray[np.int64]] = {}  # every finished slice, summed
            self.channels: set[int] = set()
            self.run += 1

    def _update(self) -> None:
        if self.meta is None:
            meta_path = self.hist_dir / "histograms_meta.json"
            if not meta_path.exists():
                return
            self.meta = json.loads(meta_path.read_text())
        for df in self._tailer.poll():
            for s in df_to_slices(df, self.spec):
                self.channels.update(ch for ch, _ in s.counts)
                if self.recent and self.recent[-1].start_us == s.start_us:
                    self.recent[-1].counts.update(s.counts)  # a slice can arrive in more than one batch
                    self.summed[-1] = (s.start_us, _summed_sparse(self.recent[-1]))
                else:
                    self.recent.append(s)
                    self.summed.append((s.start_us, _summed_sparse(s)))
                    self.n_slices += 1
                for key, c in s.counts.items():
                    if key in self.totals:
                        self.totals[key] += c
                    else:
                        self.totals[key] = c.copy()

    @property
    def spec(self) -> HistogramSpec:
        assert self.meta is not None
        return HistogramSpec(self.meta["e_lo"], self.meta["e_hi"], self.meta["bin_width"], self.meta["slice_s"])

    def state(self, since: int = 0, row_s: float | None = None) -> tuple[dict, list["CountRow"]]:
        """Everything the page needs: a small dict (run, meta, status, states, fits, ...) and the counts.

        The counts are finished slices from index `since` onward, by channel, and the open slice(s). A page
        that is starting, or far behind, gets a base instead of every slice so far: the totals of all
        finished slices, and the newest `ROWS` complete rows of the time plot (rows `row_s` long, summed over
        channels), followed by the finished slices of the row still filling, summed over channels (their
        counts are already in the totals). What it is sent then stays the same size however long the run
        has gone; the slices themselves stay on this machine.
        """
        with self._lock:
            self._update()
            info: dict[str, Any] = {"run": self.run, "meta": self.meta, "status": None, "channels": [], "states": [], "n_slices": 0,
                                    "selected_states": read_selected_states(self.hist_dir), "file_bytes": self.file_bytes()}  # fmt: skip
            if self.meta is None:
                return info, []
            current: list[HistogramSlice] = _read_if_exists(
                self.hist_dir / "histograms_current.arrows", lambda p: df_to_slices(pl.read_ipc_stream(p), self.spec), []
            )
            n, rows = self.n_slices, []
            base = since <= 0 or since > n or n - since > len(self.recent)
            if base:
                row_us, rows = self._base(row_s)
                info["row_s"] = row_us / 1e6
            else:
                rows = [CountRow.dense("slice", s.start_us / 1e6, st, ch, c) for s in list(self.recent)[len(self.recent) - (n - since):] for (ch, st), c in s.counts.items()]
            rows += [CountRow.dense("open", s.start_us / 1e6, st, ch, c) for s in current for (ch, st), c in s.counts.items()]
            return info | {
                "base": base,
                "status": _read_if_exists(self.hist_dir / "status.json", lambda p: json.loads(p.read_text()), None),
                "states": _read_if_exists(self.hist_dir / "states.json", lambda p: json.loads(p.read_text()), []),
                "channels": [str(ch) for ch in sorted(self.channels | {ch for s in current for ch, _ in s.counts})],
                "n_slices": n,
                "fits": _read_if_exists(self.hist_dir / "fits" / "fits.json", lambda p: json.loads(p.read_text()), None),
            }, rows

    def file_bytes(self) -> dict[str, int]:
        """Sizes of the run's files that exist, by path relative to the run directory (the histogram directory's parent)."""
        run_dir, out = self.hist_dir.parent, {}
        for name in RUN_FILES:
            try:
                out[name] = (run_dir / name).stat().st_size
            except FileNotFoundError:
                pass
        return out

    def select_states(self, states: object) -> None:
        """Set the states the spectrum and the fit include, for every viewer: a list of labels, or None for all.
        Raises ValueError for anything else."""
        if states is not None and not (isinstance(states, list) and len(states) <= 64 and all(isinstance(x, str) and len(x) <= 64 for x in states)):
            raise ValueError("states must be null or a list of state labels")
        with self._lock:
            self.hist_dir.mkdir(parents=True, exist_ok=True)
            write_selected_states(self.hist_dir, states)

    def _base(self, row_s: float | None) -> tuple[int, list["CountRow"]]:
        """For `state`: the row length in µs, and the totals, the newest `ROWS` complete time-plot rows, and the
        slices of the row still filling, summed over channels (the totals already hold their counts)."""
        slice_us = round(self.spec.slice_s * 1e6)
        row_us = max(1, round((row_s or self.spec.slice_s) * 1e6 / slice_us)) * slice_us
        rows = [CountRow.dense("total", None, st, ch, c) for (ch, st), c in self.totals.items()]
        if not self.summed:
            return row_us, rows
        open_row = (self.summed[-1][0] + slice_us) // row_us  # rows on a fixed grid, as the page lays them out
        sums: dict[int, dict[str, NDArray[np.int64]]] = {}
        for start_us, by_state in self.summed:
            r = start_us // row_us
            if r >= open_row:
                rows += [CountRow("slice", start_us / 1e6, st, 0, bins, counts) for st, (bins, counts) in by_state.items()]
            elif r >= open_row - ROWS:
                for st, (bins, counts) in by_state.items():
                    np.add.at(sums.setdefault(r, {}).setdefault(st, np.zeros(self.spec.nbins, np.int64)), bins, counts)
        rows += [CountRow.dense("row", r * row_us / 1e6, st, 0, c) for r in sorted(sums) for st, c in sums[r].items()]
        return row_us, rows


Sparse = tuple[NDArray[np.int32], NDArray[np.int32]]  # (nonempty bins, their counts)


@dataclass(frozen=True)
class CountRow:
    """One histogram on the wire: `kind` is "total" (all finished slices), "row" (a time-plot row, summed over
    channels, ch_num 0), "slice" (a finished slice; ch_num 0 when summed over channels) or "open" (the slice
    still filling); `t` its start, seconds; `bins` its nonempty bins and `counts` their counts."""

    kind: str
    t: float | None
    state: str
    ch_num: int
    bins: NDArray[np.int32]
    counts: NDArray[np.int32]

    @classmethod
    def dense(cls, kind: str, t: float | None, state: str, ch_num: int, counts: NDArray) -> "CountRow":
        return cls(kind, t, state, ch_num, *_sparse(counts))


def _sparse(counts: NDArray) -> Sparse:
    nz = np.flatnonzero(counts).astype(np.int32)
    return nz, counts[nz].astype(np.int32)


def _summed_sparse(s: HistogramSlice) -> dict[str, Sparse]:
    """A slice summed over channels, by state, nonempty bins only."""
    out: dict[str, NDArray[np.int64]] = {}
    for (_, st), c in s.counts.items():
        out[st] = out[st] + c if st in out else c.copy()
    return {st: _sparse(c) for st, c in out.items()}


WIRE_SCHEMA = pa.schema([
    ("kind", pa.dictionary(pa.int8(), pa.utf8())),
    ("t", pa.float64()),
    ("state", pa.dictionary(pa.int16(), pa.utf8())),
    ("ch_num", pa.int32()),
    ("bins", pa.list_(pa.uint16())),  # the nonempty bins only
    ("counts", pa.list_(pa.int32())),
])  # fmt: skip


def to_arrow_ipc(info: dict, rows: list[CountRow]) -> bytes:
    """An Arrow IPC stream of `rows` (see WIRE_SCHEMA), with `info` as JSON in the schema metadata."""
    offsets = pa.array(np.concatenate([[0], np.cumsum([len(r.bins) for r in rows])]).astype(np.int32))
    bins = np.concatenate([r.bins for r in rows]).astype(np.uint16) if rows else np.zeros(0, np.uint16)
    counts = np.concatenate([r.counts for r in rows]).astype(np.int32) if rows else np.zeros(0, np.int32)
    table = pa.table(
        [
            pa.array([r.kind for r in rows], pa.utf8()).dictionary_encode().cast(WIRE_SCHEMA.field("kind").type),
            pa.array([r.t for r in rows], pa.float64()),
            pa.array([r.state for r in rows], pa.utf8()).dictionary_encode().cast(WIRE_SCHEMA.field("state").type),
            pa.array([r.ch_num for r in rows], pa.int32()),
            pa.ListArray.from_arrays(offsets, pa.array(bins)),
            pa.ListArray.from_arrays(offsets, pa.array(counts)),
        ],
        schema=WIRE_SCHEMA.with_metadata({"mass2.live": json.dumps(info)}),
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def from_arrow_ipc(data: bytes) -> tuple[dict, pa.Table]:
    """The reverse of `to_arrow_ipc`: the info dict and the table."""
    table = pa.ipc.open_stream(data).read_all()
    return json.loads(table.schema.metadata[b"mass2.live"]), table


class MemorySampler:
    """Every `every_s` seconds, the resident memory (RSS) of this process and of the controller's processes,
    read with `ps` (macOS and Linux), so a page can show what each process involved is using."""

    def __init__(self, name: str, controller: "Controller | None", every_s: float = 2.0):
        self.name, self.controller, self.every_s = name, controller, every_s
        self.latest: list[dict] = []
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.latest = process_memory({self.name: os.getpid()} | (self.controller.pids() if self.controller else {}))
            self._stop.wait(self.every_s)


def process_memory(pids: dict[str, int]) -> list[dict]:
    """[{"name", "pid", "rss_bytes"}] for those of `pids` still running."""
    try:
        out = subprocess.run(["ps", "-o", "pid=,rss=", "-p", ",".join(map(str, pids.values()))], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rss = {int(pid): 1024 * int(kb) for pid, kb in (line.split() for line in out.splitlines() if line.strip())}  # ps reports KiB
    return [{"name": name, "pid": pid, "rss_bytes": rss[pid]} for name, pid in pids.items() if pid in rss]


T = TypeVar("T")


def _read_if_exists(path: Path, read: Callable[[Path], T], default: T) -> T:
    """Read a file another process replaces atomically; it may not exist yet."""
    try:
        return read(path)
    except FileNotFoundError:
        return default


@dataclass
class Site:
    """One run as the viewer serves it: its histograms, who runs its pipeline (if anyone), its memory report,
    and anything else its pages should be told."""

    store: HistogramStore
    controller: Controller | None
    memory: MemorySampler
    extra: Callable[[], dict] = dict


@dataclass(frozen=True)
class Route:
    """Where a request goes: a site and the path within it; or a redirect; or "busy" (no run for it now)."""

    site: Site | None
    path: str
    redirect: str | None = None
    busy: bool = False


class Router(Protocol):
    def route(self, path: str) -> Route: ...


class OneSite:
    """Every request goes to the one run."""

    def __init__(self, site: Site):
        self.site = site

    def route(self, path: str) -> Route:
        return Route(self.site, path)


BUSY_PAGE = b"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="10"><title>mass2 live analysis</title>
<body style="font-family: system-ui, sans-serif; max-width: 40em; margin: 3em auto; padding: 0 16px; line-height: 1.5">
<h1 style="font-size: 1.3rem">mass2 live analysis</h1>
<p>Every run this computer can host is in use right now. This page tries again every 10 seconds, and starts a
run of your own as soon as one is free.</p></body>"""


def _handler_for(router: Router) -> type[BaseHTTPRequestHandler]:
    page = resources.files("mass2.live.viewer").joinpath("viewer.html").read_bytes()
    arrow_js = resources.files("mass2.live.viewer").joinpath(ARROW_JS).read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            url = urlparse(self.path)
            r = router.route(url.path)
            if r.redirect:
                self.send_response(303)
                self.send_header("Location", r.redirect)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            if r.path == "/arrow.js":
                self._send(arrow_js, "text/javascript; charset=utf-8")
                return
            if r.busy:
                if r.path.startswith("/api/"):
                    self.send_error(503, "every run is in use")
                else:
                    self._send(BUSY_PAGE, "text/html; charset=utf-8", status=503)
                return
            if r.site is None:
                self.send_error(404)
                return
            store, controller = r.site.store, r.site.controller
            if r.path == "/":
                self._send(page, "text/html; charset=utf-8")
            elif r.path == "/fits/latest.png":
                png = _read_if_exists(store.hist_dir / "fits" / "latest.png", lambda p: p.read_bytes(), None)
                if png is None:
                    self.send_error(404)
                else:
                    self._send(png, "image/png")
            elif r.path == "/api/state":
                query = parse_qs(url.query)
                since, row_s = int(query.get("since", ["0"])[0]), float(query.get("row_s", ["0"])[0])
                info, rows = store.state(since, row_s if row_s > 0 else None)
                info["controller"] = controller.describe() if controller else None
                info["processes"] = r.site.memory.latest
                info |= r.site.extra()
                self._send(to_arrow_ipc(info, rows), "application/vnd.apache.arrow.stream")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            r = router.route(urlparse(self.path).path)
            if r.busy:
                self.send_error(503, "every run is in use")
                return
            site = r.site
            if site is None or r.path not in {"/api/states", "/api/dataset", "/api/speed"} or (site.controller is None and r.path != "/api/states"):
                self.send_error(404)
                return
            controller = site.controller
            try:
                request = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 65536)) or b"{}")
                if r.path == "/api/states":
                    site.store.select_states(request["states"])
                    self._send(b"{}", "application/json")
                    return
                if controller is None:
                    raise KeyError(r.path)
                if r.path == "/api/dataset":
                    controller.switch(request["key"])
                else:
                    controller.set_speed(float(request["speed"]))
            except (KeyError, ValueError, TypeError, AttributeError):
                self.send_error(400, "bad request")
                return
            self._send(json.dumps(controller.describe()).encode(), "application/json")

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            if content_type != "image/png" and "gzip" in self.headers.get("Accept-Encoding", ""):
                body = gzip.compress(body, compresslevel=5)  # the counts are mostly digits: several times smaller
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass  # keep the terminal for the tools' own output

    return Handler


def start_server(
    store: HistogramStore, port: int = 8765, host: str = "127.0.0.1", controller: Controller | None = None
) -> tuple[ThreadingHTTPServer, int]:
    """Serve one run in a background thread. Returns the server and the actual port (useful with port=0)."""
    site = Site(store, controller, MemorySampler("mass2-live-demo" if controller else "mass2-live-view", controller))
    return start_router_server(OneSite(site), port, host)


def start_router_server(router: Router, port: int = 8765, host: str = "127.0.0.1") -> tuple[ThreadingHTTPServer, int]:
    """Serve whatever `router` sends each request to, in a background thread."""
    server = ThreadingHTTPServer((host, port), _handler_for(router))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def lan_address() -> str:
    """This machine's address on its local network, for opening the viewer from a phone."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))  # no packet is sent; this only picks the outgoing interface
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def viewer_urls(host: str, port: int) -> list[str]:
    """URLs to print for a server bound to `host`."""
    if host == "0.0.0.0":
        return [f"http://{lan_address()}:{port}/", f"http://127.0.0.1:{port}/"]
    return [f"http://{host}:{port}/"]


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `mass2-live-view`."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hist_dir", type=Path, help="histogram directory written by mass2-live-apply")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--lan", action="store_true", help="serve to other devices on this network, e.g. a phone")
    args = p.parse_args(argv)
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    server, port = start_server(HistogramStore(args.hist_dir), args.port, host)
    print(f"mass2-live-view: {'  '.join(viewer_urls(host, port))}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()

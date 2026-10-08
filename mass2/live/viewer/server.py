"""Serve a live web view of the histograms written by `mass2-live-apply`.

GET  /                     the viewer page
GET  /api/state?since=K&row_s=L   JSON: meta, status, channels, states, finalized slices with index >= K (or, for a
                           page starting or far behind, a `base` of totals and time-plot rows L s long), open slice(s), fits
GET  /fits/latest.png      the latest line fit drawn by mass2-live-fit
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
import socket
import threading
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from collections.abc import Callable
from typing import Any, Protocol, TypeVar
from urllib.parse import parse_qs, urlparse

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ..arrow_stream import ArrowStreamTailer
from ..histogram import HistogramSlice, HistogramSpec, df_to_slices, sparse_counts


ROWS = 120  # time-plot rows a page is sent at the start: more than its canvas shows (at most ~105)
CATCH_UP_SLICES = 60  # a page further behind than this is sent a fresh `base` instead of the slices it missed


class Controller(Protocol):
    """What the viewer needs from whoever runs the pipeline, to offer a dataset switcher."""

    def describe(self) -> dict: ...  # {"datasets": [{"key", "title"}], "active": key, "phase": text}

    def switch(self, key: str) -> None: ...

    def set_speed(self, speed: float) -> None: ...


class HistogramStore:
    """Follows the histogram files in `hist_dir` and keeps every finalized slice in memory."""

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
            self.slices: list[HistogramSlice] = []
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
                if self.slices and self.slices[-1].start_us == s.start_us:
                    self.slices[-1].counts.update(s.counts)  # a slice can arrive in more than one batch
                else:
                    self.slices.append(s)
                for key, c in s.counts.items():
                    if key in self.totals:
                        self.totals[key] += c
                    else:
                        self.totals[key] = c.copy()

    @property
    def spec(self) -> HistogramSpec:
        assert self.meta is not None
        return HistogramSpec(self.meta["e_lo"], self.meta["e_hi"], self.meta["bin_width"], self.meta["slice_s"])

    def state(self, since: int = 0, row_s: float | None = None) -> dict:
        """Everything the page needs, with finalized slices from index `since` onward.

        A page that is starting, or far behind, gets a `base` instead of every slice so far: the totals of
        all finished slices, and the newest `ROWS` complete rows of the time plot (rows `row_s` long, summed
        over channels), followed by only the finished slices of the row still filling. What it is sent then
        stays the same size however long the run has gone; the slices themselves stay on this machine.
        """
        with self._lock:
            self._update()
            empty: dict[str, Any] = {
                "run": self.run,
                "meta": None,
                "status": None,
                "channels": [],
                "states": [],
                "slices": [],
                "current": [],
                "n_slices": 0,
            }
            if self.meta is None:
                return empty
            current: list[HistogramSlice] = _read_if_exists(
                self.hist_dir / "histograms_current.arrows", lambda p: df_to_slices(pl.read_ipc_stream(p), self.spec), []
            )
            channels = self.channels | {ch for s in current for ch, _ in s.counts}
            n = len(self.slices)
            base = None
            if since <= 0 or since > n or n - since > CATCH_UP_SLICES:
                base, since = self._base(row_s)
            return empty | {
                "base": base,
                "meta": self.meta,
                "status": _read_if_exists(self.hist_dir / "status.json", lambda p: json.loads(p.read_text()), None),
                "states": _read_if_exists(self.hist_dir / "states.json", lambda p: json.loads(p.read_text()), []),
                "channels": [str(ch) for ch in sorted(channels)],
                "n_slices": len(self.slices),
                # after a base, the slices of the row still filling are in its totals: the time plot needs only their sum over channels
                "slices": [_slice_json(s, summed=base is not None) for s in self.slices[since:]],
                "current": [_slice_json(s) for s in current],
                "fits": _read_if_exists(self.hist_dir / "fits" / "fits.json", lambda p: json.loads(p.read_text()), None),
            }

    def _base(self, row_s: float | None) -> tuple[dict, int]:
        """The `base` of `state`, and the index of the first finished slice the page still needs."""
        slice_us = round(self.spec.slice_s * 1e6)
        row_us = max(1, round((row_s or self.spec.slice_s) * 1e6 / slice_us)) * slice_us
        row_of = [s.start_us // row_us for s in self.slices]  # rows on a fixed grid, as the page lays them out
        open_row = (self.slices[-1].start_us + slice_us) // row_us if self.slices else 0
        first_open = next((k for k, r in enumerate(row_of) if r >= open_row), len(self.slices))
        sums: dict[int, dict[str, NDArray[np.int64]]] = {}
        for s, r in zip(self.slices[:first_open], row_of[:first_open]):
            if r >= open_row - ROWS:
                row = sums.setdefault(r, {})
                for (_, state), c in s.counts.items():
                    row[state] = row[state] + c if state in row else c.copy()
        rows = [{"t": r * row_us / 1e6, "counts": sparse_counts({(0, st): c for st, c in sums[r].items()})} for r in sorted(sums)]
        return {"totals": sparse_counts(self.totals), "row_s": row_us / 1e6, "rows": rows}, first_open


T = TypeVar("T")


def _read_if_exists(path: Path, read: Callable[[Path], T], default: T) -> T:
    """Read a file another process replaces atomically; it may not exist yet."""
    try:
        return read(path)
    except FileNotFoundError:
        return default


def _slice_json(s: HistogramSlice, summed: bool = False) -> dict:
    """{"t": start seconds, "counts": {state: {channel: [bin gap, count, ...]}}} (see `sparse_counts`).
    `summed`: one entry per state, channel "0", holding the sum over channels."""
    counts = s.counts
    if summed:
        counts = {}
        for (_, state), c in s.counts.items():
            counts[(0, state)] = counts[(0, state)] + c if (0, state) in counts else c.copy()
    return {"t": s.start_us / 1e6, "counts": sparse_counts(counts)}


def _handler_for(store: HistogramStore, controller: Controller | None) -> type[BaseHTTPRequestHandler]:
    page = resources.files("mass2.live.viewer").joinpath("viewer.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path == "/":
                self._send(page, "text/html; charset=utf-8")
            elif url.path == "/fits/latest.png":
                png = _read_if_exists(store.hist_dir / "fits" / "latest.png", lambda p: p.read_bytes(), None)
                if png is None:
                    self.send_error(404)
                else:
                    self._send(png, "image/png")
            elif url.path == "/api/state":
                query = parse_qs(url.query)
                since, row_s = int(query.get("since", ["0"])[0]), float(query.get("row_s", ["0"])[0])
                body = store.state(since, row_s if row_s > 0 else None) | {"controller": controller.describe() if controller else None}
                self._send(json.dumps(body).encode(), "application/json")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if controller is None or path not in {"/api/dataset", "/api/speed"}:
                self.send_error(404)
                return
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                if path == "/api/dataset":
                    controller.switch(request["key"])
                else:
                    controller.set_speed(float(request["speed"]))
            except (KeyError, ValueError, TypeError):
                self.send_error(400, "bad request")
                return
            self._send(json.dumps(controller.describe()).encode(), "application/json")

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
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
    """Serve in a background thread. Returns the server and the actual port (useful with port=0)."""
    server = ThreadingHTTPServer((host, port), _handler_for(store, controller))
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

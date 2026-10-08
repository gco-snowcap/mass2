"""Refit one spectral line, on all channels summed, while the live analysis runs.

This is its own process so the analysis never waits for it. It follows the finished time slices that
`mass2-live-apply` appends to HIST_DIR/histograms.arrows, keeps a running sum over all channels for each
experiment state, and at most every `--every` seconds (when new slices have arrived) fits the region of
interest, in the states selected in HIST_DIR/selected_states.json (all states when there is none), with the
same mass2 line model and binning as `Channel.linefit`. The viewer writes that selection, and it is the one
its spectrum shows; a new selection is fitted at once and starts a new series of fits. Each fit is appended
to HIST_DIR/fits/fits.json and drawn with `LineModelResult.plotm()` to HIST_DIR/fits/latest.png.

Command line:  mass2-live-fit HIST_DIR --line 600 --dlo 20 --dhi 20 [--every 10]
"""

import argparse
import io
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .parent import exit_with_parent
from .arrow_stream import ArrowStreamTailer, write_text_atomically
from .histogram import HistogramSpec, df_to_slices


SELECTED_STATES = "selected_states.json"  # in HIST_DIR: a JSON list of state labels, or null for all states


def read_selected_states(hist_dir: str | Path) -> list[str] | None:
    """The states the viewer has selected for its spectrum and the fit; None means all states."""
    try:
        return json.loads((Path(hist_dir) / SELECTED_STATES).read_text())
    except FileNotFoundError:
        return None


def write_selected_states(hist_dir: str | Path, states: Sequence[str] | None) -> None:
    Path(hist_dir).mkdir(parents=True, exist_ok=True)
    write_text_atomically(json.dumps(None if states is None else sorted(set(states))), Path(hist_dir) / SELECTED_STATES)


@dataclass(frozen=True)
class RoiFit:
    """A line to fit: the `line` argument of `Channel.linefit`, the window around its peak, and where it came from."""

    line: str | float
    dlo: float
    dhi: float
    source: str = ""

    @property
    def label(self) -> str:
        return self.line if isinstance(self.line, str) else f"{self.line:g} eV line"


def fit_roi(counts: NDArray, spec: HistogramSpec, roi: RoiFit) -> tuple[object, dict]:
    """Fit `roi` to `counts` (a histogram at `spec`'s bins). Returns the lmfit result and a JSON-ready summary.

    Mirrors `Channel.linefit`: same model, guess, fixed dph_de = 1 and bins; the bins are the histogram's own.
    """
    import mass2  # noqa: PLC0415  imported here so the other live tools never pay for the fitting stack

    model = mass2.calibration.algorithms.get_model(roi.line)
    pe = model.spect.peak_energy
    lo = int(np.ceil((pe - roi.dlo - spec.e_lo) / spec.bin_width))
    hi = int(np.floor((pe + roi.dhi - spec.e_lo) / spec.bin_width))
    y = np.asarray(counts[lo:hi], dtype=float)
    bin_centers = spec.e_lo + (np.arange(lo, hi) + 0.5) * spec.bin_width
    params = model.guess(y, bin_centers=bin_centers, dph_de=1)
    params["dph_de"].set(1.0, vary=False)
    result = model.fit(y, params, bin_centers=bin_centers, minimum_bins_per_fwhm=3)
    result.set_label_hints(
        binsize=spec.bin_width, ds_shortname="all channels", attr_str="energy", unit_str="eV", cut_hint="good pulses"
    )
    summary: dict = {
        "counts_in_roi": int(y.sum()), "redchi": float(result.redchi),
        "roi_lo": float(spec.e_lo + lo * spec.bin_width), "roi_hi": float(spec.e_lo + hi * spec.bin_width),
    }  # fmt: skip
    for name, key in [("fwhm", "fwhm"), ("peak_ph", "peak"), ("integral", "integral"), ("background", "background")]:
        p = result.params[name]
        summary[key] = [float(p.value), None if p.stderr is None else float(p.stderr)]
    return result, summary


def plotm_png(result: object, title: str) -> bytes:
    """The fit drawn by `LineModelResult.plotm()`, as a small PNG."""
    import matplotlib  # noqa: PLC0415  (only the fit process needs plotting)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=80)
    result.plotm(ax=ax, title=title)  # type: ignore[attr-defined]
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


class LiveFitter:
    """Running sums of the finished slices, one per experiment state, and the fits made from them."""

    def __init__(self, spec: HistogramSpec, roi: RoiFit):
        self.spec, self.roi = spec, roi
        self.counts: dict[str, NDArray] = {}  # state -> counts summed over channels
        self.states: list[str] | None = None  # the states fitted; None = all
        self.slices = 0
        self.latest_us: int | None = None
        self.fits: list[dict] = []

    def add(self, df: pl.DataFrame) -> None:
        for s in df_to_slices(df, self.spec):
            for (_, state), c in s.counts.items():
                self.counts.setdefault(state, np.zeros(self.spec.nbins))
                self.counts[state] += c
            self.slices += 1
            self.latest_us = s.start_us + self.spec.slice_us

    def select(self, states: list[str] | None) -> bool:
        """Fit only `states` from now on (None = all). A change starts a new series of fits; returns whether it changed."""
        if states == self.states:
            return False
        self.states, self.fits = states, []
        return True

    def summed(self) -> NDArray:
        """The counts in the selected states, summed."""
        out = np.zeros(self.spec.nbins)
        for state, c in self.counts.items():
            if self.states is None or state in self.states:
                out += c
        return out

    def fit(self) -> tuple[dict, bytes] | None:
        """Fit the selected states' running sum; None if the region has too few counts yet."""
        t0 = time.perf_counter()
        try:
            result, summary = fit_roi(self.summed(), self.spec, self.roi)
        except Exception as exc:  # a bad fit must never stop the live tools
            summary, result = {"error": str(exc)}, None
        fit_ms = 1000 * (time.perf_counter() - t0)
        if result is None or summary["counts_in_roi"] < 50:
            return None
        t1 = time.perf_counter()
        which = "all states" if self.states is None else ", ".join(self.states) or "no states"
        png = plotm_png(result, f"{self.roi.label}, all channels, {which}, {self.slices} slices")
        entry = {"t": (self.latest_us or 0) / 1e6, "slices": self.slices, "states": self.states, "fit_ms": round(fit_ms, 1)}
        entry |= summary | {"plot_ms": round(1000 * (time.perf_counter() - t1), 1)}
        self.fits.append(entry)
        return entry, png


def run_fits(hist_dir: str | Path, roi: RoiFit, every_s: float = 10.0, poll_s: float = 1.0) -> None:
    """Follow HIST_DIR until the histogram stream ends, refitting at most every `every_s` seconds."""
    hist_dir = Path(hist_dir)
    meta_path = hist_dir / "histograms_meta.json"
    while not meta_path.exists():
        time.sleep(poll_s)
    meta = json.loads(meta_path.read_text())
    fitter = LiveFitter(HistogramSpec(meta["e_lo"], meta["e_hi"], meta["bin_width"], meta["slice_s"]), roi)
    out = hist_dir / "fits"
    out.mkdir(exist_ok=True)
    tailer = ArrowStreamTailer(hist_dir / "histograms.arrows")
    last_fit, fitted_slices = 0.0, 0
    while True:
        for df in tailer.poll():
            fitter.add(df)
        changed = fitter.select(read_selected_states(hist_dir))
        due = time.time() - last_fit >= every_s or tailer.ended
        if changed or (due and fitter.slices > fitted_slices):
            made = fitter.fit()
            last_fit, fitted_slices = time.time(), fitter.slices
            if made is not None or changed:
                if made is not None:
                    (out / "latest.png.tmp").write_bytes(made[1])
                    (out / "latest.png.tmp").replace(out / "latest.png")
                write_text_atomically(
                    json.dumps({
                        "roi": asdict(roi) | {"label": roi.label},
                        "every_s": every_s,
                        "states": fitter.states,
                        "fits": fitter.fits,
                    }),
                    out / "fits.json",
                )
        if tailer.ended:
            return
        time.sleep(poll_s)


def parse_line(text: str) -> str | float:
    """A line name like MnKAlpha, or an energy in eV."""
    try:
        return float(text)
    except ValueError:
        return text


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `mass2-live-fit`."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hist_dir", type=Path, help="histogram directory written by mass2-live-apply")
    p.add_argument("--line", type=parse_line, required=True, help="line name (e.g. MnKAlpha) or energy in eV")
    p.add_argument("--dlo", type=float, default=50, help="fit from this far below the peak, eV (default 50)")
    p.add_argument("--dhi", type=float, default=50, help="fit to this far above the peak, eV (default 50)")
    p.add_argument("--every", type=float, default=10, help="refit at most this often, seconds (default 10)")
    p.add_argument("--source", default="", help="where this fit comes from, shown in the viewer")
    args = p.parse_args(argv)
    exit_with_parent()
    try:
        run_fits(args.hist_dir, RoiFit(args.line, args.dlo, args.dhi, args.source), args.every)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

"""Figure style: one palette and one set of mark specs for every figure, in a light and a dark variant.

Colors follow the model, never its rank, so a model has the same color in every figure. The categorical order is a
colorblind-checked palette (adjacent pairs separated under protan/deutan/tritan simulation)."""

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

FIGURES = Path(__file__).resolve().parents[1] / "figures"

THEMES = {
    "light": dict(
        surface="#ffffff",
        text="#0b0b0b",
        text2="#52514e",
        muted="#8a8984",
        grid="#e6e5e1",
        series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    ),
    "dark": dict(
        surface="#0d1117",
        text="#f0efec",
        text2="#c3c2b7",
        muted="#8f8e88",
        grid="#2b2f36",
        series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    ),
}
ORDER = [
    "gptoss120b",
    "qwen27b-fp8",
    "qwen38-flash-next",
    "dsv4-flash",
    "glm53-ssmbf16",
    "qwen27b-fp8-dflash",
    "flint-sglang",
    "nemotron35-super-vl",
]
MARKERS = dict(zip(ORDER, ["o", "s", "^", "D", "v", "P", "p", "h"]))


def color(model, theme):
    return THEMES[theme]["series"][ORDER.index(model)]


@contextmanager
def figure(name, theme, size=(8, 4.8), **kw):
    t = THEMES[theme]
    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "text.color": t["text"],
        "axes.labelcolor": t["text2"],
        "axes.edgecolor": t["grid"],
        "xtick.color": t["text2"],
        "ytick.color": t["text2"],
        "axes.grid": True,
        "grid.color": t["grid"],
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.markersize": 6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": t["text"],
    }
    with mpl.rc_context(rc):
        fig, axes = plt.subplots(figsize=size, **kw)
        yield fig, axes, t
        fig.tight_layout()
        FIGURES.mkdir(exist_ok=True)
        fig.savefig(FIGURES / f"{name}-{theme}.svg", metadata={"Date": None}, bbox_inches="tight", pad_inches=0.15)
        if theme == "light":
            fig.savefig(FIGURES / f"{name}.png", dpi=160, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)


def log_axis(ax, which, ticks=None, labels=None):
    """Log scale with plain tick labels and no minor labels."""
    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    if ticks is not None:
        axis.set_ticks(ticks, labels or [f"{v:g}" for v in ticks])
    else:
        axis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(mt.NullFormatter())

"""Shared matplotlib styling for the project's figures: light surface, recessive axes, thin marks."""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"  # series 1
ORANGE = "#eb6834"  # series 2
RED = "#e34948"  # series 8
CRITICAL = "#d03b3b"  # status: rejected / failing
BLUE_RAMP = {100: "#cde2fb", 250: "#86b6ef", 450: "#2a78d6", 600: "#184f95"}

# Diverging blue <-> red with a neutral gray midpoint (blue = better than the benchmark).
DIVERGING = LinearSegmentedColormap.from_list(
    "red_gray_blue", ["#9f2d2c", RED, "#f0efec", "#3987e5", "#184f95"]
)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "text.color": INK,
        "text.parse_math": False,  # labels contain literal dollar signs
        "font.family": "sans-serif",
    })


def thousands(x: float, _pos: int | None = None) -> str:
    return f"{x:,.0f}"

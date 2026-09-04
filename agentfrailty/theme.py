"""
Figure theme: one palette, two validated modes.

Values come from a validated categorical palette. Only the first three
categorical slots are used, which is the set that clears the all-pairs
colour-vision gates in both modes -- so no figure here needs a fourth series,
and none of them has one.

Rules followed, and why each shows up in the code below:
  * dark mode is SELECTED, not an inverted light mode -- its own steps from the
    same hues, chosen for the dark surface
  * text wears ink tokens, never the series colour; a coloured mark beside a
    label carries identity instead
  * grid and axes are recessive hairlines
  * 2px lines, >=8px markers
  * a legend whenever there are 2+ series, plus direct labels (<=4 series), so
    identity is never carried by colour alone
"""

from __future__ import annotations

LIGHT = {
    "mode": "light",
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": ["#2a78d6", "#eb6834", "#1baf7a"],
}

DARK = {
    "mode": "dark",
    "surface": "#1a1a19",
    "page": "#0d0d0d",
    "ink": "#ffffff",
    "ink2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "series": ["#3987e5", "#d95926", "#199e70"],
}

MODES = (LIGHT, DARK)


def apply(fig, ax, t, title="", subtitle="", xlabel="", ylabel=""):
    """Surfaces, spines, grid, and the title block. Ink tokens only."""
    fig.patch.set_facecolor(t["page"])
    ax.set_facecolor(t["surface"])

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(1.0)

    ax.tick_params(colors=t["muted"], labelsize=9, length=3, width=1.0)
    ax.grid(True, axis="y", color=t["grid"], linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)

    if xlabel:
        ax.set_xlabel(xlabel, color=t["ink2"], fontsize=10, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=t["ink2"], fontsize=10, labelpad=8)
    if title:
        ax.set_title(title, color=t["ink"], fontsize=13, loc="left",
                     pad=22 if subtitle else 12, fontweight="600")
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes,
                color=t["ink2"], fontsize=9.5, va="bottom")


def legend(ax, t):
    leg = ax.legend(frameon=False, fontsize=9.5, loc="best")
    for text in leg.get_texts():
        text.set_color(t["ink2"])          # ink, never the series colour
    return leg


def save(fig, path_stem, t):
    import matplotlib.pyplot as plt
    out = f"{path_stem}-{t['mode']}.png"
    fig.savefig(out, dpi=200, facecolor=t["page"], bbox_inches="tight")
    plt.close(fig)
    return out


def direct_labels(ax, items, t, min_gap=0.06, dx=8):
    """
    Place end-of-line labels, pushed apart so they never collide.

    The rule is a legend for 2+ series AND direct labels for <=4, so identity
    never rests on colour alone. Two series ending at 0.16 and 0.14 will
    overlap if labelled naively -- the first render of the self-conditioning
    figure did exactly that. Sort by value and enforce a minimum gap in axis
    fraction, nudging downward.

    items: [(x, y, label)] in data coordinates.
    """
    lo, hi = ax.get_ylim()
    span = hi - lo
    placed = sorted(items, key=lambda it: -it[1])
    ys = []
    for _, y, _ in placed:
        yf = (y - lo) / span
        if ys and ys[-1] - yf < min_gap:
            yf = ys[-1] - min_gap
        ys.append(yf)
    for (x, _, label), yf in zip(placed, ys):
        ax.annotate(label, xy=(x, lo + yf * span), xytext=(dx, 0),
                    textcoords="offset points", color=t["ink2"],
                    fontsize=9, va="center", annotation_clip=False)

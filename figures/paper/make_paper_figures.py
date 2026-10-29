#!/usr/bin/env python3
"""Data panels of the Reverb paper, in the figure style of the STATOR (LLM-Share) paper: one PDF (and a PNG preview)
per panel, a canvas of 3.4 x 1.85 in that the paper prints at 0.32 of the text width (about 0.51 of its size, so the
14-pt text prints at about 7 pt), the value on each bar, a shared legend as a separate strip, one fixed colour per
entity, no bold text.  Nothing is typed in by hand: analytic panels are computed here, measured panels read the
stored results of this repository.  PDFs carry no date, so a rerun gives the same bytes.

  python3 figures/paper/make_paper_figures.py          (from the repository root; numpy and matplotlib)

  bg_pairs_cdf      correlation of two users' channels, i.i.d. Rayleigh, 2/4/8 receive antennas (analysis)
  bg_pairs_loss     share of user pairs that lose more than 3 dB under zero-forcing (analysis)
  bg_pairs_legend   legend strip of the two panels above
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE

# One colour per entity in every figure.  Receive antennas are ordered, so one hue from light to dark.
ANT = {2: "#9cc3ea", 4: "#4f8fd6", 8: "#1f4e96"}
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
PANEL = (3.4, 1.85)
VALUE, LEGEND = 12, 14
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14, "font.weight": "normal",
    "axes.labelweight": "normal", "axes.labelsize": 14.5, "axes.linewidth": 0.8,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "xtick.color": INK, "ytick.color": INK,
    "legend.fontsize": LEGEND, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def style(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.tick_params(length=0)


def save(fig, name):
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=220, metadata={"Software": None})
    plt.close(fig)
    print("wrote", name)


def legend(name, entries):
    fig = plt.figure(figsize=(1.9 * len(entries) + 0.6, 0.38))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in entries]
    fig.legend(handles, [label for label, _ in entries], loc="center", ncol=len(entries), fontsize=LEGEND,
               borderpad=0.15, columnspacing=1.6, handlelength=1.3)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=220, metadata={"Software": None})
    plt.close(fig)
    print("wrote", name)


def pairs():
    """For N receive antennas and i.i.d. Rayleigh channels, the squared correlation of two users' channels,
    |h1^H h2|^2 / (|h1|^2 |h2|^2), follows Beta(1, N-1).  Its CDF in the correlation x is 1 - (1 - x^2)^(N-1).
    Zero-forcing keeps 1 - rho^2 of a user's channel gain, so a loss above 3 dB means rho^2 > 1/2, with
    probability (1/2)^(N-1)."""
    x = np.linspace(0, 1, 401)
    fig, ax = plt.subplots(figsize=PANEL)
    for n, c in ANT.items():
        ax.plot(x, 100 * (1 - (1 - x ** 2) ** (n - 1)), color=c, linewidth=2.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Correlation of Two Users")
    ax.set_ylabel("Pairs (%)")
    style(ax, grid="both")
    save(fig, "bg_pairs_cdf")

    fig, ax = plt.subplots(figsize=PANEL)
    share = {n: 100 * 0.5 ** (n - 1) for n in ANT}
    bars = ax.bar(range(len(share)), list(share.values()), width=0.55, color=list(ANT.values()), zorder=3)
    for b, v in zip(bars, share.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}", ha="center", va="bottom", fontsize=VALUE)
    ax.set_xticks(range(len(share)))
    ax.set_xticklabels([str(n) for n in share])
    ax.set_xlabel("Receive Antennas")
    ax.set_ylim(0, 62)
    ax.set_yticks([0, 25, 50])
    ax.set_ylabel("Pairs (%)")
    style(ax)
    save(fig, "bg_pairs_loss")

    legend("bg_pairs_legend", [(f"{n} Antennas", c) for n, c in ANT.items()])


if __name__ == "__main__":
    pairs()

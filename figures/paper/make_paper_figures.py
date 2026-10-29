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
  Motivational Evaluation (measured; RESULT_SUFFIX=_f35 selects the 3.5 GHz runs, empty selects the 2.14 GHz runs):
  mot_retx, mot_lost   conventional receiver: retransmissions per 100 TBs and TBs lost after four transmissions, macro
                       and small cells at 3/30/120 km/h, with a direct path and with a blocked path (N3, N1, P1)
  mot_corr             first-transmission failure rate of the conventional receiver by the correlation of the pair
  mot_receivers        failed TBs that each receiver decodes again from the same samples, macro cell (N5)
  mot_runtime          time per slot of the L1 (cuPHY median), the small and the large NRx on an A100 (N13)
  mot_codeblocks       failed TBs the large NRx recovers, by the number of code blocks the conventional receiver failed
  mot_delay            delivery time of the failed TBs of a macro cell: retransmission only, or waiting for every
                       recovery (src/sim/harq_latency.py, DDDSU, the timing of src/analysis/paper_numbers.py)
  mot_gpu              failed TBs recovered before the deadline, by cells per server, one GPU or four GPUs (N16)
  mot_lanes            time of one large-NRx run when several run at once on one A100 (N13)
  mot_interference, mot_interference_small   retransmission reduction against the conventional receiver under
                       neighbor-cell interference, published and interference-trained large NRx, macro and small cell (N15)
  mot_fail_legend, mot_time_legend, mot_gpu_legend, mot_interference_legend   legend strips
"""
import glob
import json
import os
import re

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


def legend(name, entries, width=None):
    fig = plt.figure(figsize=(width or 1.9 * len(entries) + 0.6, 0.38))
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


ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results")
SUFFIX = os.environ.get("RESULT_SUFFIX", "")
N16 = os.environ.get("N16_SUFFIX", "")   # "d110_f35": GPU runtime at the 11.0 ms deadline, 3.5 GHz slots
CELL = {"Macro Cell": "#2b6cb0", "Small Cell": "#9ec5ea"}
RX = {"More Iterations": "#dedcd3", "Iterative": "#c2bfb2", "K-Best": "#a39f8f", "Small Neural": "#f3c27a",
      "Large Neural": "#e07b39"}
CONV = "#6b685c"                       # the conventional receiver (cuPHY)
BOUND = "#8c897b"                      # receivers with the true channel: an upper bound
GPUS = {"One GPU": "#9a9a92", "Four GPUs": "#2f8f6f"}
NRX = {"Published Large NRx": "#e07b39", "Interference-Trained": "#8a3d0f"}
L1_MEDIAN_MS = 1.4                     # median cuPHY run time per slot, the Timing class of src/analysis/paper_numbers.py
WAIT = {"Retransmission Only": "#6b685c", "Wait for Recovery": "#c4561d"}


def runs(*names):
    return [n + SUFFIX for n in names]


def merged(names, structure="ai_and_ran"):
    """TB-weighted mean of the per-run statistics of results/<name>.json."""
    rows = [json.load(open(os.path.join(RES, f"{n}.json")))["all"][structure] for n in names]
    w = np.array([r["tbs"] for r in rows], float)
    return {k: float(np.dot(w, [r[k] for r in rows]) / w.sum()) for k in ("resends_per_100", "residual_after_all_pct")}


def traces(names):
    corr, conv, rec, cb = [], [], [], []
    for n in names:
        for f in sorted(glob.glob(os.path.join(RES, n, "c*.npz")), key=lambda f: int(re.search(r"c(\d+)\.npz$", f).group(1))):
            z = np.load(f)
            u = z["conv_t0"].shape[1]
            corr.append(np.repeat(z["corr"][:, 0], u))
            conv.append(z["conv_t0"].reshape(-1))
            rec.append(z["nrx_c0"].reshape(-1))
            cb.append(z["cbfail_t0"].reshape(-1))
    return [np.concatenate(x) for x in (corr, conv, rec, cb)]


def bars(ax, groups, series, colors, fmt="{:.1f}", values=True, width=0.38):
    """Grouped bars: series[label] = values per group."""
    n = len(series)
    for i, (label, vals) in enumerate(series.items()):
        x = np.arange(len(groups)) + (i - (n - 1) / 2) * width
        bs = ax.bar(x, vals, width=width * 0.92, color=colors[label], zorder=3)
        if values:
            for b, v in zip(bs, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=VALUE - 2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)


CONDITIONS = {   # macro cell runs, small cell runs (each list: runs of one condition, different seeds)
    "3\nkm/h": (("n3_uma_v3_cb", "n3_uma_v3_more"), ("n1_umi_v3_cb",)),
    "30\nkm/h": (("n3_uma_v30", "n3_uma_v30_more"), ("n1_umi_v30",)),
    "120\nkm/h": (("n3_uma_v120", "n3_uma_v120_more"), ("n3_umi_v120",)),
    "Direct\nPath": (("n3_uma_los_v3", "n3_uma_los_more"), ("p1_umi_los_v3", "p1_umi_los_more")),
    "Blocked\nPath": (("n3_uma_nlos_v3", "n3_uma_nlos_more"), ("p1_umi_nlos_v3", "p1_umi_nlos_more")),
}


def motivation_failures():
    st = {c: (merged(runs(*m)), merged(runs(*s))) for c, (m, s) in CONDITIONS.items()}
    for key, name, ylab, top in (("resends_per_100", "mot_retx", "Per 100 TBs", 35),
                                 ("residual_after_all_pct", "mot_lost", "TBs (%)", 5)):
        fig, ax = plt.subplots(figsize=PANEL)
        bars(ax, list(st), {"Macro Cell": [st[c][0][key] for c in st], "Small Cell": [st[c][1][key] for c in st]}, CELL,
             values=False)
        ax.set_ylim(0, top)
        ax.set_ylabel(ylab)
        ax.tick_params(axis="x", labelsize=10.5)
        style(ax)
        save(fig, name)
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    fig, ax = plt.subplots(figsize=PANEL)
    series = {}
    for label, names in (("Macro Cell", runs("n3_uma_v3_cb", "n3_uma_v3_more")), ("Small Cell", runs("n1_umi_v3_cb"))):
        corr, conv, _, _ = traces(names)
        series[label] = [100 * (~conv[(corr >= lo) & (corr < hi)]).mean() for lo, hi in zip(edges[:-1], edges[1:])]
    for i, (label, vals) in enumerate(series.items()):
        x = np.arange(5) * 0.2 + 0.1 + (i - 0.5) * 0.08
        ax.bar(x, vals, width=0.075, color=CELL[label], zorder=3)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.set_ylim(0, 35)
    ax.set_ylabel("Failed TBs (%)", fontsize=13.5)
    ax.set_xlabel("Correlation of Two Users")
    style(ax)
    save(fig, "mot_corr")
    legend("mot_fail_legend", list(CELL.items()))


def motivation_redecode():
    names = runs("n5_uma_v3", "n5_uma_more1", "n5_uma_more2", "n5_uma_more3")
    reps = [json.load(open(os.path.join(RES, f"{n}.json"))) for n in names]
    fails = np.array([r["fail_pct"]["baseline_lslin_lmmse"] * r["slots"] for r in reps])   # weight: failed TBs
    keys = {"More Iterations": "baseline_lslin_lmmse@50", "Iterative": "lslin_idd3", "K-Best": "baseline_lmmse_kbest",
            "Small Neural": "nrx_rt", "Large Neural": "nrx"}
    vals = {k: float(np.dot(fails, [r["recovers_pct_of_conventional_failures"][s] for r in reps]) / fails.sum())
            for k, s in keys.items()}
    bound = float(np.dot(fails, [r["recovers_pct_of_conventional_failures"]["baseline_perf_csi_kbest"] for r in reps]) / fails.sum())
    fig, ax = plt.subplots(figsize=PANEL)
    y = np.arange(len(vals))[::-1]
    bs = ax.barh(y, list(vals.values()), height=0.62, color=[RX[k] for k in vals], zorder=3)
    for b, v in zip(bs, vals.values()):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.0f}", ha="left", va="center", fontsize=VALUE)
    ax.axvline(bound, color=BOUND, linestyle=(0, (4, 3)), linewidth=1.4, zorder=2)
    ax.text(bound + 1.5, y[1], f"True\nchannel\n{bound:.0f}", ha="left", va="center", fontsize=VALUE - 1.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(list(vals), fontsize=12.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Decoded Again (%)")
    style(ax, grid="x")
    save(fig, "mot_receivers")

    lanes = json.load(open(os.path.join(RES, "n13_lanes.json")))
    one = {k: next(r for r in lanes[k] if r["lanes"] == 1)["latency_ms_mean"] for k in ("small", "large")}
    t = {"L1\n(cuPHY)": (L1_MEDIAN_MS, CONV), "Small\nNeural": (one["small"], RX["Small Neural"]),
         "Large\nNeural": (one["large"], RX["Large Neural"])}
    fig, ax = plt.subplots(figsize=PANEL)
    bs = ax.bar(range(len(t)), [v for v, _ in t.values()], width=0.55, color=[c for _, c in t.values()], zorder=3)
    for b, (v, _) in zip(bs, t.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=VALUE)
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels(list(t), fontsize=12)
    ax.set_ylim(0, 5.4)
    ax.set_ylabel("Time (ms)")
    style(ax)
    save(fig, "mot_runtime")

    fig, ax = plt.subplots(figsize=PANEL)
    series = {}
    for label, names in (("Macro Cell", runs("n3_uma_v3_cb", "n3_uma_v3_more")), ("Small Cell", runs("n1_umi_v3_cb"))):
        _, conv, rec, cb = traces(names)
        f = ~conv
        groups = [(1, 3), (4, 6), (7, 9), (10, 10)]
        series[label] = [100 * (rec[f & (cb >= lo) & (cb <= hi)]).mean() for lo, hi in groups]
    bars(ax, ["1-3", "4-6", "7-9", "All 10"], series, CELL, fmt="{:.0f}")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Recovered (%)", fontsize=13)
    ax.set_xlabel("Failed Code Blocks of a TB")
    style(ax)
    save(fig, "mot_codeblocks")
    legend("mot_time_legend", list(WAIT.items()) + list(CELL.items()), width=9.4)


class Timing:          # the timing of src/analysis/paper_numbers.py (Evaluation Setup)
    l1, run, deadline, lead = 1.4, 6.3, 11.0, 4.0


def motivation_delay():
    """CDF of the delivery time of the failed TBs of a macro cell, without recovery and waiting for every recovery."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src", "sim"))
    from harq_latency import simulate
    good, total = {}, None
    for n in runs("n3_uma_v3_cb", "n3_uma_v3_more"):
        for f in sorted(glob.glob(os.path.join(RES, n, "c*.npz")), key=lambda f: int(re.search(r"c(\d+)\.npz$", f).group(1))):
            z = np.load(f)
            total = int(z["total"])
            for k in z.files:
                if k.startswith(("conv_", "sep_", "nrx_")):
                    good.setdefault(k, []).append(z[k].reshape(-1))
    g = {k: np.concatenate(v) for k, v in good.items()}
    conv, rec = g["conv_t0"], g["nrx_c0"]
    sep = [None] + [g[f"sep_c{t}"] for t in range(1, total)]
    both = [None] + [g[f"sep_c{t}"] | g[f"nrx_c{t}"] for t in range(1, total)]
    fig, ax = plt.subplots(figsize=PANEL)
    for label, pol, steps in (("Retransmission Only", "harq_only", sep), ("Wait for Recovery", "wait", both)):
        done, _ = simulate(conv, rec, steps, None, pol, "DDDSU", Timing)
        d = np.sort(done[~conv])
        x = np.concatenate([[0], np.repeat(d[np.isfinite(d)], 2), [45]])
        y = np.concatenate([[0, 0], np.repeat(np.arange(1, np.isfinite(d).sum() + 1) / d.size * 100, 2)])[:x.size]
        ax.plot(x, y, color=WAIT[label], linewidth=2.4)
    ax.set_xlim(0, 40)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Delivery Time (ms)")
    ax.set_ylabel("Failed TBs (%)", fontsize=13.5)
    style(ax, grid="both")
    save(fig, "mot_delay")


def motivation_server():
    rows = {}
    for line in open(os.path.join(RES, "n16_summary.txt")):
        f = line.split()
        if len(f) >= 10 and f[0] in ("allb68" + N16, "allb68g4" + N16):
            rows.setdefault(f[0], {}).setdefault(int(f[2]), []).append(float(f[9]))
    fig, ax = plt.subplots(figsize=PANEL)
    for label, pol in (("One GPU", "allb68" + N16), ("Four GPUs", "allb68g4" + N16)):
        cells = sorted(rows[pol])
        ax.plot(cells, [np.mean(rows[pol][c]) for c in cells], color=GPUS[label], linewidth=2.4, marker="o", markersize=6)
    ax.set_xlim(0, 25)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.set_ylim(0, 55)
    ax.set_yticks([0, 25, 50])
    ax.set_xlabel("Cells per Server")
    ax.set_ylabel("Recovered (%)", fontsize=13)
    style(ax, grid="both")
    save(fig, "mot_gpu")

    lanes = json.load(open(os.path.join(RES, "n13_lanes.json")))["large"]
    lanes = [r for r in lanes if r["lanes"] <= 4]
    fig, ax = plt.subplots(figsize=PANEL)
    bs = ax.bar([r["lanes"] for r in lanes], [r["latency_ms_mean"] for r in lanes], width=0.55,
                color=RX["Large Neural"], zorder=3)
    for b, r in zip(bs, lanes):
        ax.text(b.get_x() + b.get_width() / 2, r["latency_ms_mean"] + 0.4, f"{r['latency_ms_mean']:.1f}",
                ha="center", va="bottom", fontsize=VALUE)
    ax.set_xticks([r["lanes"] for r in lanes])
    ax.set_xlabel("Runs at Once on One GPU")
    ax.set_ylim(0, 25)
    ax.set_ylabel("Time (ms)")
    style(ax)
    save(fig, "mot_lanes")

    for cell, name in (("uma", "mot_interference"), ("umi", "mot_interference_small")):
        inr = [0, 3, 6, 10]
        pub = runs(f"irc_{cell}_v3", f"ici_{cell}_inr3", f"ici_{cell}_inr6", f"ici_{cell}_inr10")
        ici = runs(f"ici_{cell}_v3_iciaware", f"ici_{cell}_inr3_iciaware", f"ici_{cell}_inr6_iciaware",
                   f"ici_{cell}_inr10_iciaware")
        fig, ax = plt.subplots(figsize=PANEL)
        for label, names in (("Published Large NRx", pub), ("Interference-Trained", ici)):
            y = [merged([n])["resends_per_100"] / merged([n], "ours")["resends_per_100"] for n in names]
            ax.plot(inr, y, color=NRX[label], linewidth=2.4, marker="o", markersize=6)
            ax.text(inr[-1] + 0.4, y[-1], f"{y[-1]:.1f}\u00d7", va="center", fontsize=VALUE, color=INK)
        ax.set_xlim(-0.5, 12)
        ax.set_xticks(inr)
        ax.set_ylim(1, 3)
        ax.set_yticks([1, 2, 3])
        ax.set_xlabel("Interference (dB)")
        ax.set_ylabel("Reduction (\u00d7)")
        style(ax, grid="both")
        save(fig, name)
    legend("mot_gpu_legend", list(GPUS.items()))
    legend("mot_interference_legend", list(NRX.items()), width=5.6)


if __name__ == "__main__":
    pairs()
    motivation_failures()
    motivation_redecode()
    motivation_delay()
    motivation_server()

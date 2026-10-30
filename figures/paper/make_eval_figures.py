#!/usr/bin/env python3
"""Data panels of the Evaluation of the Reverb paper, in the style of make_paper_figures.py (its style, save, legend and
bar helpers, one PDF and a PNG preview per panel, one colour per entity).  Reverb is the first bar of every group and
the number above a baseline bar is its multiple of Reverb.  Every value is read from the stored results.

  RESULT_SUFFIX=_f35 N16_SUFFIX=d110_f35 python3 figures/paper/make_eval_figures.py      (from the repository root)

  eval_retx, eval_lost   retransmissions per 100 TBs and TBs lost after four transmissions: Reverb (waiting for every
                         recovery), AI-for-RAN and the conventional receiver; macro cell, blocked and direct path of the
                         macro cell, small cell (N3, N1)
  eval_speed             retransmission reduction of Reverb at 3, 30 and 120 km/h, macro and small cell
  eval_delay_cdf         delivery time of the failed TBs of a macro cell: conventional, Reverb with Latency-Neutral and
                         with Full Wait (src/sim/harq_latency.py, the timing of src/analysis/paper_numbers.py)
  eval_delay_retx        retransmissions per 100 TBs of the same three
  eval_la, eval_la_small throughput of Reverb and AI-for-RAN under outer-loop link adaptation (target block error rate
                         10%) as a multiple of the conventional receiver, by SNR, macro and small cell (N6)
  eval_gpu_four, eval_gpu_one   failed TBs recovered before the deadline by cells per server, four GPUs and one GPU,
                         Reverb and the schedulers it is compared with (N16, S1)
  eval_gpu_l1            L1 blocks past their deadline with four GPUs: no recovery, Reverb, two and four recoveries at
                         once per GPU
  eval_gpu_deadline      failed TBs recovered by the recovery deadline, one GPU with 4 cells and four GPUs with 16 cells
  eval_gpu_effect, eval_gpu_goodput   retransmission reduction and throughput under link adaptation by cells per server
                         with only the recoveries that finish on time (results/server_effect<suffix>.json)
  eval_pair_macro, eval_pair_small   retransmissions per 100 TBs with a random partner and with the best partner among
                         4-32 candidates by MMSE sum rate, macro and small cell (N20, U2); dashed: Reverb, random partner
  eval_rx_retx           retransmissions per 100 TBs of a macro cell with each receiver in the L1 and in the recovery path
                         (paper_numbers.discussion: Reverb, Reverb with the real-time NRx as the L1, the real-time NRx
                         as the recovery receiver, AI-for-RAN, the conventional receiver)
  eval_rx_runs           recovery runs per 100 TBs of Reverb with the conventional L1 and with the real-time NRx as the L1
  eval_rx_bound          failed TBs of the LMMSE receiver that K-best detection, the large NRx and K-best detection with the
                         true channel decode, four (n5) and eight (n4b_eval_s30_uma) receive antennas, macro cell
  eval_effect_legend, eval_delay_legend, eval_gpu_legend, eval_rx_legend, eval_rx_bound_legend   legend strips (two rows of Figure 13)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_paper_figures as base  # noqa: E402
from make_paper_figures import plt, style, save, merged, runs, CELL, CONV, RX, GPUS, VALUE, LEGEND, PANEL, PANEL4, RES  # noqa: E402

REVERB = "#c4561d"
SCHEME = {"Reverb": REVERB, "AI-for-RAN": RX["Real-Time NRx"], "Conventional": CONV}
WAITS = {"Reverb, Full Wait": REVERB, "Reverb, Latency-Neutral": "#ee9b63", "Conventional": CONV}
SCHED = {"Reverb": REVERB, "All TBs Before CRC": "#7b5ea7", "All TBs Before CRC, No Rejection": "#c9b8e6",
         "No Rejection": "#3d5a80", "Two Recoveries per GPU": "#b23a6f", "Four Recoveries per GPU": "#e8a3c4",
         "No Recovery": CONV}
N16 = base.N16 or "d110_f35"
CARRIER = N16[N16.index("_"):] if "_" in N16 else ""


def existing(names):
    """The runs of a condition that have finished (results/<name>.json); missing seeds are reported, not invented."""
    have = [n for n in names if os.path.exists(os.path.join(RES, f"{n}.json"))]
    for n in names:
        if n not in have:
            print("  not yet:", n)
    return have


def multiples(ax, bars_by_label, ours, top, fmt="{:.1f}×"):
    """Directly above each baseline bar, its value as a multiple of Reverb's bar in the same group; a label that would
    overlap the label to its left (bars of about the same height) moves up by one text line."""
    pad, line = 0.02 * top, 0.09 * top
    for j, r in enumerate(ours):
        y_prev = -np.inf
        for label, bs in bars_by_label.items():
            if label == "Reverb":
                continue
            b = bs[j]
            v = b.get_height()
            y = max(v + pad, y_prev + line)
            ax.text(b.get_x() + b.get_width() / 2, y, fmt.format(v / r), ha="center", va="bottom", fontsize=VALUE - 2.5)
            y_prev = y


def grouped(ax, groups, series, colors, width=0.3):
    out = {}
    n = len(series)
    for i, (label, vals) in enumerate(series.items()):
        x = np.arange(len(groups)) + (i - (n - 1) / 2) * width
        out[label] = ax.bar(x, vals, width=width * 0.92, color=colors[label], zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    return out


EFFECT = {   # condition: runs of the condition (different seeds)
    "Macro\nCell": ("n3_uma_v3_cb", "n3_uma_v3_more"),
    "Blocked\nPath": ("n3_uma_nlos_v3", "n3_uma_nlos_more"),
    "Direct\nPath": ("n3_uma_los_v3", "n3_uma_los_more"),
    "Small\nCell": ("n1_umi_v3_cb",),
}
SPEED = {"3": (("n3_uma_v3_cb", "n3_uma_v3_more"), ("n1_umi_v3_cb",)),
         "30": (("n3_uma_v30", "n3_uma_v30_more"), ("n1_umi_v30",)),
         "120": (("n3_uma_v120", "n3_uma_v120_more"), ("n3_umi_v120",))}


def effect():
    st = {}
    for g, names in EFFECT.items():
        have = existing(runs(*names))
        st[g] = {s: merged(have, k) for s, k in (("Reverb", "ours"), ("AI-for-RAN", "ai_for_ran"),
                                                  ("Conventional", "ai_and_ran"))}
    for key, name, ylab, top in (("resends_per_100", "eval_retx", "Per 100 TBs", 42),
                                 ("residual_after_all_pct", "eval_lost", "TBs (%)", 6)):
        fig, ax = plt.subplots(figsize=PANEL)
        series = {s: [st[g][s][key] for g in st] for s in SCHEME}
        bs = grouped(ax, list(st), series, SCHEME)
        multiples(ax, bs, series["Reverb"], top)
        ax.set_ylim(0, top)
        ax.set_ylabel(ylab)
        ax.tick_params(axis="x", labelsize=10.5)
        style(ax)
        save(fig, name)
    fig, ax = plt.subplots(figsize=PANEL)
    x = np.arange(len(SPEED))
    for i, label in enumerate(CELL):
        y = []
        for sp, cells in SPEED.items():
            have = existing(runs(*cells[i]))
            y.append(merged(have, "ai_and_ran")["resends_per_100"] / merged(have, "ours")["resends_per_100"])
        ax.plot(x, y, color=CELL[label], linewidth=2.4, marker="o", markersize=6)
        for xi, yi in zip(x, y):
            ax.text(xi, yi + 0.12, f"{yi:.1f}×", ha="center", va="bottom", fontsize=VALUE - 1)
    ax.set_xticks(x)
    ax.set_xticklabels(list(SPEED))
    ax.set_xlim(-0.4, len(SPEED) - 0.6)
    ax.set_ylim(1, 3)
    ax.set_yticks([1, 2, 3])
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Reduction (×)")
    style(ax, grid="both")
    save(fig, "eval_speed")
    entries = list(SCHEME.items()) + list(CELL.items())
    base.legend("eval_effect_legend", entries, width=9.6)


def delay():
    sys.path.insert(0, os.path.join(base.ROOT, "src", "analysis"))
    import paper_numbers as pn   # its trace loader and timing: the panels show the numbers of numbers.tex
    from harq_latency import simulate
    conv, rec, steps, conv_steps, cand = pn.load_traces(pn.MACRO_SETS)
    res = {}
    for label, pol in (("Conventional", "harq_only"), ("Reverb, Latency-Neutral", "select_cb5"),
                       ("Reverb, Full Wait", "wait")):
        done, tx = simulate(conv, rec, conv_steps if pol == "harq_only" else steps, cand, pol, "DDDSU", pn.Timing, l1_steps=conv_steps)
        res[label] = (done[~conv], 100.0 * (tx - 1).sum() / len(tx))
    fig, ax = plt.subplots(figsize=PANEL4)
    for label in ("Conventional", "Reverb, Latency-Neutral", "Reverb, Full Wait"):
        d = np.sort(res[label][0])
        fin = np.isfinite(d)
        x = np.concatenate([[0], np.repeat(d[fin], 2), [45]])
        y = np.concatenate([[0, 0], np.repeat(np.arange(1, fin.sum() + 1) / d.size * 100, 2)])[:x.size]
        ax.plot(x, y, color=WAITS[label], linewidth=2.2)
    ax.set_xlim(0, 40)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Delivery Time (ms)")
    ax.set_ylabel("Failed TBs (%)", fontsize=13)
    style(ax, grid="both")
    save(fig, "eval_delay_cdf")

    fig, ax = plt.subplots(figsize=PANEL4)
    order = ["Reverb, Full Wait", "Reverb, Latency-Neutral", "Conventional"]
    vals = [res[k][1] for k in order]
    bs = ax.bar(range(3), vals, width=0.55, color=[WAITS[k] for k in order], zorder=3)
    for b, v in zip(bs, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=VALUE)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Full\nWait", "Latency-\nNeutral", "Conven-\ntional"], fontsize=10.5)
    ax.set_ylim(0, 25)
    ax.set_ylabel("Per 100 TBs")
    style(ax)
    save(fig, "eval_delay_retx")

    for name, run in (("eval_la", "n6_uma_v3"), ("eval_la_small", "n6_umi_v3")):
        import json
        olla = json.load(open(os.path.join(RES, f"{run}{base.SUFFIX}_olla.json")))["results"]
        snr = sorted({int(float(k.split("_")[0][3:])) for k in olla})
        fig, ax = plt.subplots(figsize=PANEL4)
        for label, key in (("Reverb", "ours"), ("AI-for-RAN", "ai_for_ran")):
            y = [olla[f"snr{x}_target0.1"][key]["goodput_vs_ai_and_ran"] for x in snr]
            ax.plot(snr, y, color=SCHEME[label], linewidth=2.4, marker="o", markersize=6)
        ax.axhline(1, color=CONV, linewidth=1.6, linestyle=(0, (4, 3)), zorder=2)
        ax.set_xticks(snr)
        ax.set_ylim(0.9, 1.6)
        ax.set_yticks([1.0, 1.2, 1.4, 1.6])
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Throughput (×)", fontsize=13)
        style(ax, grid="both")
        save(fig, name)
    base.legend("eval_delay_legend", list(WAITS.items()) + [("AI-for-RAN", SCHEME["AI-for-RAN"])], width=10.4)


def runtime_rows():
    """Recovered share of failed TBs and L1 blocks past 4 ms per (policy tag, cells), mean over seeds, from
    results/n16_summary.txt (src/analysis/n16_summary.py)."""
    rec, late = {}, {}
    for line in open(os.path.join(RES, "n16_summary.txt")):
        f = line.split()
        if len(f) < 11 or not f[1].isdigit():
            continue
        rec.setdefault(f[0], {}).setdefault(int(f[2]), []).append(float(f[9]))
        late.setdefault(f[0], {}).setdefault(int(f[2]), []).append(float(f[3]))
    mean = lambda d: {t: {c: float(np.mean(v)) for c, v in cs.items()} for t, cs in d.items()}
    return mean(rec), mean(late)


def gpu():
    rec, late = runtime_rows()

    def tag(policy, gpus, extra=""):
        p = {"Reverb": "allb68", "All TBs Before CRC": "parab68", "All TBs Before CRC, No Rejection": "parb68",
             "No Rejection": "allb68", "Two Recoveries per GPU": "allb68", "Four Recoveries per GPU": "allb68",
             "No Recovery": "offb68"}[policy]
        x = {"No Rejection": "admit0", "Two Recoveries per GPU": "l2", "Four Recoveries per GPU": "l4"}.get(policy, "")
        return f"{p}{'g4' if gpus == 4 else ''}{N16}{x}"

    shown = ["Reverb", "All TBs Before CRC", "All TBs Before CRC, No Rejection", "No Rejection", "Two Recoveries per GPU"]
    for name, gpus, ticks in (("eval_gpu_four", 4, [4, 8, 12, 16, 20, 24]), ("eval_gpu_one", 1, [1, 2, 4, 6, 8])):
        fig, ax = plt.subplots(figsize=PANEL4)
        for label in reversed(shown):      # Reverb drawn last, on top
            r = rec[tag(label, gpus)]
            cells = sorted(r)
            ax.plot(cells, [r[c] for c in cells], color=SCHED[label], linewidth=2.6 if label == "Reverb" else 2.0,
                    marker="o", markersize=5, zorder=4 if label == "Reverb" else 3)
        ax.set_xticks(ticks)
        ax.set_xlim(ticks[0] - 0.6, ticks[-1] + 0.6)
        ax.set_ylim(0, 55)
        ax.set_yticks([0, 25, 50])
        ax.set_xlabel("Cells per Server")
        ax.set_ylabel("Recovered (%)", fontsize=13)
        style(ax, grid="both")
        save(fig, name)

    fig, ax = plt.subplots(figsize=PANEL4)
    for label in ("No Recovery", "Four Recoveries per GPU", "Two Recoveries per GPU", "Reverb"):
        r = late[tag(label, 4)]
        cells = sorted(r)
        ax.plot(cells, [max(r[c], 0.005) for c in cells], color=SCHED[label], linewidth=2.6 if label == "Reverb" else 2.0,
                marker="o", markersize=5)
    ax.set_yscale("log")
    ax.set_ylim(0.005, 100)
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.set_xticks([4, 8, 12, 16, 20, 24])
    ax.set_xlim(3.4, 24.6)
    ax.set_xlabel("Cells per Server")
    ax.set_ylabel("Late L1 (%)", fontsize=13)
    style(ax, grid="both")
    save(fig, "eval_gpu_l1")

    fig, ax = plt.subplots(figsize=PANEL4)
    dl = [90, 100, 110, 120, 130]
    for label, gpus, cells, ytext in (("One GPU", 1, 4, 14.5), ("Four GPUs", 4, 16, 26.5)):
        y = [rec[f"allb68{'g4' if gpus == 4 else ''}d{d}{CARRIER}"][cells] for d in dl]
        ax.plot([d / 10 for d in dl], y, color=GPUS[label], linewidth=2.4, marker="o", markersize=5)
        ax.text(13.1, ytext, f"{label}, {cells} Cells", ha="right", va="center", fontsize=VALUE - 1, color=base.INK)
    ax.set_xticks([9, 10, 11, 12, 13])
    ax.set_xlim(8.7, 13.3)
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_xlabel("Deadline (ms)")
    ax.set_ylabel("Recovered (%)", fontsize=13)
    style(ax, grid="both")
    save(fig, "eval_gpu_deadline")

    # server level: the trace-driven gains with only the recoveries that finish on time (src/analysis/server_effect.py)
    eff = json.load(open(os.path.join(RES, f"server_effect{base.SUFFIX}.json")))
    conv = eff["conventional"]["retx_per_100"]
    every = eff["rows"]["every_recovery"]
    for name, key, ylab, ref in (("eval_gpu_effect", "retx", "Reduction (×)", conv / every["retx_per_100"]),
                                 ("eval_gpu_goodput", "goodput", "Throughput (×)", every["goodput_gain_la"])):
        fig, ax = plt.subplots(figsize=PANEL4)
        for label, gpus in (("One GPU", "one"), ("Four GPUs", "four")):
            rows = sorted((r for k, r in eff["rows"].items() if r.get("gpus") == gpus), key=lambda r: r["cells"])
            y = [conv / r["retx_per_100"] if key == "retx" else r["goodput_gain_la"] for r in rows]
            ax.plot([r["cells"] for r in rows], y, color=GPUS[label], linewidth=2.4, marker="o", markersize=5)
            # direct labels: one GPU right of its last point, four GPUs above its curve at 12 cells
            if gpus == "one":
                ax.text(rows[-1]["cells"] + 0.7, y[-1] + (0.0 if key == "retx" else -0.001), label, ha="left",
                        va="center", fontsize=VALUE - 2.5, color=base.INK)
            else:
                ya = y[[r["cells"] for r in rows].index(12)]
                ax.text(12.4, ya + (0.12 if key == "retx" else 0.03), label, ha="left", va="bottom",
                        fontsize=VALUE - 2, color=base.INK)
        ax.axhline(ref, color=REVERB, linewidth=1.6, linestyle=(0, (4, 3)), zorder=2)
        ax.axhline(1, color=CONV, linewidth=1.2, linestyle=(0, (2, 2)), zorder=2)
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
        ax.set_xlim(0, 25)
        ax.set_ylim(0.9, 2.5 if key == "retx" else 1.4)
        ax.set_yticks([1.0, 1.5, 2.0, 2.5] if key == "retx" else [1.0, 1.1, 1.2, 1.3, 1.4])
        ax.set_xlabel("Cells per Server")
        ax.set_ylabel(ylab, fontsize=13)
        style(ax, grid="both")
        save(fig, name)

    entries = [(k, SCHED[k]) for k in shown + ["Four Recoveries per GPU", "No Recovery"]]
    fig = plt.figure(figsize=(10.5, 1.0))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in entries]
    fig.legend(handles, [k for k, _ in entries], loc="center", ncol=3, fontsize=LEGEND, borderpad=0.15,
               columnspacing=1.6, handlelength=1.3)
    for ext, kw in (("pdf", {"metadata": {"CreationDate": None, "ModDate": None}}),
                    ("png", {"dpi": 220, "metadata": {"Software": None}})):
        fig.savefig(os.path.join(base.OUT, f"eval_gpu_legend.{ext}"), **kw)
    plt.close(fig)
    print("wrote eval_gpu_legend")


PAIR = {"Macro Cell": "p2b_uma_u", "Small Cell": "p2b_u"}   # random partner (8 users) and the best partner by sum rate
PAIR_CANDIDATES = [4, 8, 16, 32]


def pairing():
    """Retransmissions per 100 TBs with a random partner and with the best partner among 4-32 candidates by MMSE sum rate
    (N20/U2): Reverb, AI-for-RAN and the conventional receiver; the dashed line is Reverb with a random partner."""
    for (cell, pre), name in zip(PAIR.items(), ("eval_pair_macro", "eval_pair_small")):
        groups, sets = ["Random"], [f"{pre}8_random"]
        for k in PAIR_CANDIDATES:
            if existing(runs(f"{pre}{k}_pf")):
                groups.append(f"Best\nof {k}")
                sets.append(f"{pre}{k}_pf")
        series = {s: [merged(runs(n), k)["resends_per_100"] for n in sets]
                  for s, k in (("Reverb", "ours"), ("AI-for-RAN", "ai_for_ran"), ("Conventional", "ai_and_ran"))}
        fig, ax = plt.subplots(figsize=PANEL)
        bs = grouped(ax, groups, series, SCHEME, width=0.27)
        top = 24 if cell == "Macro Cell" else 20
        multiples(ax, bs, series["Reverb"], top)
        ax.axhline(series["Reverb"][0], color=REVERB, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
        ax.set_ylim(0, top)
        ax.set_ylabel("Per 100 TBs")
        ax.tick_params(axis="x", labelsize=10.5)
        style(ax)
        save(fig, name)
    base.legend("eval_pair_legend", list(SCHEME.items()), width=6.4)

PLACE = {"Reverb": REVERB, "Reverb on AI-for-RAN": "#5b8c5a", "Real-Time NRx as Recovery": "#c99a2e",
         "AI-for-RAN": RX["Real-Time NRx"], "Conventional": CONV}


def receivers():
    """Section 5.3: the receivers in the L1 and in the recovery path (macro traces, every recovery on time), the recovery
    runs of each L1, and the share of the failed TBs that the large NRx decodes against the bound with four and eight
    receive antennas."""
    import paper_numbers as pn
    v, raw = pn.discussion()
    vals = {"Reverb": raw["large_recovery_retx_per_100"], "Reverb on AI-for-RAN": raw["combo_retx_per_100"],
            "Real-Time NRx as Recovery": raw["small_recovery_retx_per_100"], "AI-for-RAN": raw["ai_for_ran_retx_per_100"],
            "Conventional": raw["conventional_retx_per_100"]}
    fig, ax = plt.subplots(figsize=PANEL)
    bs = ax.bar(range(len(vals)), list(vals.values()), width=0.72, color=[PLACE[k] for k in vals], zorder=3)
    for i, (k, b) in enumerate(zip(vals, bs)):
        txt = f"{b.get_height():.1f}" if i == 0 else f"{b.get_height() / vals['Reverb']:.1f}×"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, txt, ha="center", va="bottom", fontsize=VALUE - 2.5)
    ax.set_xticks([])
    ax.set_ylim(0, 26)
    ax.set_ylabel("Per 100 TBs")
    style(ax)
    save(fig, "eval_rx_retx")

    fig, ax = plt.subplots(figsize=PANEL)
    runs_ = {"Reverb": raw["recovery_runs_per_100_conv_l1"], "Reverb on AI-for-RAN": raw["recovery_runs_per_100_small_l1"]}
    xs = [0, 1.4]
    bs = ax.bar(xs, list(runs_.values()), width=0.62, color=[PLACE[k] for k in runs_], zorder=3)
    for b in bs:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=VALUE - 1)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Conventional", "Real-Time\nNRx"], fontsize=10)
    ax.set_xlabel("L1 Receiver", fontsize=12)
    ax.set_xlim(-0.6, 2.0)
    ax.set_ylim(0, 25)
    ax.set_ylabel("Per 100 TBs")
    style(ax)
    save(fig, "eval_rx_runs")

    four = [json.load(open(os.path.join(RES, f"{n}.json"))) for n in runs("n5_uma_v3", "n5_uma_more1", "n5_uma_more2", "n5_uma_more3")]
    w = np.array([r["fail_pct"]["baseline_lslin_lmmse"] * r["slots"] for r in four])
    eight = json.load(open(os.path.join(RES, f"n4b_eval_s30_uma{base.SUFFIX}.json")))["recovers_pct_of_conventional_failures"]
    keys = {"K-Best": "baseline_lmmse_kbest", "Large NRx": "nrx", "True Channel": "baseline_perf_csi_kbest"}
    series = {k: [float(np.dot(w, [r["recovers_pct_of_conventional_failures"][s] for r in four]) / w.sum()), eight[s]]
              for k, s in keys.items()}
    fig, ax = plt.subplots(figsize=PANEL)
    colors = {"K-Best": RX["K-Best"], "Large NRx": RX["Large NRx"], "True Channel": "white"}
    out = grouped(ax, ["4 RX", "8 RX"], series, colors, width=0.27)
    for b in out["True Channel"]:
        b.set_hatch("////"); b.set_edgecolor(base.BOUND); b.set_linewidth(0.8)
    for label, bs in out.items():
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=VALUE - 3)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Decoded (%)", fontsize=13)
    style(ax)
    save(fig, "eval_rx_bound")
    items = list(PLACE.items())
    base.legend("eval_rx_legend", items[:3], width=8.4)
    base.legend("eval_rx_bound_legend", items[3:] + [("K-Best", RX["K-Best"]), ("Large NRx", RX["Large NRx"])], width=8.2)


AI_RATES = {"No AI": "#c4561d", "AI 4 Requests/s per GPU": "#e8955a", "AI 8 Requests/s per GPU": "#f3c9a4"}


def ai():
    """N22: failed TBs recovered with AI inference on the same GPUs (static MPS share 60%) against the runs without AI,
    four GPUs and one GPU, and the AI requests within the SLO with and without recovery (results/n22_summary.json)."""
    n22 = json.load(open(os.path.join(RES, "n22_summary.json")))["rows"]
    rec, _ = runtime_rows()
    for name, g, cells in (("eval_ai_four", 4, [8, 16, 24]), ("eval_ai_one", 1, [1, 2, 4])):
        fig, ax = plt.subplots(figsize=PANEL4)
        base_tag = f"allb68{'g4' if g == 4 else ''}{N16}"
        ax.plot(cells, [rec[base_tag][c] for c in cells], color=AI_RATES["No AI"], linewidth=2.6, marker="o", markersize=5)
        for label, rate in (("AI 4 Requests/s per GPU", 4), ("AI 8 Requests/s per GPU", 8)):
            ax.plot(cells, [n22[f"all_g{g}_r{rate}_c{c}"]["recovered_pct"] for c in cells], color=AI_RATES[label],
                    linewidth=2.2, marker="o", markersize=5)
        ax.set_xticks(cells)
        ax.set_xlim(cells[0] - 0.08 * (cells[-1] - cells[0]), cells[-1] + 0.08 * (cells[-1] - cells[0]))
        ax.set_ylim(0, 50)
        ax.set_yticks([0, 25, 50])
        ax.set_xlabel("Cells per Server")
        ax.set_ylabel("Recovered (%)", fontsize=13)
        style(ax, grid="both")
        save(fig, name)
    fig, ax = plt.subplots(figsize=PANEL4)
    cells = [8, 16, 24]
    for label, rate in (("AI 4 Requests/s per GPU", 4), ("AI 8 Requests/s per GPU", 8)):
        for pol, ls in (("all", "-"), ("off", (0, (4, 3)))):
            ax.plot(cells, [n22[f"{pol}_g4_r{rate}_c{c}"]["ai_within_slo_pct"] for c in cells], color=AI_RATES[label],
                    linewidth=2.2, linestyle=ls, marker="o" if pol == "all" else None, markersize=5)
    ax.set_xticks(cells)
    ax.set_xlim(6.7, 25.3)
    ax.set_ylim(80, 101)
    ax.set_yticks([80, 90, 100])
    ax.set_xlabel("Cells per Server")
    ax.set_ylabel("Within SLO (%)", fontsize=13)
    style(ax, grid="both")
    save(fig, "eval_ai_slo")
    base.legend("eval_ai_legend", list(AI_RATES.items()), width=9.6)


ICI = {   # bar label: (color, run structure)
    "Reverb, NRx Trained with Interference": REVERB,
    "Reverb, NRx Trained without Interference": "#f0a46b",
    "Conventional": CONV,
    "Reverb on MMSE-IRC": "#7a2f0c",
    "MMSE-IRC": "#9fa8b3",
}
ICI_RUNS = {"Macro Cell": [("10 dB", "ici_uma_inr10"), ("20 dB,\n2 Users", "i1_uma_k2_inr20"), ("20 dB,\n4 Users", "i1_uma_k4_inr20")],
            "Small Cell": [("10 dB", "ici_umi_inr10"), ("20 dB,\n2 Users", "i1_umi_k2_inr20"), ("20 dB,\n4 Users", "i1_umi_k4_inr20")]}


def ici_values(run):
    """Retransmissions per 100 TBs of the five bars of a condition, None for a run that has not finished; the runs of the
    NRx trained with interference carry the same conventional and MMSE-IRC receivers."""
    trained, published = (os.path.join(RES, f"{run}{x}{base.SUFFIX}.json") for x in ("_iciaware", ""))
    t = json.load(open(trained))["all"] if os.path.exists(trained) else None
    p = json.load(open(published))["all"] if os.path.exists(published) else None
    ref = t or p
    if ref is None:
        return None
    val = lambda d, k: d[k]["resends_per_100"] if d and k in d else None
    return {"Reverb, NRx Trained with Interference": val(t, "ours"), "Reverb, NRx Trained without Interference": val(p, "ours"),
            "Conventional": val(ref, "ai_and_ran"), "Reverb on MMSE-IRC": val(t, "ours_irc"), "MMSE-IRC": val(ref, "ai_and_ran_irc")}


def interference():
    for (cell, conds), name in zip(ICI_RUNS.items(), ("eval_ici_macro", "eval_ici_small")):
        groups = [(g, ici_values(r)) for g, r in conds if ici_values(r) and ici_values(r)["Reverb, NRx Trained with Interference"]]
        fig, ax = plt.subplots(figsize=PANEL)
        n, width = len(ICI), 0.16
        top = 36
        for gi, (g, v) in enumerate(groups):
            for i, label in enumerate(ICI):
                if v[label] is None:
                    continue
                x = gi + (i - (n - 1) / 2) * width
                ax.bar(x, v[label], width=width * 0.92, color=ICI[label], zorder=3)
                ref = {"Conventional": "Reverb, NRx Trained with Interference", "MMSE-IRC": "Reverb on MMSE-IRC"}.get(label)
                if ref and v[ref]:
                    ax.text(x, v[label] + 0.6, f"{v[label] / v[ref]:.1f}×", ha="center", va="bottom", fontsize=VALUE - 3.5)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g for g, _ in groups])
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_ylim(0, top)
        ax.set_ylabel("Per 100 TBs")
        ax.tick_params(axis="x", labelsize=10.5)
        style(ax)
        save(fig, name)
    fig = plt.figure(figsize=(10.4, 0.75))     # two rows: the five labels do not fit in one
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ICI.values()]
    fig.legend(handles, list(ICI), loc="center", ncol=3, fontsize=LEGEND, borderpad=0.15, columnspacing=1.6,
               handlelength=1.3, frameon=False)
    for ext, kw in (("pdf", {"metadata": {"CreationDate": None, "ModDate": None}}),
                    ("png", {"dpi": 220, "metadata": {"Software": None}})):
        fig.savefig(os.path.join(base.OUT, f"eval_ici_legend.{ext}"), **kw)
    plt.close(fig)
    print("wrote eval_ici_legend")


SENS_RX = {"K-Best, LMMSE Estimate": "#a39f8f", "K-Best, True Channel": "#8c897b"}


def sensitivity():
    """Retransmissions per 100 TBs with 273, 106 and 52 resource blocks (N3, N21), and the failed TBs of the LMMSE
    receiver that K-best detection decodes with the LMMSE estimate and with the true channel, by receive antennas and
    users (N5 with four seeds for 4 antennas and 2 users, U1 for the others)."""
    groups = [("273\nPRBs", ("n3_uma_v3_cb", "n3_uma_v3_more")), ("106\nPRBs", ("n21_uma_p106_v3",)), ("52\nPRBs", ("n21_uma_p52_v3",))]
    st = {g: {s: merged(existing(runs(*n)), k) for s, k in (("Reverb", "ours"), ("AI-for-RAN", "ai_for_ran"),
                                                         ("Conventional", "ai_and_ran"))} for g, n in groups}
    fig, ax = plt.subplots(figsize=PANEL)
    series = {s: [st[g][s]["resends_per_100"] for g, _ in groups] for s in SCHEME}
    bs = grouped(ax, [g for g, _ in groups], series, SCHEME)
    multiples(ax, bs, series["Reverb"], 36)
    ax.set_ylim(0, 36)
    ax.set_ylabel("Per 100 TBs")
    ax.tick_params(axis="x", labelsize=10.5)
    style(ax)
    save(fig, "eval_prb")
    conf = [("4 RX,\n2 Users", ("n5_uma_v3", "n5_uma_more1", "n5_uma_more2", "n5_uma_more3")), ("4 RX,\n4 Users", ("u1_uma_r4_u4",)),
            ("8 RX,\n4 Users", ("u1_uma_r8_u4_b1",)), ("16 RX,\n2 Users", ("u1_uma_r16_u2_b1",))]
    vals = {k: [] for k in SENS_RX}
    shown = []
    for g, names in conf:
        rs = [json.load(open(os.path.join(RES, f"{n}.json"))) for n in existing(runs(*names))]
        if not rs:
            continue
        w = np.array([r["fail_pct"]["baseline_lslin_lmmse"] * r["slots"] for r in rs])
        for label, key in zip(SENS_RX, ("baseline_lmmse_kbest", "baseline_perf_csi_kbest")):
            vals[label].append(float(np.dot(w, [r["recovers_pct_of_conventional_failures"][key] for r in rs]) / w.sum()))
        shown.append(g)
    fig, ax = plt.subplots(figsize=PANEL)
    bs = grouped(ax, shown, vals, SENS_RX, width=0.36)
    for label, bars in bs.items():
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.0f}", ha="center", va="bottom",
                    fontsize=VALUE - 3)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Decoded (%)", fontsize=13)
    ax.tick_params(axis="x", labelsize=10.5)
    style(ax)
    save(fig, "eval_users")
    base.legend("eval_sens_legend", list(SCHEME.items()) + list(SENS_RX.items()), width=10.4)


if __name__ == "__main__":
    os.chdir(base.ROOT)      # paper_numbers.py reads results/ and src/sim relative to the repository root
    effect()
    delay()
    pairing()
    gpu()
    ai()
    interference()
    sensitivity()
    receivers()

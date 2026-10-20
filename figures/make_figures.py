#!/usr/bin/env python3
"""Evaluation figures of the recovery paper (README, docs/RESULTS.md).  Same rules as the Antiphase evaluation figures:
one metric per panel, one color per structure, ours first and blue, bold text, legend on top, and above every other
bar how many times ours that structure is ("how many times our resends" for resends: >1 means it needs more).
Every number comes from results/.  usage: python3 figures/make_figures.py  (from the repository root)"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
                            "hatch.linewidth": 0.7, "axes.linewidth": 0.9})
import matplotlib.pyplot as plt
from matplotlib import font_manager
for _f in ("/usr/share/fonts/truetype/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/LiberationSans-Bold.ttf"):
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
matplotlib.rcParams["font.family"] = "Liberation Sans"     # the conda matplotlib has no bold DejaVu Sans

sys.path.insert(0, "src"); sys.path.insert(0, "src/sim")
OUT = "figures/eval"
os.makedirs(OUT, exist_ok=True)
FS = 10.5
INK = "#111111"
OURS, L1, SMALL = "#1f77b4", "#9a9a9a", "#7fbf6a"
NAMES = {"ours": "Our Strategy: L1 + NRx (Large Model), Recovery",
         "ai_and_ran": "Conventional Receiver (L1) Only",
         "ai_for_ran": "L1 + NRx (Small Model), Every Block"}
COLORS = {"ours": OURS, "ai_and_ran": L1, "ai_for_ran": SMALL}
HATCH = {"ours": False, "ai_and_ran": False, "ai_for_ran": True}
ORDER = ["ours", "ai_and_ran", "ai_for_ran"]


def R(name):
    return json.load(open(f"results/{name}.json"))


def kind(color, hatched):
    if hatched:
        return dict(facecolor="white", edgecolor=color, linewidth=1.3, hatch="////")
    return dict(facecolor=color, edgecolor="black", linewidth=0.8)


def style(ax, ylabel="", xlabel=""):
    ax.set_ylabel(ylabel, fontsize=FS + 1, color=INK)
    ax.set_xlabel(xlabel, fontsize=FS + 1, color=INK)
    ax.grid(axis="y", color="#bbbbbb", linewidth=0.6, linestyle=(0, (4, 3)))
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FS, colors=INK, length=3)


def legend(fig, keys, y=1.0, ncol=None, names=None, colors=None, hatches=None):
    names, colors, hatches = names or NAMES, colors or COLORS, hatches or HATCH
    h = [plt.Rectangle((0, 0), 1, 1, **kind(colors[k], hatches[k])) for k in keys]
    fig.legend(h, [names[k] for k in keys], loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(keys),
               frameon=False, fontsize=FS, handlelength=1.3, columnspacing=1.2, handletextpad=0.45)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


def grouped(ax, groups, values, keys=ORDER, lower_better=True, digits=1, mult=True, ymax=None):
    """values[k][g]: bars per group, ours first; above the other bars: that structure / ours (lower better) or
    ours / that structure (higher better)."""
    w = 0.26
    top = max(max(values[k]) for k in keys)
    for j, k in enumerate(keys):
        off = (j - (len(keys) - 1) / 2) * (w + 0.02)
        for g, v in enumerate(values[k]):
            ax.bar(g + off, v, width=w, zorder=3, **kind(COLORS[k], HATCH[k]))
            o = values["ours"][g]
            if k == "ours" or not mult:
                t, c = f"{v:.{digits}f}", (OURS if k == "ours" else INK)
            else:
                r = (v / o) if lower_better else (o / v)
                t, c = f"{r:.1f}×", INK
            ax.text(g + off, v + top * 0.015, t, ha="center", va="bottom", fontsize=FS - 2, color=c, rotation=0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=FS - 0.5)
    ax.set_ylim(0, ymax or top * 1.18)


# ---------------------------------------------------------------- 1. resends and residual failures across channels
def harq_conditions():
    umi = [("n1_umi_v3", "3 km/h"), ("n1_umi_v30", "30 km/h"), ("n1_umi_v60", "60 km/h"), ("n3_umi_v120", "120 km/h"),
           ("p1_umi_los_v3", "Direct\nPath"), ("p1_umi_nlos_v3", "No Direct\nPath")]
    uma = [("n3_uma_v3", "3 km/h"), ("n3_uma_v30", "30 km/h"), ("n3_uma_v120", "120 km/h"),
           ("n3_uma_los_v3", "Direct\nPath"), ("n3_uma_nlos_v3", "No Direct\nPath")]
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 7.4), gridspec_kw={"width_ratios": [6, 5]})
    for col, (sets, title) in enumerate(((umi, "Small Cell (3GPP UMi)"), (uma, "Macro Cell (3GPP UMa)"))):
        res = {k: [R(s)["all"][k]["resends_per_100"] for s, _ in sets] for k in ORDER}
        rsd = {k: [R(s)["all"][k]["residual_after_all_pct"] for s, _ in sets] for k in ORDER}
        grouped(axes[0][col], [l for _, l in sets], res)
        grouped(axes[1][col], [l for _, l in sets], rsd, digits=2)
        axes[0][col].set_title(f"({'ab'[col]}) {title}: Resends per 100 Blocks", fontsize=FS + 0.5, color=INK)
        axes[1][col].set_title(f"({'cd'[col]}) {title}: Blocks Never Delivered by HARQ (%)", fontsize=FS + 0.5, color=INK)
        style(axes[0][col], "Resends per 100 Blocks" if col == 0 else "")
        style(axes[1][col], "Never Delivered (%)" if col == 0 else "")
    legend(fig, ORDER, y=1.0)
    fig.text(0.5, 0.925, "Above other bars: how many times our resends (or our undelivered blocks) that structure has",
             ha="center", fontsize=FS - 1, color="#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=1.2, w_pad=1.2)
    save(fig, "harq_conditions")


# ---------------------------------------------------------------- 2. robustness: every condition in one forest plot; signal strength
def _summary_rows():
    import csv
    return {r["set"]: r for r in csv.DictReader(open("results/summary/harq.csv"))}


def robustness():
    rows = _summary_rows()
    groups = [("Small Cell (3GPP UMi)", [
        ("n1_umi_v3", "3 km/h (20,480 blocks)"), ("n1_umi_v30", "30 km/h"), ("n1_umi_v60", "60 km/h"), ("n3_umi_v120", "120 km/h"),
        ("p1_umi_los_v3", "Direct path"), ("p1_umi_los_more", "Direct path, new seeds"), ("p1_umi_nlos_v3", "No direct path"),
        ("p1_umi_nlos_more", "No direct path, new seeds"), ("mcs12_umi", "MCS 12"), ("mcs16_umi", "MCS 16"),
        ("wide_umi", "Eb/No 2, 4, 14, 16 dB"), ("n1b_umi_v3_alone", "Resend alone")]),
        ("Macro Cell (3GPP UMa)", [
        ("n3_uma_v3", "3 km/h"), ("n3_uma_v3_more", "3 km/h, new seeds"), ("n3_uma_v30", "30 km/h"), ("n3_uma_v30_more", "30 km/h, new seeds"),
        ("n3_uma_v120", "120 km/h"), ("n3_uma_v120_more", "120 km/h, new seeds"), ("n3_uma_los_v3", "Direct path"),
        ("n3_uma_los_more", "Direct path, new seeds"), ("n3_uma_nlos_v3", "No direct path"), ("n3_uma_nlos_more", "No direct path, new seeds"),
        ("mcs12_uma", "MCS 12"), ("mcs16_uma", "MCS 16"), ("wide_uma", "Eb/No 2, 4, 14, 16 dB"), ("n1b_uma_v3_alone", "Resend alone"),
        ("n1b_uma_v30_alone", "Resend alone, 30 km/h"), ("n1b_uma_los_alone", "Resend alone, direct path"),
        ("n1b_uma_nlos_alone", "Resend alone, no direct path")])]
    items, y, ticks, heads = [], 0, [], []
    for title, sets in groups:
        heads.append((y, title)); y += 1
        for s, lab in sets:
            if s in rows:
                items.append((y, s, lab)); ticks.append((y, lab)); y += 1
        y += 0.6
    fig, ax = plt.subplots(figsize=(10.6, 0.29 * y + 1.4))
    for yy, s, lab in items:
        r = rows[s]
        lo, hi = (float(v) for v in r["resend_ratio_ci"].split("-"))
        ratio = float(r["resend_ratio"])
        small = float(r["resends_small_nrx"]) / float(r["resends_l1"])
        ax.plot([lo, hi], [yy, yy], color=OURS, linewidth=2.2, zorder=3)
        ax.plot(ratio, yy, "o", color=OURS, markersize=7, zorder=4)
        ax.plot(small, yy, "^", markerfacecolor="white", markeredgecolor="#4d9a3a", markersize=7, markeredgewidth=1.5, zorder=4)
        ax.text(1.215, yy, f"{ratio:.2f} [{lo:.2f}-{hi:.2f}]", va="center", ha="left", fontsize=FS - 1.5, color=INK)
    for yy, title in heads:
        ax.text(0.16, yy, title, va="center", ha="left", fontsize=FS + 0.5, color=INK, fontweight="bold")
    ax.axvline(1.0, color="#555555", linewidth=1.2, linestyle="--", zorder=2)
    ax.text(1.012, -0.6, "L1 only = 1", ha="left", va="center", fontsize=FS - 1.5, color="#555555")
    ax.set_yticks([t[0] for t in ticks]); ax.set_yticklabels([t[1] for t in ticks], fontsize=FS - 1)
    ax.set_ylim(y - 0.4, -1.4); ax.set_xlim(0.15, 1.2)
    ax.text(1.215, -0.6, "Ours / L1 [95%]", ha="left", fontsize=FS - 1.5, color=INK, fontweight="bold")
    ax.grid(axis="x", color="#cccccc", linewidth=0.6, linestyle=(0, (4, 3))); ax.set_axisbelow(True)
    ax.tick_params(labelsize=FS - 0.5, colors=INK, length=3)
    ax.set_xlabel("Resends per Block, Relative to the Conventional Receiver (L1) Only  (left = fewer resends)", fontsize=FS, color=INK)
    h = [plt.Line2D([], [], color=OURS, marker="o", linewidth=2.2, markersize=7),
         plt.Line2D([], [], color="white", marker="^", markerfacecolor="white", markeredgecolor="#4d9a3a", markersize=7, markeredgewidth=1.5)]
    fig.legend(h, ["Our Strategy: L1 + NRx (Large Model), Recovery (95% interval over chunks)", "L1 + NRx (Small Model), Every Block"],
               loc="upper center", ncol=2, frameon=False, fontsize=FS - 0.5)
    fig.tight_layout(rect=(0, 0, 0.88, 0.97))
    save(fig, "robustness")

    # signal strength: share of failures recovered and resend ratio by Eb/No (four sets cover 2..16 dB)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))
    for env, sets, mk in (("Small Cell (UMi)", ["wide_umi", "n1_umi_v3"], "o"), ("Macro Cell (UMa)", ["wide_uma", "n3_uma_v3"], "s")):
        pts = {}
        for s in sets:
            for p, v in R(s)["by_point"].items():
                pts[float(p)] = v
        xs = sorted(pts)
        rec = [pts[x]["ours"]["recovered_pct_of_failures"] for x in xs]
        ratio = [pts[x]["ai_and_ran"]["resends_per_100"] / max(pts[x]["ours"]["resends_per_100"], 1e-9) for x in xs]
        c = OURS if "Macro" in env else "#6baed6"
        axes[0].plot(xs, rec, marker=mk, color=c, linewidth=2, label=env)
        axes[1].plot(xs, ratio, marker=mk, color=c, linewidth=2, label=env)
    axes[0].set_title("(a) Failed Blocks Recovered (%)", fontsize=FS + 0.5, color=INK)
    axes[1].set_title("(b) Resends of L1 Only / Resends of Ours", fontsize=FS + 0.5, color=INK)
    style(axes[0], "Recovered (%)", "Eb/No (dB)"); style(axes[1], "Times Our Resends", "Eb/No (dB)")
    axes[1].axhline(1, color="#777777", linewidth=1, linestyle="--")
    axes[0].set_ylim(0, 85); axes[1].set_ylim(0.8, 4.2)
    axes[0].legend(frameon=False, fontsize=FS)
    fig.tight_layout(w_pad=1.5)
    save(fig, "signal_strength")

# ---------------------------------------------------------------- 3. where failures and recoveries are: pair similarity
def similarity():
    edges = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.0), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for sets, name, c, mk in ((["n1_umi_v3"], "Small Cell (UMi)", "#6baed6", "o"), (["n3_uma_v3", "n3_uma_v3_more"], "Macro Cell (UMa)", OURS, "s")):
        conv, rec, rho = [], [], []
        for s in sets:
            for f in sorted(glob.glob(f"results/{s}/c*.npz")):
                d = np.load(f)
                conv.append(d["conv_t0"].reshape(-1)); rec.append(d["nrx_c0"].reshape(-1))
                rho.append(np.repeat(d["corr"][:, 0], d["conv_t0"].shape[1]))
        conv, rec, rho = np.concatenate(conv), np.concatenate(rec), np.concatenate(rho)
        b = np.clip(np.digitize(rho, edges) - 1, 0, len(edges) - 2)
        mid = (edges[:-1] + edges[1:]) / 2
        share = np.array([100.0 * (b == k).mean() for k in range(len(mid))])
        fails = np.array([(~conv & (b == k)).sum() for k in range(len(mid))])
        recp = np.array([100.0 * (rec & ~conv & (b == k)).sum() / max(1, fails[k]) for k in range(len(mid))])
        ok = fails >= 30                                  # bins with at least 30 failed blocks
        axes[0].plot(mid[ok], recp[ok], marker=mk, color=c, linewidth=2.2, markersize=6, label=name)
        axes[1].fill_between(mid, share, step="mid", color=c, alpha=0.35, linewidth=0)
        axes[1].step(mid, share, where="mid", color=c, linewidth=1.5, label=name)
    axes[0].set_title("(a) Failed Blocks Recovered by Our Strategy (%), by Similarity of the Two Users' Channels", fontsize=FS + 0.3, color=INK)
    axes[1].set_title("(b) Share of Blocks (%)", fontsize=FS + 0.3, color=INK)
    style(axes[0], "Recovered (%)"); style(axes[1], "Blocks (%)", "Similarity of the Two Users' Channels (0 = Separable, 1 = Same Direction)")
    axes[0].set_ylim(0, 90); axes[0].legend(frameon=False, fontsize=FS, loc="upper left")
    for ax in axes:
        ax.axvline(0.4, color="#999999", linestyle=":", linewidth=1.2)
    axes[0].text(0.41, 84, "similar pairs (≥ 0.4)", fontsize=FS - 1.5, color="#666666")
    axes[1].set_xlim(0, 1)
    fig.tight_layout(h_pad=1.0)
    save(fig, "similarity")

# ---------------------------------------------------------------- 4. which receiver recovers failed blocks (same slots)
def receivers():
    rx = [("nrx", "NRx (Large\nModel)", OURS, False), ("baseline_lmmse_kbest", "K-best\nDetection", "#8c8c8c", False),
          ("nrx_rt", "NRx (Small\nModel)", SMALL, True), ("lslin_idd3", "Iterative Detection\nand Decoding", "#b0b0b0", False),
          ("baseline_lslin_lmmse@50", "LDPC with\n50 Iterations", "#d0d0d0", False)]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
    for ax, s, t in ((axes[0], "n5_umi_v3", "(a) Small Cell (UMi), 2,797 Failed Blocks"), (axes[1], "n5_uma_v3", "(b) Macro Cell (UMa)")):
        d = R(s)["recovers_pct_of_conventional_failures"]
        bound = d["baseline_perf_csi_kbest"]
        o = d["nrx"]
        for i, (k, lab, c, h) in enumerate(rx):
            v = d[k]
            ax.bar(i, v, 0.7, zorder=3, **kind(c, h))
            ax.text(i, v + 1, f"{v:.1f}" if k == "nrx" else f"{o / v:.1f}×", ha="center", fontsize=FS - 0.5, color=OURS if k == "nrx" else INK)
        ax.axhline(bound, color="#d62728", linestyle="--", linewidth=1.2)
        ax.text(len(rx) - 0.5, bound + 1, f"Perfect Channel + K-best: {bound:.0f}", ha="right", fontsize=FS - 1.5, color="#d62728")
        ax.set_xticks(range(len(rx))); ax.set_xticklabels([r[1] for r in rx], fontsize=FS - 1.5)
        ax.set_ylim(0, 80); ax.set_title(t, fontsize=FS + 0.5, color=INK)
        style(ax, "Failed Blocks Recovered (%)")
    fig.text(0.5, 0.95, "Above other bars: how many times that receiver's recoveries the Large Model makes", ha="center", fontsize=FS - 1, color="#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.5)
    save(fig, "receivers")


# ---------------------------------------------------------------- 5. latency: blocks not delivered within a time
def latency():
    from harq_latency import simulate

    class A: l1, run, deadline, lead = 1.4, 6.3, 11.5, 4.0
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    for ax, s, t in ((axes[0], "n1_umi_v3", "(a) Small Cell (UMi)"), (axes[1], "n3_uma_v3_more", "(b) Macro Cell (UMa)")):
        good, total = {}, None
        for f in sorted(glob.glob(f"results/{s}/c*.npz")):
            d = np.load(f); total = int(d["total"])
            for k in d.files:
                if k.startswith(("conv_", "sep_", "nrx_")):
                    good.setdefault(k, []).append(d[k].reshape(-1))
        good = {k: np.concatenate(v) for k, v in good.items()}
        conv, rec = good["conv_t0"], good["nrx_c0"]
        steps = [None] + [good[f"sep_c{t}"] | good[f"nrx_c{t}"] for t in range(1, total)]
        conv_steps = [None] + [good[f"sep_c{t}"] for t in range(1, total)]
        xs = np.arange(2, 41, 0.5)
        for pol, st, c, ls, lab in (("parallel", steps, OURS, "-", "Our Strategy (Recovery beside HARQ)"),
                                    ("harq_only", conv_steps, L1, "-", "Conventional Receiver (L1) Only"),
                                    ("wait", steps, "#ff7f0e", "--", "Recovery First, Then Resend")):
            done, _ = simulate(conv, rec, st, None, pol, "DDDSU", A)
            ax.plot(xs, [100 * (done > x).mean() for x in xs], color=c, linestyle=ls, linewidth=2.2, label=lab)
        ax.set_yscale("log"); ax.set_ylim(0.3, 20); ax.set_xlim(2, 40)
        ax.set_title(t, fontsize=FS + 0.5, color=INK)
        style(ax, "Blocks Not Yet Delivered (%)", "Time After the First Transmission (ms)")
    axes[0].legend(frameon=False, fontsize=FS - 1)
    fig.tight_layout(w_pad=1.5)
    save(fig, "latency")


# ---------------------------------------------------------------- 6. failed code blocks predict recovery; choosing whom to wait for
def code_blocks():
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.4), gridspec_kw={"width_ratios": [1, 1.1]})
    for s, c, lab in (("n7_cb_v3_lead4", "#6baed6", "Small Cell (UMi)"), ("n7_n3_uma_v3_more_lead4", OURS, "Macro Cell (UMa)")):
        d = R(s)["recovery_by_failed_cbs"]
        ks = sorted(d, key=int)
        axes[0].plot([int(k) for k in ks], [d[k]["recovered_pct"] for k in ks], marker="o", color=c, linewidth=2, label=lab)
    axes[0].set_xticks(range(1, 11)); axes[0].set_ylim(0, 105)
    axes[0].set_title("(a) Recovered by Failed Code Blocks (of 10)", fontsize=FS + 0.5, color=INK)
    style(axes[0], "Failed Blocks Recovered (%)", "Code Blocks the Conventional Receiver Failed")
    axes[0].legend(frameon=False, fontsize=FS)
    pols = [("harq_only", "L1 only"), ("parallel", "Recovery beside HARQ"), ("select_cb5", "Wait if ≤ 5 code blocks\nfailed (ours)"),
            ("wait", "Always wait"), ("select_oracle", "Perfect choice (bound)")]
    ax = axes[1]
    for s, c, m, lab in (("n7_cb_v3_lead4", "#6baed6", "o", "Small Cell (UMi)"), ("n7_n3_uma_v3_more_lead4", OURS, "s", "Macro Cell (UMa)")):
        d = R(s)["policies"]
        base = d["DDDSU/harq_only"]["unrecovered_p99_ms"]
        ys = [d[f"DDDSU/{p}"]["uplink_slots_per_100"] for p, _ in pols]
        ax.plot(ys, range(len(pols)), marker=m, color=c, markersize=10, linestyle="none", zorder=4, label=lab)
        if c == OURS:
            delay = [d[f"DDDSU/{p}"]["unrecovered_p99_ms"] - base for p, _ in pols]
    for i in range(len(pols)):
        ax.axhline(i, color="#eeeeee", linewidth=1, zorder=1)
        ax.text(1.03, i, f"+{delay[i]:.1f} ms" if delay[i] > 0.05 else "0 ms", transform=ax.get_yaxis_transform(), va="center",
                fontsize=FS - 1, color="#d62728" if delay[i] > 0.05 else INK)
    ax.text(1.03, -0.75, "Added\ndelay", transform=ax.get_yaxis_transform(), va="center", fontsize=FS - 2, color="#555555")
    ax.set_yticks(range(len(pols))); ax.set_yticklabels([l for _, l in pols], fontsize=FS - 1)
    ax.get_yticklabels()[2].set_color(OURS)
    ax.set_ylim(len(pols) - 0.5, -0.5); ax.set_xlim(0, 22)
    ax.grid(axis="x", color="#cccccc", linewidth=0.6, linestyle=(0, (4, 3))); ax.set_axisbelow(True)
    ax.tick_params(labelsize=FS - 0.5, colors=INK)
    ax.set_xlabel("Resends per 100 Blocks (Lower Is Better)", fontsize=FS, color=INK)
    ax.set_title("(b) When to Wait for Recovery: Resends and Delay", fontsize=FS + 0.5, color=INK)
    ax.legend(frameon=False, fontsize=FS - 1, loc="lower left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "code_blocks")

# ---------------------------------------------------------------- 7. link adaptation
def link_adaptation():
    sets = [("n6_umi_v3", "(a) Small Cell (UMi)"), ("n6_uma_v3", "(b) Macro Cell (UMa)"), ("n6_uma_nlos", "(c) Macro Cell, No Direct Path")]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), sharey=True)
    for ax, (s, t) in zip(axes, sets):
        d = json.load(open(f"results/{s}_olla.json"))["results"]
        snr = sorted({float(k.split("_")[0][3:]) for k in d})
        for tgt, ls in (("0.1", "-"), ("0.03", "--")):
            ours = [d[f"snr{x:g}_target{tgt}"]["ours"]["goodput_vs_ai_and_ran"] for x in snr]
            small = [d[f"snr{x:g}_target{tgt}"]["ai_for_ran"]["goodput_vs_ai_and_ran"] for x in snr]
            ax.plot(snr, ours, color=OURS, linestyle=ls, marker="o", linewidth=2.2, label=f"Our Strategy, Target {float(tgt):.0%} First Failures")
            ax.plot(snr, small, color=SMALL, linestyle=ls, marker="^", linewidth=1.8, label=f"NRx (Small Model) Every Block, Target {float(tgt):.0%}")
        ax.axhline(1, color=L1, linewidth=1.5)
        ax.set_title(t, fontsize=FS + 0.5, color=INK)
        style(ax, "Throughput / L1 Only" if ax is axes[0] else "", "SNR (dB)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=FS - 0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=1.2)
    save(fig, "link_adaptation")


# ---------------------------------------------------------------- 8. pairing
def pairing():
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3), sharey=False)
    for ax, (pref, title, minrho) in zip(axes, (("p2b_u", "(a) Small Cell (UMi)", [8, 16]), ("p2b_uma_u", "(b) Macro Cell (UMa)", [8]))):
        xs = [1, 4, 8, 16]
        sets = [f"{pref}8_random", f"{pref}4_partner", f"{pref}8_partner", f"{pref}16_partner"]
        l1 = [R(s)["all"]["ai_and_ran"]["resends_per_100"] for s in sets]
        ours = [R(s)["all"]["ours"]["resends_per_100"] for s in sets]
        ax.plot(xs, l1, marker="o", color=L1, linewidth=2.4, markersize=8, label="Conventional Receiver (L1) Only")
        ax.plot(xs, ours, marker="o", color=OURS, linewidth=2.4, markersize=8, label="Our Strategy (Recovery)")
        ax.axhline(ours[0], color=OURS, linestyle=":", linewidth=1.4)
        ax.text(1.5, ours[0] + 0.25, "ours with a random partner", va="bottom", fontsize=FS - 2, color=OURS)
        for k in minrho:
            s = f"{pref}{k}_minrho"
            ax.plot(k, R(s)["all"]["ai_and_ran"]["resends_per_100"], marker="D", markerfacecolor="white", markeredgecolor=L1,
                    markeredgewidth=1.6, markersize=8, linestyle="none")
            ax.plot(k, R(s)["all"]["ours"]["resends_per_100"], marker="D", markerfacecolor="white", markeredgecolor=OURS,
                    markeredgewidth=1.6, markersize=8, linestyle="none")
        ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels(["1\n(random)", "4", "8", "16"])
        ax.set_xlim(0.8, 22)
        ax.set_title(title + ": Resends per 100 Blocks", fontsize=FS + 0.5, color=INK)
        style(ax, "Resends per 100 Blocks", "Users the Scheduler Chooses User 0's Partner From")
        ax.set_ylim(0, max(l1 + [R(f"{pref}{k}_minrho")["all"]["ai_and_ran"]["resends_per_100"] for k in minrho]) * 1.12)
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="D", markerfacecolor="white", markeredgecolor="#555555", markeredgewidth=1.6, linestyle="none", markersize=8))
    l.append("Partner Chosen by Similarity Only (Hollow)")
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=FS - 0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.5)
    save(fig, "pairing")

# ---------------------------------------------------------------- 9. more users, more antennas
def users_antennas():
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))
    u = R("n8a_users")
    vals = [u[str(k)]["fail_pct"] for k in (1, 2, 3, 4)]
    axes[0].bar(range(4), vals, 0.6, zorder=3, **kind(L1, False))
    for i, v in enumerate(vals):
        axes[0].text(i, v + 1, f"{v:.0f}", ha="center", fontsize=FS)
    axes[0].set_xticks(range(4)); axes[0].set_xticklabels(["1", "2", "3", "4"])
    axes[0].set_title("(a) L1 Failures by Users on the Same Resources", fontsize=FS, color=INK)
    style(axes[0], "First Transmissions Failed (%)", "Users Sharing the Resources")
    rx = [("nrx", "NRx (Large Model)", OURS, False), ("baseline_lmmse_kbest", "K-best Detection", "#8c8c8c", False),
          ("baseline_perf_csi_kbest", "Perfect Channel + K-best (Bound)", "white", True)]
    for ax, sets, t in ((axes[1], [("n5_umi_v3", "2 Users\n(Original)"), ("n8b_u3_kbest", "3 Users\n(Fine-Tuned)"), ("n8b_u4_kbest", "4 Users\n(Fine-Tuned)")], "(b) More Users (4 Antennas, UMi)"),
                        (axes[2], [("n5_umi_v3", "4 Antennas\nSmall Cell"), ("n4b_eval_r2", "8 Antennas\nSmall Cell"), ("n5_uma_v3", "4 Antennas\nMacro Cell"), ("n4b_eval_r2_uma", "8 Antennas\nMacro Cell")], "(c) More Antennas (2 Users)")):
        for g, (s, _) in enumerate(sets):
            d = R(s)["recovers_pct_of_conventional_failures"]
            for j, (k, lab, c, h) in enumerate(rx):
                if k not in d:
                    continue
                ax.bar(g + (j - 1) * 0.27, d[k], 0.25, zorder=3, **kind("#d62728" if h else c, h))
                ax.text(g + (j - 1) * 0.27, d[k] + 1, f"{d[k]:.0f}", ha="center", fontsize=FS - 2.5, color=OURS if k == "nrx" else INK)
        ax.set_xticks(range(len(sets))); ax.set_xticklabels([l for _, l in sets], fontsize=FS - 1.5)
        ax.set_ylim(0, 95); ax.set_title(t, fontsize=FS, color=INK)
        style(ax, "Failed Blocks Recovered (%)")
    h = [plt.Rectangle((0, 0), 1, 1, **kind(OURS, False)), plt.Rectangle((0, 0), 1, 1, **kind("#8c8c8c", False)),
         plt.Rectangle((0, 0), 1, 1, **kind("#d62728", True))]
    fig.legend(h, [r[1] for r in rx], loc="upper center", ncol=3, frameon=False, fontsize=FS)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.3)
    save(fig, "users_antennas")


# ---------------------------------------------------------------- 10. GPU: throughput of one A100 and cells it can recover for
def gpu():
    d = R("n13_lanes")
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))
    for name, c, mk in (("large", OURS, "o"), ("small", SMALL, "^")):
        k = [r["lanes"] for r in d[name]]
        axes[0].plot(k, [r["runs_per_s"] for r in d[name]], marker=mk, color=c, linewidth=2, label=f"NRx ({name.title()} Model)")
        axes[1].plot(k, [r["latency_ms_p50"] for r in d[name]], marker=mk, color=c, linewidth=2)
    axes[0].set_title("(a) Runs per Second on One A100", fontsize=FS, color=INK)
    axes[1].set_title("(b) Time per Run (ms, Median)", fontsize=FS, color=INK)
    style(axes[0], "Runs per Second", "Copies Running at Once"); style(axes[1], "ms", "Copies Running at Once")
    axes[0].legend(frameon=False, fontsize=FS); axes[0].set_ylim(0, 800)
    t_large = 1.0 / d["large"][0]["runs_per_s"]
    pred = R("n10_predictor")
    for key, need, c, lab in (("n1_umi_v3_cb", R("n9_demand_v3")["by_point"]["8.0"]["p_slot_needs_recovery"], "#6baed6", "Small Cell (UMi)"),
                              ("n3_uma_v3_more", R("n9_demand_n3_uma_v3")["by_point"]["all"]["p_slot_needs_recovery"], OURS, "Macro Cell (UMa)")):
        fr = [float(f) for f in pred[key]["budget_trees"]]
        kept = [100 * v for v in pred[key]["budget_trees"].values()]
        cells = [1.0 / (400.0 * need * f * t_large) for f in fr]
        axes[2].plot(kept, cells, marker="o", color=c, linewidth=2.2, label=lab)
        for f, x, y in zip(fr, kept, cells):
            if f in (0.1, 0.3, 1.0):
                up = c != OURS
                axes[2].annotate(f"{f:.0%}", (x, y), xytext=(7, 4) if up else (-7, -11), textcoords="offset points",
                                 ha="left" if up else "right", fontsize=FS - 2, color="#555555")
    axes[2].set_yscale("log"); axes[2].set_xlim(0, 112); axes[2].set_ylim(1.9, 40); axes[2].set_xticks([0, 20, 40, 60, 80, 100])
    axes[2].yaxis.set_major_locator(matplotlib.ticker.FixedLocator([2, 3, 4, 6, 10, 20, 30]))
    axes[2].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axes[2].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[2].text(0.03, 0.04, "label = share of failures\nsent to the Large Model", transform=axes[2].transAxes, fontsize=FS - 2, color="#555555")
    axes[2].set_title("(c) Cells per A100 against Recovery Value Kept", fontsize=FS, color=INK)
    style(axes[2], "Cells One A100 Recovers For", "Recoverable Blocks Recovered (%)")
    axes[2].legend(frameon=False, fontsize=FS - 1)
    fig.tight_layout(w_pad=1.3)
    save(fig, "gpu")

# ---------------------------------------------------------------- 12. predicting recoverable blocks; GPU budget
def predictor():
    d = R("n10_predictor")
    sets = [k for k in d if k != "pooled"]
    lab = {"n1_umi_v3_cb": "UMi 3 km/h", "n3_umi_v120": "UMi 120 km/h", "mcs12_umi": "UMi MCS 12", "mcs16_umi": "UMi MCS 16",
           "wide_umi": "UMi wide Eb/No", "p1_umi_los_more": "UMi direct path", "p1_umi_nlos_more": "UMi no direct path",
           "n3_uma_v3_more": "UMa 3 km/h", "n3_uma_v30_more": "UMa 30 km/h", "n3_uma_v120_more": "UMa 120 km/h",
           "n3_uma_los_more": "UMa direct path", "n3_uma_nlos_more": "UMa no direct path", "mcs12_uma": "UMa MCS 12",
           "mcs16_uma": "UMa MCS 16", "wide_uma": "UMa wide Eb/No"}
    order = sorted(sets, key=lambda s: (s.split("_")[0] != "n1" and "uma" in s, d[s]["auc_trees"] - d[s]["auc_cbfail"]))
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    for yy, s in enumerate(order):
        a, b = d[s]["auc_cbfail"], d[s]["auc_trees"]
        axes[0].plot([a, b], [yy, yy], color="#bbbbbb", linewidth=2.5, zorder=2)
        axes[0].plot(a, yy, "o", color=L1, markersize=8, zorder=3)
        axes[0].plot(b, yy, "o", color=OURS, markersize=8, zorder=4)
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([lab.get(s, s) for s in order], fontsize=FS - 1)
    axes[0].set_xlim(0.55, 1.0); axes[0].invert_yaxis()
    axes[0].grid(axis="x", color="#cccccc", linewidth=0.6, linestyle=(0, (4, 3))); axes[0].set_axisbelow(True)
    axes[0].tick_params(labelsize=FS - 0.5, colors=INK)
    axes[0].set_xlabel("AUC of Predicting Recovery When L1 Finishes (0.5 = Guessing)", fontsize=FS, color=INK)
    axes[0].set_title("(a) Failed Code Blocks Only (Gray) vs. Predictor (Blue)", fontsize=FS + 0.5, color=INK)
    fr = [0.0] + [float(k) for k in d["pooled"]["budget_trees"]]
    for key, c, name in (("pooled", OURS, "Predictor, All 15 Conditions"), ("n1_umi_v3_cb", "#6baed6", "Predictor, UMi 3 km/h"),
                         ("n3_uma_v3_more", "#08306b", "Predictor, UMa 3 km/h")):
        axes[1].plot(fr, [0.0] + [100 * v for v in d[key]["budget_trees"].values()], color=c, marker="o", linewidth=2, label=name)
    axes[1].plot(fr, [0.0] + [100 * v for v in d["pooled"]["budget_cbfail"].values()], color=L1, marker="s", linewidth=1.8, label="Failed Code Blocks Only, All")
    axes[1].plot([0, 1], [0, 100], color="#777777", linestyle="--", linewidth=1.2, label="Random Choice")
    axes[1].set_xticks([0, .2, .4, .6, .8, 1.0]); axes[1].set_xticklabels(["0", "20", "40", "60", "80", "100"])
    axes[1].set_title("(b) Recoverable Blocks Captured vs. GPU Spent", fontsize=FS + 0.5, color=INK)
    style(axes[1], "Recoverable Blocks Captured (%)", "Failed Blocks Given to the Large Model (%)")
    axes[1].legend(frameon=False, fontsize=FS - 2, loc="lower right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "predictor")

# ---------------------------------------------------------------- 13. inter-cell interference: IRC baseline, published vs interference-aware receiver
def interference():
    xs, xl = [0, 3, 6, 10], ["None", "3 dB", "6 dB", "10 dB"]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.0), sharex=True)
    for col, ch in enumerate(("umi", "uma")):
        ratio = {"pub": [], "aware": []}; rec = {"pub": [], "aware": []}
        for inr in ("none", "3", "6", "10"):
            pub = R(f"irc_{ch}_v3" if inr == "none" else f"ici_{ch}_inr{inr}")["all"]
            aw = R(f"ici_{ch}_v3_iciaware" if inr == "none" else f"ici_{ch}_inr{inr}_iciaware")["all"]
            irc = pub["ai_and_ran_irc"]["resends_per_100"]
            ratio["pub"].append(pub["ours_irc"]["resends_per_100"] / irc); ratio["aware"].append(aw["ours_irc"]["resends_per_100"] / irc)
            rec["pub"].append(pub["ours_irc"]["recovered_pct_of_failures"]); rec["aware"].append(aw["ours_irc"]["recovered_pct_of_failures"])
        for k, c, lab in (("aware", OURS, "IRC + NRx (Large Model), Trained with Interference"), ("pub", "#9ecae1", "IRC + NRx (Large Model), Published")):
            axes[0][col].plot(xs, ratio[k], marker="o", color=c, linewidth=2.4, markersize=8, label=lab)
            axes[1][col].plot(xs, rec[k], marker="o", color=c, linewidth=2.4, markersize=8, label=lab)
            other = ratio["pub" if k == "aware" else "aware"]
            for x, v, o in zip(xs, ratio[k], other):
                axes[0][col].annotate(f"{v:.2f}", (x, v), xytext=(0, 7 if v >= o else -15), textcoords="offset points", ha="center",
                                      fontsize=FS - 2, color=INK)
        axes[0][col].axhline(1.0, color=L1, linewidth=2.2)
        axes[0][col].text(-0.4, 1.01, "IRC only", ha="left", va="bottom", fontsize=FS - 1.5, color="#666666")
        axes[0][col].set_ylim(0.45, 1.08)
        name = "Small Cell (UMi)" if ch == "umi" else "Macro Cell (UMa)"
        axes[0][col].set_title(f"({'ab'[col]}) {name}: Resends Relative to IRC Only", fontsize=FS, color=INK)
        axes[1][col].set_title(f"({'cd'[col]}) {name}: Failed Blocks of IRC Recovered (%)", fontsize=FS, color=INK)
        style(axes[0][col], "Resends / IRC Only" if col == 0 else "")
        style(axes[1][col], "Recovered (%)" if col == 0 else "", "Interference over Thermal Noise (Nominal)")
        axes[1][col].set_ylim(0, max(rec["pub"] + rec["aware"]) * 1.2)
        axes[1][col].set_xticks(xs); axes[1][col].set_xticklabels(xl)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=FS - 0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.2, w_pad=1.2)
    save(fig, "interference")

for f in (interference, predictor, harq_conditions, robustness, similarity, receivers, latency, code_blocks, link_adaptation, pairing, users_antennas, gpu):
    try:
        f()
    except Exception as e:                       # one broken figure must not stop the others
        print("FAILED", f.__name__, repr(e))

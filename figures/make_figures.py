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


# ---------------------------------------------------------------- 2. robustness: retransmissions alone, code rates, signal strength
def robustness():
    alone = [("n1_umi_v3", "Small Cell\nSame Pair"), ("n1b_umi_v3_alone", "Small Cell\nResend Alone"),
             ("n3_uma_v3", "Macro Cell\nSame Pair"), ("n1b_uma_v3_alone", "Macro Cell\nResend Alone"),
             ("n3_uma_nlos_v3", "Macro, Blocked\nSame Pair"), ("n1b_uma_nlos_alone", "Macro, Blocked\nResend Alone")]
    mcs = [("mcs12_umi", "Small Cell\nMCS 12"), ("n1_umi_v3", "Small Cell\nMCS 14"), ("mcs16_umi", "Small Cell\nMCS 16"),
           ("mcs12_uma", "Macro Cell\nMCS 12"), ("n3_uma_v3", "Macro Cell\nMCS 14"), ("mcs16_uma", "Macro Cell\nMCS 16")]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.6))
    for ax, sets, t in ((axes[0], alone, "(a) Resend with the Same Partner or Alone"), (axes[1], mcs, "(b) Other Code Rates")):
        res = {k: [R(s)["all"][k]["resends_per_100"] for s, _ in sets] for k in ORDER}
        grouped(ax, [l for _, l in sets], res)
        ax.set_title(t, fontsize=FS + 0.5, color=INK)
        style(ax, "Resends per 100 Blocks")
        ax.tick_params(axis="x", labelsize=FS - 2)
    legend(fig, ORDER, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.5)
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
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    bins = ["0.0-0.4", "0.4-0.6", "0.6-1.0"]
    labels = ["Separable\n(< 0.4)", "Similar\n(0.4-0.6)", "Very Similar\n(≥ 0.6)"]
    for ax, s, t in ((axes[0], "n1_umi_v3", "(a) Small Cell (UMi)"), (axes[1], "n3_uma_v3_more", "(b) Macro Cell (UMa)")):
        d = R(s)
        share = [100 * d["similarity_share"][b] for b in bins]
        rec = [d["by_similarity"][b]["ours"]["recovered_pct_of_failures"] for b in bins]
        x = np.arange(3)
        ax.bar(x - 0.2, share, 0.38, zorder=3, **kind(L1, False))
        ax.bar(x + 0.2, rec, 0.38, zorder=3, **kind(OURS, False))
        for i in range(3):
            ax.text(x[i] - 0.2, share[i] + 1, f"{share[i]:.0f}", ha="center", fontsize=FS - 1)
            ax.text(x[i] + 0.2, rec[i] + 1, f"{rec[i]:.0f}", ha="center", fontsize=FS - 1, color=OURS)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=FS - 0.5)
        ax.set_ylim(0, 80); ax.set_title(t, fontsize=FS + 0.5, color=INK)
        style(ax, "Percent", "Similarity of the Two Users' Channels")
    h = [plt.Rectangle((0, 0), 1, 1, **kind(L1, False)), plt.Rectangle((0, 0), 1, 1, **kind(OURS, False))]
    fig.legend(h, ["Share of Blocks (%)", "Failed Blocks Recovered by Our Strategy (%)"], loc="upper center", ncol=2,
               frameon=False, fontsize=FS)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.5)
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
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2))
    for s, c, lab in (("n7_cb_v3_lead4", "#6baed6", "Small Cell (UMi)"), ("n7_n3_uma_v3_more_lead4", OURS, "Macro Cell (UMa)")):
        d = R(s)["recovery_by_failed_cbs"]
        ks = sorted(d, key=int)
        axes[0].plot([int(k) for k in ks], [d[k]["recovered_pct"] for k in ks], marker="o", color=c, linewidth=2, label=lab)
    axes[0].set_xticks(range(1, 11)); axes[0].set_ylim(0, 105)
    axes[0].set_title("(a) Recovered by Failed Code Blocks (of 10)", fontsize=FS + 0.5, color=INK)
    style(axes[0], "Failed Blocks Recovered (%)", "Code Blocks the Conventional Receiver Failed")
    axes[0].legend(frameon=False, fontsize=FS)
    pols = [("harq_only", "L1 Only", L1), ("parallel", "Recovery\nbeside HARQ", "#6baed6"), ("select_cb5", "Wait if ≤ 5\nFailed CBs", OURS),
            ("wait", "Always Wait\n(+5 ms)", "#ff7f0e"), ("select_oracle", "Perfect\nChoice", "#d0d0d0")]
    x = np.arange(len(pols))
    for j, (s, lab) in enumerate((("n7_cb_v3_lead4", "Small Cell"), ("n7_n3_uma_v3_more_lead4", "Macro Cell"))):
        d = R(s)["policies"]
        vals = [d[f"DDDSU/{p}"]["uplink_slots_per_100"] for p, _, _ in pols]
        for i, (p, _, c) in enumerate(pols):
            axes[1].bar(i + (j - 0.5) * 0.38, vals[i], 0.36, zorder=3, **kind(c, j == 1))
            axes[1].text(i + (j - 0.5) * 0.38, vals[i] + 0.3, f"{vals[i]:.1f}", ha="center", fontsize=FS - 2.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels([p[1] for p in pols], fontsize=FS - 1.5)
    axes[1].set_title("(b) Resends per 100 Blocks (Solid: Small Cell, Hatched: Macro Cell)", fontsize=FS, color=INK)
    style(axes[1], "Resends per 100 Blocks")
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
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
    for ax, sets, t in ((axes[0], [("p2b_u8_random", "Random\nPartner"), ("p2b_u4_partner", "Best of 4"), ("p2b_u8_partner", "Best of 8"),
                                   ("p2b_u16_partner", "Best of 16"), ("p2b_u8_minrho", "Least Similar\nof 8")], "(a) Small Cell (UMi)"),
                        (axes[1], [("p2b_uma_u8_random", "Random\nPartner"), ("p2b_uma_u4_partner", "Best of 4"), ("p2b_uma_u8_partner", "Best of 8"),
                                   ("p2b_uma_u16_partner", "Best of 16"), ("p2b_uma_u8_minrho", "Least Similar\nof 8")], "(b) Macro Cell (UMa)")):
        keys = ["ours", "ai_and_ran"]
        res = {k: [R(s)["all"][k]["resends_per_100"] for s, _ in sets] for k in keys}
        grouped(ax, [l for _, l in sets], res, keys=keys)
        ax.axhline(res["ours"][0], color=OURS, linestyle=":", linewidth=1.3)
        ax.set_title(t + ": Resends per 100 Blocks", fontsize=FS + 0.5, color=INK)
        style(ax, "Resends per 100 Blocks", "How the Scheduler Picks User 0's Partner")
    legend(fig, ["ours", "ai_and_ran"], y=1.0)
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
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    for name, c, mk in (("large", OURS, "o"), ("small", SMALL, "^")):
        k = [r["lanes"] for r in d[name]]
        axes[0].plot(k, [r["runs_per_s"] for r in d[name]], marker=mk, color=c, linewidth=2, label=f"NRx ({name.title()} Model)")
        axes[1].plot(k, [r["latency_ms_p50"] for r in d[name]], marker=mk, color=c, linewidth=2)
    axes[0].set_title("(a) Runs per Second on One A100", fontsize=FS, color=INK)
    axes[1].set_title("(b) Time per Run (ms, Median)", fontsize=FS, color=INK)
    style(axes[0], "Runs per Second", "Copies Running at Once"); style(axes[1], "ms", "Copies Running at Once")
    axes[0].legend(frameon=False, fontsize=FS); axes[0].set_ylim(0, 800)
    large = d["large"][0]["runs_per_s"]; small_t = 1.0 / d["small"][0]["runs_per_s"]; large_t = 1.0 / large
    cas = R("n12_union")
    rows = []
    for env, need, frac in (("Small Cell", R("n9_demand_v3")["by_point"]["8.0"]["p_slot_needs_recovery"], cas["n5_umi"]["cascade_small_then_large"]["large_runs_per_failure"]),
                            ("Macro Cell", R("n9_demand_n3_uma_v3")["by_point"]["all"]["p_slot_needs_recovery"], cas["n5_uma"]["cascade_small_then_large"]["large_runs_per_failure"])):
        per_cell = 400.0 * need                                  # recoveries per second per cell (one uplink slot per 2.5 ms)
        rows.append((env, 1.0 / (per_cell * large_t), 1.0 / (per_cell * (small_t + frac * large_t))))
    x = np.arange(len(rows))
    axes[2].bar(x - 0.2, [r[1] for r in rows], 0.38, zorder=3, **kind(L1, False))
    axes[2].bar(x + 0.2, [r[2] for r in rows], 0.38, zorder=3, **kind(OURS, False))
    for i, r in enumerate(rows):
        axes[2].text(i - 0.2, r[1] + 0.05, f"{r[1]:.1f}", ha="center", fontsize=FS)
        axes[2].text(i + 0.2, r[2] + 0.05, f"{r[2]:.1f} ({r[2] / r[1]:.1f}×)", ha="center", fontsize=FS, color=OURS)
    axes[2].set_xticks(x); axes[2].set_xticklabels([r[0] for r in rows])
    axes[2].set_ylim(0, max(r[2] for r in rows) * 1.3)
    axes[2].set_title("(c) Cells One A100 Recovers For (All Failures)", fontsize=FS, color=INK)
    style(axes[2], "Cells per GPU")
    h = [plt.Rectangle((0, 0), 1, 1, **kind(L1, False)), plt.Rectangle((0, 0), 1, 1, **kind(OURS, False))]
    axes[2].legend(h, ["Large Model on Every Failure", "Small Model First, Large if It Fails"], frameon=False, fontsize=FS - 1.5, loc="upper left")
    fig.tight_layout(w_pad=1.3)
    save(fig, "gpu")


for f in (harq_conditions, robustness, similarity, receivers, latency, code_blocks, link_adaptation, pairing, users_antennas, gpu):
    try:
        f()
    except Exception as e:                       # one broken figure must not stop the others
        print("FAILED", f.__name__, repr(e))

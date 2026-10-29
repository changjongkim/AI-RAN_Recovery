#!/usr/bin/env python3
"""Numbers of the Reverb paper: every number the manuscript states is a LaTeX macro written here from the stored
results, so the text never carries a typed number (paper repo rule 2.1).  The paper repo copies the output with
scripts/update_numbers.sh; a rerun must reproduce the file byte for byte.

  python3 src/analysis/paper_numbers.py        (from the repository root; numpy only)
  outputs: results/paper/numbers.tex (macros), results/paper/numbers.json (values, sources, checks)

Macros (value, source):
  \\RetxCutFullWait     retransmissions per 100 TBs, conventional receiver only / Reverb waiting for every recovery,
                       macro cell, the 20,480 TBs of n3_uma_v3_cb + n3_uma_v3_more (results/n3_uma_v3_all.json)
  \\RetxCutNoDelay      the same with the latency-neutral rule (wait only when at most 5 of 10 code blocks failed),
                       from the latency simulation (src/sim/harq_latency.py, DDDSU, L1 1.4 ms, recovery 6.3 ms,
                       recovery deadline 11.5 ms, lead 4.0 ms) on the same traces; checked: no added p99 delay for
                       the TBs recovery does not decode
  \\LostCutFullWait     TBs never received after four transmissions, conventional / Reverb waiting for every recovery
  \\ThroughputGainLA    goodput under outer-loop link adaptation (target 10%), Reverb / conventional, mean over the
                       SNR points 6-16 dB, macro cell (results/n6_uma_v3_olla.json)
  \\PairingCutBestSixteen  retransmissions per 100 TBs, conventional receiver with the best partner among 16
                       candidates by MMSE sum rate (results/p2b_uma_u16_pf.json) / Reverb with a random partner
                       (results/p2b_uma_u8_random.json), macro cell, the same seeds
Setup macros (text, Evaluation Setup): \\SetupLOneMs, \\SetupRecoveryRunMs, \\SetupRecoveryDeadlineMs, \\SetupLeadMs (the
  timing of the latency simulation, class Timing), \\SetupSlots and \\SetupTBs (slots and TBs of the macro traces),
  \\SetupTbBytes (TB size), \\SetupTransmissions, \\SetupRetxGapMs (channel time between transmissions), \\SetupEbNoLow
  and \\SetupEbNoHigh (Eb/No points), all read from the macro traces of results/n3_uma_v3_all.json and its chunks
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src/sim")
from harq_latency import simulate  # noqa: E402

OUT = "results/paper"
MACRO_SETS = ["n3_uma_v3_cb", "n3_uma_v3_more"]


class Timing:
    l1, run, deadline, lead = 1.4, 6.3, 11.5, 4.0


SUFFIX = os.environ.get("RESULT_SUFFIX", "")   # "_f35": the 3.5 GHz runs (motivation macros; the others follow later)


def stats(names, structure="ai_and_ran"):
    rows = [json.load(open(f"results/{n}{SUFFIX}.json"))["all"][structure] for n in names]
    w = np.array([r["tbs"] for r in rows], float)
    return {k: float(np.dot(w, [r[k] for r in rows]) / w.sum()) for k in ("resends_per_100", "residual_after_all_pct")}


def per_slot(names):
    """Per TB: correlation of the pair, conventional result, recovery result, failed code blocks; per slot: any failure."""
    corr, conv, rec, cb, slot_fail = [], [], [], [], []
    for n in names:
        for f in sorted(glob.glob(f"results/{n}{SUFFIX}/c*.npz")):
            z = np.load(f)
            u = z["conv_t0"].shape[1]
            corr.append(np.repeat(z["corr"][:, 0], u)); conv.append(z["conv_t0"].reshape(-1))
            rec.append(z["nrx_c0"].reshape(-1)); cb.append(z["cbfail_t0"].reshape(-1))
            slot_fail.append(~z["conv_t0"].all(axis=1))
    return [np.concatenate(x) for x in (corr, conv, rec, cb, slot_fail)]


def motivation(lat):
    """Macros of the Motivational Evaluation, from the same results as the panels of figures/paper/make_paper_figures.py."""
    v = {}
    macro, small = ["n3_uma_v3_cb", "n3_uma_v3_more"], ["n1_umi_v3_cb"]
    for tag, names in (("Macro", macro), ("Small", small)):
        corr, conv, rec, cb, slot_fail = per_slot(names)
        f = ~conv
        v[f"MotFirstFail{tag}"] = (100 * f.mean(), 1)
        v[f"MotRecovered{tag}"] = (100 * (rec & f).sum() / f.sum(), 0)
        v[f"MotCorrFailLow{tag}"] = (100 * f[corr < 0.2].mean(), 0)
        v[f"MotCorrFailHigh{tag}"] = (100 * f[corr >= 0.8].mean(), 0)
        v[f"MotFewCb{tag}"] = (100 * rec[f & (cb >= 1) & (cb <= 3)].mean(), 0)
        v[f"MotAllCb{tag}"] = (100 * rec[f & (cb == 10)].mean(), 0)
        if tag == "Macro":
            v["MotAllCbShare"] = (100 * (f & (cb == 10)).sum() / f.sum(), 0)
            v["MotSlotsWithFailure"] = (100 * slot_fail.mean(), 0)
    conds = {"": (["n3_uma_v3_cb", "n3_uma_v3_more"], ["n1_umi_v3_cb"]),
             "Direct": (["n3_uma_los_v3", "n3_uma_los_more"], ["p1_umi_los_v3", "p1_umi_los_more"]),
             "Blocked": (["n3_uma_nlos_v3", "n3_uma_nlos_more"], ["p1_umi_nlos_v3", "p1_umi_nlos_more"])}
    for c, (m, s) in conds.items():
        sm, ss = stats(m), stats(s)
        v[f"MotRetx{c}Macro"] = (sm["resends_per_100"], 1); v[f"MotRetx{c}Small"] = (ss["resends_per_100"], 1)
        v[f"MotLost{c}Macro"] = (sm["residual_after_all_pct"], 1); v[f"MotLost{c}Small"] = (ss["residual_after_all_pct"], 1)
    reps = [json.load(open(f"results/{n}{SUFFIX}.json")) for n in ("n5_uma_v3", "n5_uma_more1", "n5_uma_more2", "n5_uma_more3")]
    w = np.array([r["fail_pct"]["baseline_lslin_lmmse"] * r["slots"] for r in reps])
    for name, key in (("Iterations", "baseline_lslin_lmmse@50"), ("Idd", "lslin_idd3"), ("Kbest", "baseline_lmmse_kbest"),
                      ("SmallNrx", "nrx_rt"), ("LargeNrx", "nrx"), ("Bound", "baseline_perf_csi_kbest")):
        v[f"MotRx{name}"] = (float(np.dot(w, [r["recovers_pct_of_conventional_failures"][key] for r in reps]) / w.sum()), 0)
    lanes = json.load(open("results/n13_lanes.json"))
    large = {r["lanes"]: r for r in lanes["large"] if r["lanes"] <= 4}
    v["MotTimeSmallNrx"] = (lanes["small"][0]["latency_ms_mean"], 1)
    v["MotTimeLargeNrx"] = (large[1]["latency_ms_mean"], 1)
    v["MotTimeLargeFour"] = (large[4]["latency_ms_mean"], 1)
    v["MotRunsMin"] = (min(r["runs_per_s"] for r in large.values()), 0)
    v["MotRunsMax"] = (max(r["runs_per_s"] for r in large.values()), 0)
    v["MotGpuTimePerSlot"] = (v["MotSlotsWithFailure"][0] / 100 * large[1]["latency_ms_mean"], 1)
    v["MotRecoveryResultMs"] = (Timing.l1 + Timing.run, 1)
    v["MotFirstRetxMs"] = (lat["first_retx_ms"], 1)
    v["MotWaitDelayMs"] = (lat["wait_delay_ms"], 1)
    v["MotPNinetyNineNoWait"] = (lat["harq_only"]["p99_unrecovered_ms"], 1)
    v["MotPNinetyNineWait"] = (lat["wait"]["p99_unrecovered_ms"], 1)
    v["MotRetxWait"] = (lat["wait"]["retx_per_100"], 1)
    rows = {}
    for line in open("results/n16_summary.txt"):
        f = line.split()
        if len(f) >= 10 and f[0] in ("allb68", "allb68g4"):
            rows.setdefault((f[0], int(f[2])), []).append(float(f[9]))
    g = lambda pol, c: float(np.mean(rows[(pol, c)]))
    v["MotGpuOneCell"] = (g("allb68", 1), 0); v["MotGpuOneEight"] = (g("allb68", 8), 0)
    v["MotGpuFourEight"] = (g("allb68g4", 8), 0); v["MotGpuFourMax"] = (g("allb68g4", 24), 0)
    red = lambda n: stats([n])["resends_per_100"] / stats([n], "ours")["resends_per_100"]
    v["MotIciPubNone"] = (red("irc_uma_v3"), 1); v["MotIciPubTen"] = (red("ici_uma_inr10"), 1)
    v["MotIciTrainedNone"] = (red("ici_uma_v3_iciaware"), 1); v["MotIciTrainedTen"] = (red("ici_uma_inr10_iciaware"), 1)
    return v


def load_traces(sets):
    parts = []
    for s in sets:
        good, total = {}, None
        for f in sorted(glob.glob(f"results/{s}/c*.npz")):
            d = np.load(f)
            total = int(d["total"])
            for k in d.files:
                if k.startswith(("conv_", "sep_", "nrx_")):
                    good.setdefault(k, []).append(d[k].reshape(-1))
            good.setdefault("cbfail", []).append(d["cbfail_t0"].reshape(-1))
        parts.append(({k: np.concatenate(v) for k, v in good.items()}, total))
    total = parts[0][1]

    def cat(k):
        return np.concatenate([g[k] for g, _ in parts])
    conv, rec, cbfail = cat("conv_t0"), cat("nrx_c0"), cat("cbfail")
    steps = [None] + [cat(f"sep_c{t}") | cat(f"nrx_c{t}") for t in range(1, total)]
    conv_steps = [None] + [cat(f"sep_c{t}") for t in range(1, total)]
    return conv, rec, steps, conv_steps, (cbfail >= 0) & (cbfail <= 5)


def latency_policies():
    conv, rec, steps, conv_steps, cand = load_traces(MACRO_SETS)
    out = {}
    dones = {}
    for pol in ("harq_only", "wait", "select_cb5"):
        done, tx = simulate(conv, rec, conv_steps if pol == "harq_only" else steps, cand, pol, "DDDSU", Timing)
        unrec = ~conv & ~(rec & (Timing.l1 + Timing.run <= Timing.deadline)) & np.isfinite(done)
        dones[pol] = (done, unrec)
        out[pol] = {"retx_per_100": float(100.0 * (tx - 1).sum() / len(tx)),
                    "p99_unrecovered_ms": float(np.percentile(done[unrec], 99)),
                    "tbs": int(len(tx))}
    (d0, u0), (d1, u1) = dones["harq_only"], dones["wait"]
    out["first_retx_ms"] = float(d0[u0].min())                      # the first retransmission is decoded
    out["wait_delay_ms"] = float(d1[u1].min() - d0[u0].min())      # waiting moves it by this much
    return out


def main():
    allstats = json.load(open("results/n3_uma_v3_all.json"))["all"]
    lat = latency_policies()
    olla = json.load(open("results/n6_uma_v3_olla.json"))["results"]
    snr = sorted({float(k.split("_")[0][3:]) for k in olla})
    goodput = [olla[f"snr{x:g}_target0.1"]["ours"]["goodput_vs_ai_and_ran"] for x in snr]
    pf16 = json.load(open("results/p2b_uma_u16_pf.json"))["all"]
    rand = json.load(open("results/p2b_uma_u8_random.json"))["all"]

    conv_retx, ours_retx = allstats["ai_and_ran"]["resends_per_100"], allstats["ours"]["resends_per_100"]
    added = lat["select_cb5"]["p99_unrecovered_ms"] - lat["harq_only"]["p99_unrecovered_ms"]
    checks = {
        "latency_sim_full_wait_equals_harq_stats": abs(lat["wait"]["retx_per_100"] - ours_retx) < 1e-9,
        "latency_sim_conventional_equals_harq_stats": abs(lat["harq_only"]["retx_per_100"] - conv_retx) < 1e-9,
        "no_added_delay_latency_neutral": added <= 0.05,
    }
    values = {**motivation(lat),
        "RetxCutFullWait": (conv_retx / ours_retx, 1),
        "RetxCutNoDelay": (lat["harq_only"]["retx_per_100"] / lat["select_cb5"]["retx_per_100"], 1),
        "LostCutFullWait": (allstats["ai_and_ran"]["residual_after_all_pct"] / allstats["ours"]["residual_after_all_pct"], 1),
        "ThroughputGainLA": (float(np.mean(goodput)), 2),
        "PairingCutBestSixteen": (pf16["ai_and_ran"]["resends_per_100"] / rand["ours"]["resends_per_100"], 1),
    }
    full = json.load(open("results/n3_uma_v3_all.json"))
    meta = json.loads(str(np.load(sorted(glob.glob(f"results/{MACRO_SETS[0]}/c*.npz"))[0])["meta"]))
    points = sorted(float(k) for k in full["by_point"])
    setup = {"SetupLOneMs": f"{Timing.l1:.1f}", "SetupRecoveryRunMs": f"{Timing.run:.1f}",
             "SetupRecoveryDeadlineMs": f"{Timing.deadline:.1f}", "SetupLeadMs": f"{Timing.lead:.1f}",
             "SetupSlots": f"{full['slots']:,}", "SetupTBs": f"{full['slots'] * full['users']:,}",
             "SetupTbBytes": f"{meta['tb_bytes']:,}", "SetupTransmissions": f"{full['transmissions']}",
             "SetupRetxGapMs": f"{meta['generator']['gap_ms']:.1f}",
             "SetupEbNoLow": f"{points[0]:g}", "SetupEbNoHigh": f"{points[-1]:g}"}
    os.makedirs(OUT, exist_ok=True)
    report = {"values": {k: v for k, (v, _) in values.items()}, "setup": setup,
              "inputs": {"retx_per_100": {"conventional": conv_retx, "reverb_full_wait": ours_retx,
                                          "latency_policies": lat},
                         "lost_pct": {"conventional": allstats["ai_and_ran"]["residual_after_all_pct"],
                                      "reverb_full_wait": allstats["ours"]["residual_after_all_pct"]},
                         "goodput_vs_conventional_by_snr": dict(zip([f"{x:g}" for x in snr], goodput)),
                         "pairing_retx_per_100": {"conventional_best_of_16_sum_rate": pf16["ai_and_ran"]["resends_per_100"],
                                                  "reverb_random_partner": rand["ours"]["resends_per_100"]},
                         "added_p99_delay_latency_neutral_ms": added},
              "checks": checks}
    json.dump(report, open(f"{OUT}/numbers.json", "w"), indent=1, sort_keys=True)
    lines = ["% Generated by airan_recovery/src/analysis/paper_numbers.py from the stored results.  Do not edit.",
             "% Values, sources and checks: airan_recovery/results/paper/numbers.json"]
    for k, (v, digits) in values.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v:.{digits}f}}}")
    for k, v in setup.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    open(f"{OUT}/numbers.tex", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[2:]))
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        print("CHECKS FAILED:", bad)
        sys.exit(1)


if __name__ == "__main__":
    main()

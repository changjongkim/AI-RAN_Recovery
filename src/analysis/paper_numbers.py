#!/usr/bin/env python3
"""Numbers of the Reverb paper: every number the manuscript states is a LaTeX macro written here from the stored
results, so the text never carries a typed number (paper repo rule 2.1).  The paper repo copies the output with
scripts/update_numbers.sh; a rerun must reproduce the file byte for byte.

  python3 src/analysis/paper_numbers.py        (from the repository root; numpy only)
  outputs: results/paper/numbers.tex (macros), results/paper/numbers.json (values, sources, checks)

Macros (value, source):
  \\RetxCutFullWait     retransmissions per 100 TBs, conventional receiver only / Reverb waiting for every recovery,
                       macro cell, the 20,480 TBs of n3_uma_v3_cb + n3_uma_v3_more (results/n3_uma_v3_all.json)
  \\RetxCutNoDelay      the same with the latency-neutral rule (at the first transmission wait only when at most 5 of
                       10 code blocks failed, wait at every retransmission), from the latency simulation
                       (src/sim/harq_latency.py, DDDSU, L1 1.4 ms, recovery path measured, recovery deadline 11.0 ms,
                       lead 4.0 ms) on the same traces; checked: within every delay the share of the failed TBs
                       delivered is at least that of the conventional receiver (a lost TB is never delivered)
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
  \\SetupRecoveryRunMs is the 99th percentile and \\SetupRecoveryPathMs the median of the recovery path measured in the slot
  runtime (src/analysis/recovery_path.py); \\SetupRecoveryBoundMs, \\SetupArrivalMarginMs (prediction error of the rejection at
  the L1 result), \\SetupBoundLowMs and \\SetupBoundHighMs are settings of
  the real-time runs; \\SetupCarrierGhz the carrier frequency of the channels (generator metadata, 2.14 GHz before
  the field existed).  Eval*: the values of Figures 12-14 (function evaluation, the same runs as the panels).
  NeutralLaterPct / FullWaitLaterPct: failed TBs delivered later than without recovery (Latency-Neutral,
  Full Wait), NeutralEarlierPct: delivered earlier, WaitAddedDelayMs: the delay one wait adds to the next retransmission
  (the same at every transmission, checked), FullWaitMaxAddedMs: the largest added delay of a TB under Full Wait,
  FullWaitBehindMaxPp / FullWaitAheadFromMs: the largest shortfall of the delivered share of Full Wait against the
  conventional receiver and the delay from which Full Wait delivers more; MotPNinetyNine*: 99th-percentile delivery
  time of the failed TBs the first recovery does not decode, among those delivered with and without waiting;
  MotLaterRec* / MotLaterNext*: after the L1 fails a retransmission, the share of these TBs that the NRx of the combined
  buffer decodes / that the next retransmission decodes (range over the retransmissions, conventional order of events)
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src/sim")
from harq_latency import simulate, uplink_times  # noqa: E402

OUT = os.environ.get("PAPER_NUMBERS_OUT", "results/paper")   # another folder to compare without replacing
MACRO_SETS = ["n3_uma_v3_cb", "n3_uma_v3_more"]


class Timing:
    l1, run, deadline, lead = 1.4, 6.3, 11.0, 4.0   # deadline: retransmission slot 15.0 ms minus the lead 4.0 ms


SUFFIX = os.environ.get("RESULT_SUFFIX", "")   # "_f35": the 3.5 GHz runs (every macro since 2026-10-29 19:30)
N16 = os.environ.get("N16_SUFFIX", "")         # "d110_f35": GPU runtime at the 11.0 ms deadline on the 3.5 GHz slots
# the rejection of Section 4.3 (at the L1 result, prediction error 0.15 ms; src/runtime/controller_arrival.py, N30): the
# tag of every Reverb run of Figures 14-15 (recovered share, late L1, deadline sweep, AI, server replay)
ARRIVAL, ARRIVAL_MARGIN_MS = "arrivalpred60arrivalmargin015", 0.15
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recovery_path  # noqa: E402
PATH = recovery_path.stats(N16)   # recovery path in the slot runtime, one cell on one GPU
Timing.run = PATH["timeline_run_ms"]   # 99th percentile, rounded up to 0.1 ms (was a typed 6.3 ms before 2026-10-30)
BOUND_MS, BOUND_LOW_MS, BOUND_HIGH_MS = 6.8, 6.3, 7.3   # settings of the real-time runs: latest start D - BOUND, swept range
L1_DEADLINE_MS = 4.0   # the slot runtime counts an L1 result later than 4.0 ms after the slot as late (late_tbs)


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
    # the same slots decoded with the LMMSE estimate and with the true channel: macro cells from the receiver runs (four
    # seeds), small cells from the four-antenna run of Figure 3(b) (p1_ant4_v3 at 2.14 GHz, p1_umi_ant4 at 3.5 GHz; the
    # small-cell receiver run n5_umi_v3 has no 3.5 GHz version)
    small = "p1_ant4_v3" if not SUFFIX else "p1_umi_ant4" + SUFFIX
    for tag, sets in (("Macro", [n + SUFFIX for n in ("n5_uma_v3", "n5_uma_more1", "n5_uma_more2", "n5_uma_more3")]),
                      ("Small", [small])):
        rs = [json.load(open(f"results/{n}.json")) for n in sets]
        wt = np.array([r["slots"] for r in rs], float)
        est = float(np.dot(wt, [r["fail_pct"]["baseline_lslin_lmmse"] for r in rs]) / wt.sum())
        true = float(np.dot(wt, [r["fail_pct"]["baseline_perf_csi_kbest"] for r in rs]) / wt.sum())
        v[f"MotEstFail{tag}"] = (est, 1); v[f"MotTrueFail{tag}"] = (true, 1)
        if all("baseline_lmmse_kbest" in r["fail_pct"] for r in rs):
            # the share of the channel estimate with the detector fixed: K-best detection with the LMMSE estimate
            # against K-best detection with the true channel (the small-cell run has no K-best with the estimate)
            kbest = float(np.dot(wt, [r["fail_pct"]["baseline_lmmse_kbest"] for r in rs]) / wt.sum())
            v[f"MotKbestFail{tag}"] = (kbest, 1)
            v[f"MotEstShare{tag}"] = (100 * (kbest - true) / kbest, 0)
    # median delay spread of the 3GPP TR 38.901 models without a direct path at 3.5 GHz (Table 7.5-6: lgDS =
    # a log10(b + fc) + c; UMa uses fc = 6 GHz below 6 GHz), the parameters of the channel generator (Sionna)
    v["MotDelaySpreadMacro"] = (1e9 * 10 ** (-0.204 * np.log10(6.0) - 6.28), 0)
    v["MotDelaySpreadSmall"] = (1e9 * 10 ** (-0.24 * np.log10(1.0 + 3.5) - 6.83), 0)
    lanes = json.load(open("results/n13_lanes.json"))
    large = {r["lanes"]: r for r in lanes["large"] if r["lanes"] <= 4}
    v["MotTimeSmallNrx"] = (lanes["small"][0]["latency_ms_mean"], 1)
    v["MotTimeLargeNrx"] = (large[1]["latency_ms_mean"], 1)
    v["MotTimeLargeFour"] = (large[4]["latency_ms_mean"], 1)
    v["MotRunsMin"] = (min(r["runs_per_s"] for r in large.values()), 0)
    v["MotRunsMax"] = (max(r["runs_per_s"] for r in large.values()), 0)
    v["MotGpuTimePerSlot"] = (v["MotSlotsWithFailure"][0] / 100 * PATH["p50_ms"], 1)   # the whole recovery path occupies the GPU
    v["MotRecoveryResultMs"] = (Timing.l1 + Timing.run, 1)
    v["MotFirstRetxMs"] = (lat["first_retx_ms"], 1)
    v["MotWaitDelayMs"] = (lat["wait_delay_ms"], 1)
    v["MotPNinetyNineNoWait"] = (lat["wait"]["p99_paired_conventional_ms"], 1)
    v["MotPNinetyNineWait"] = (lat["wait"]["p99_paired_ms"], 1)
    v["MotRetxWait"] = (lat["wait"]["retx_per_100"], 1)
    v["MotLaterRecLow"] = (min(lat["later_recovery_pct"]), 0); v["MotLaterRecHigh"] = (max(lat["later_recovery_pct"]), 0)
    v["MotLaterNextLow"] = (min(lat["later_next_retx_pct"]), 0); v["MotLaterNextHigh"] = (max(lat["later_next_retx_pct"]), 0)
    rows = {}
    for line in open("results/n16_summary.txt"):
        f = line.split()
        if len(f) >= 10 and f[0] in ("allb68" + N16, "allb68g4" + N16):
            rows.setdefault((f[0], int(f[2])), []).append(float(f[9]))
    g = lambda pol, c: float(np.mean(rows[(pol + N16, c)]))
    v["MotGpuOneCell"] = (g("allb68", 1), 0); v["MotGpuOneEight"] = (g("allb68", 8), 0)
    v["MotGpuFourEight"] = (g("allb68g4", 8), 0); v["MotGpuFourMax"] = (g("allb68g4", 24), 0)
    red = lambda n: stats([n])["resends_per_100"] / stats([n], "ours")["resends_per_100"]
    v["MotIciPubNone"] = (red("irc_uma_v3"), 1); v["MotIciPubTen"] = (red("ici_uma_inr10"), 1)
    v["MotIciTrainedNone"] = (red("ici_uma_v3_iciaware"), 1); v["MotIciTrainedTen"] = (red("ici_uma_inr10_iciaware"), 1)
    v["MotIciPubNoneSmall"] = (red("irc_umi_v3"), 1); v["MotIciPubTenSmall"] = (red("ici_umi_inr10"), 1)
    v["MotIciTrainedNoneSmall"] = (red("ici_umi_v3_iciaware"), 1); v["MotIciTrainedTenSmall"] = (red("ici_umi_inr10_iciaware"), 1)
    v["MotIciConvNone"] = (stats(["irc_uma_v3"])["resends_per_100"], 1); v["MotIciConvTen"] = (stats(["ici_uma_inr10"])["resends_per_100"], 1)
    return v


def load_traces(sets):
    parts = []
    for s in sets:
        good, total = {}, None
        for f in sorted(glob.glob(f"results/{s}{SUFFIX}/c*.npz")):
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


def delivered_share(done, fail, at):
    """Share (%) of the failed TBs delivered within each delay of at; a TB lost after the last transmission is never
    delivered.  The shares change only at delivery times, so evaluating them there compares two policies exactly."""
    d = np.sort(done[fail])
    return 100.0 * np.searchsorted(d, at, side="right") / fail.sum()


def latency_policies():
    conv, rec, steps, conv_steps, cand = load_traces(MACRO_SETS)
    fail = ~conv
    first_rec = rec & (Timing.l1 + Timing.run <= Timing.deadline)
    out, dones = {}, {}
    for pol in ("harq_only", "wait", "select_cb5"):
        done, tx = simulate(conv, rec, conv_steps if pol == "harq_only" else steps, cand, pol, "DDDSU", Timing, l1_steps=conv_steps)
        dones[pol] = done
        out[pol] = {"retx_per_100": float(100.0 * (tx - 1).sum() / len(tx)),
                    "lost_pct": float(100.0 * (~np.isfinite(done)).mean()), "tbs": int(len(tx))}
    d0 = dones["harq_only"]
    for pol in ("select_cb5", "wait"):
        d = dones[pol]
        later = fail & np.isfinite(d0) & (d > d0 + 1e-9)
        out[pol]["later_pct_of_failed"] = float(100.0 * later.sum() / fail.sum())
        out[pol]["earlier_pct_of_failed"] = float(100.0 * (fail & (d < d0 - 1e-9)).sum() / fail.sum())
        out[pol]["added_delay_ms_max"] = float((d[later] - d0[later]).max()) if later.any() else 0.0
        at = np.unique(np.concatenate([[0.0], d0[fail & np.isfinite(d0)], d[fail & np.isfinite(d)]]))
        diff = delivered_share(d, fail, at) - delivered_share(d0, fail, at)
        out[pol]["share_min_diff_pp"] = float(diff.min())
        behind = np.flatnonzero(diff < -1e-9)
        if len(behind):
            out[pol]["behind_from_ms"] = float(at[behind[0]])
            out[pol]["ahead_from_ms"] = float(at[behind[-1] + 1])   # from here on the share is at least the conventional one
            assert (diff[behind[-1] + 1:] >= -1e-9).all()
        pair = fail & ~first_rec & np.isfinite(d0) & np.isfinite(d)   # the same TBs under both policies
        out[pol]["p99_paired_ms"] = float(np.percentile(d[pair], 99))
        out[pol]["p99_paired_conventional_ms"] = float(np.percentile(d0[pair], 99))
    # after the L1 fails retransmission k (and every transmission before it): the share the NRx of the combined buffer
    # decodes, and the share the next retransmission decodes with the L1 alone
    alive, later_rec, later_next = fail.copy(), [], []
    for k in range(1, len(steps)):
        alive = alive & ~conv_steps[k]
        later_rec.append(float(100.0 * (steps[k] & ~conv_steps[k])[alive].mean()))
        if k + 1 < len(steps):
            later_next.append(float(100.0 * conv_steps[k + 1][alive].mean()))
    out["later_recovery_pct"], out["later_next_retx_pct"] = later_rec, later_next
    out["delivered_at_recovery_pct"] = float(100.0 * (fail & (dones["wait"] <= Timing.l1 + Timing.run + 1e-9)).sum() / fail.sum())
    out["delivered_final_pct"] = {p: float(100.0 * (fail & np.isfinite(dones[p])).sum() / fail.sum()) for p in dones}
    out["neutral_waits_pct_of_failed"] = float(100.0 * cand[fail].mean())
    out["neutral_waiting_recovered_pct"] = float(100.0 * rec[fail & cand].mean())
    # Recovery Gate with K = 0..10 failed code blocks at the first transmission (waits at every retransmission)
    cb = np.concatenate([np.load(f)["cbfail_t0"].reshape(-1) for s_ in MACRO_SETS for f in sorted(glob.glob(f"results/{s_}{SUFFIX}/c*.npz"))])
    ksweep = {}
    for k in range(0, 11):
        dk, txk = simulate(conv, rec, steps, (cb >= 0) & (cb <= k), "select_cb5", "DDDSU", Timing, l1_steps=conv_steps)
        at = np.unique(np.concatenate([[0.0], d0[fail & np.isfinite(d0)], dk[fail & np.isfinite(dk)]]))
        ksweep[k] = {"retx_per_100": float(100.0 * (txk - 1).sum() / len(txk)),
                     "share_min_diff_pp": float((delivered_share(dk, fail, at) - delivered_share(d0, fail, at)).min())}
    out["k_sweep"] = ksweep
    # the same recoveries without the decision: every failed TB receives the grant at the L1 result (parallel)
    dp, txp = simulate(conv, rec, steps, cand, "parallel", "DDDSU", Timing, l1_steps=conv_steps)
    out["parallel"] = {"retx_per_100": float(100.0 * (txp - 1).sum() / len(txp)), "lost_pct": float(100.0 * (~np.isfinite(dp)).mean())}
    # Full Wait with the first recovery only (no NRx decode of a retransmission)
    df, txf = simulate(conv, rec, conv_steps, cand, "wait", "DDDSU", Timing, l1_steps=conv_steps)
    out["first_only"] = {"retx_per_100": float(100.0 * (txf - 1).sum() / len(txf)), "lost_pct": float(100.0 * (~np.isfinite(df)).mean())}
    out["first_retx_ms"] = float(d0[fail & np.isfinite(d0)].min())   # the first retransmission is decoded
    # one wait moves the next retransmission by this much, at the first transmission and at every retransmission
    ul = uplink_times("DDDSU", 300.0)
    nxt = lambda t: ul[np.searchsorted(ul, t)]
    shifts = {round(float(nxt(s + r + Timing.run + Timing.lead) - nxt(s + r + Timing.lead)), 9)
              for s, r in [(0.0, Timing.l1)] + [(u, 0.5 + Timing.l1) for u in ul[1:9]]}
    out["wait_delay_ms"] = shifts.pop() if len(shifts) == 1 else float("nan")
    return out


EVAL_EFFECT = {   # condition: runs (different seeds), as figures/paper/make_eval_figures.py
    "Macro": ["n3_uma_v3_cb", "n3_uma_v3_more"], "Blocked": ["n3_uma_nlos_v3", "n3_uma_nlos_more"],
    "Direct": ["n3_uma_los_v3", "n3_uma_los_more"], "Small": ["n1_umi_v3_cb"]}
EVAL_SPEED = {3: (["n3_uma_v3_cb", "n3_uma_v3_more"], ["n1_umi_v3_cb"]), 30: (["n3_uma_v30", "n3_uma_v30_more"], ["n1_umi_v30"]),
              120: (["n3_uma_v120", "n3_uma_v120_more"], ["n3_umi_v120"])}


def runtime_rows():
    """Recovered share of failed TBs and L1 blocks past the L1 deadline per (policy tag, cells), mean over seeds, from
    results/n16_summary.txt, as figures/paper/make_eval_figures.py."""
    rec, late = {}, {}
    for line in open("results/n16_summary.txt"):
        f = line.split()
        if len(f) < 11 or not f[1].isdigit():
            continue
        rec.setdefault(f[0], {}).setdefault(int(f[2]), []).append(float(f[9]))
        late.setdefault(f[0], {}).setdefault(int(f[2]), []).append(float(f[3]))
    mean = lambda d: {t: {c: float(np.mean(v)) for c, v in cs.items()} for t, cs in d.items()}
    return mean(rec), mean(late)


def evaluation(lat):
    """Macros of the Evaluation (Figures 12-14), from the same results as figures/paper/make_eval_figures.py."""
    v = {}
    st = {g: {k: stats(n, k) for k in ("ours", "ai_for_ran", "ai_and_ran")} for g, n in EVAL_EFFECT.items()}
    for g, x in st.items():
        v[f"EvalRetxReverb{g}"] = (x["ours"]["resends_per_100"], 1)
        v[f"EvalRetxAi{g}"] = (x["ai_for_ran"]["resends_per_100"], 1)
        v[f"EvalRetxConv{g}"] = (x["ai_and_ran"]["resends_per_100"], 1)
        v[f"EvalRetxCut{g}"] = (x["ai_and_ran"]["resends_per_100"] / x["ours"]["resends_per_100"], 1)
        v[f"EvalRetxCutAi{g}"] = (x["ai_for_ran"]["resends_per_100"] / x["ours"]["resends_per_100"], 1)
        v[f"EvalLostCut{g}"] = (x["ai_and_ran"]["residual_after_all_pct"] / x["ours"]["residual_after_all_pct"], 1)
        v[f"EvalLostCutAi{g}"] = (x["ai_for_ran"]["residual_after_all_pct"] / x["ours"]["residual_after_all_pct"], 1)
    cut = lambda n: stats(n)["resends_per_100"] / stats(n, "ours")["resends_per_100"]
    fast = max(EVAL_SPEED)
    v["EvalSpeedMacroFast"], v["EvalSpeedSmallFast"] = (cut(EVAL_SPEED[fast][0]), 1), (cut(EVAL_SPEED[fast][1]), 1)
    v["EvalSpeedMax"] = (fast, 0)
    # the channel of a retransmission later than the gap of the traces (a wait can move the retransmission up to Ur):
    # the macro traces at 3 and 30 km/h against the same seeds with the channel 12.5 and 15.0 ms later (gap runs)
    for name, run in (("Slow", "n3_uma_v3_cb"), ("Fast", "n3_uma_v30")):
        v[f"EvalGap{name}Base"] = (cut([run]), 2)
        for g, label in (("125", "Twelve"), ("150", "Fifteen")):
            if os.path.exists(f"results/{run}_gap{g}{SUFFIX}.json"):
                v[f"EvalGap{name}{label}"] = (cut([f"{run}_gap{g}"]), 2)
    gaps = [(v[f"EvalGap{n}{l}"][0], v[f"EvalGap{n}Base"][0]) for n in ("Slow", "Fast") for l in ("Twelve", "Fifteen")
            if f"EvalGap{n}{l}" in v]
    v["EvalGapChangePct"] = (100.0 * max(abs(a / b - 1.0) for a, b in gaps), 1)
    # pairing: a random partner against the best partner by sum rate among the most candidates with a finished run
    for cell, pre in (("Macro", "p2b_uma_u"), ("Small", "p2b_u")):
        ks = [k for k in (4, 8, 16, 32) if os.path.exists(f"results/{pre}{k}_pf{SUFFIX}.json")]
        rnd = {k: stats([f"{pre}8_random"], k)["resends_per_100"] for k in ("ours", "ai_for_ran", "ai_and_ran")}
        best = {k: stats([f"{pre}{ks[-1]}_pf"], k)["resends_per_100"] for k in ("ours", "ai_for_ran", "ai_and_ran")}
        ours_pf = [stats([f"{pre}{k}_pf"], "ours")["resends_per_100"] for k in ks]
        v[f"EvalPairCand{cell}"] = (ks[-1], 0)
        v[f"EvalPairRandomReverb{cell}"] = (rnd["ours"], 1)
        v[f"EvalPairRandomConv{cell}"] = (rnd["ai_and_ran"], 1)
        v[f"EvalPairBestConv{cell}"] = (best["ai_and_ran"], 1)
        v[f"EvalPairBestAi{cell}"] = (best["ai_for_ran"], 1)
        v[f"EvalPairCutConv{cell}"] = (best["ai_and_ran"] / rnd["ours"], 1)
        v[f"EvalPairCutAi{cell}"] = (best["ai_for_ran"] / rnd["ours"], 1)
        v[f"EvalPairBestReverbLow{cell}"], v[f"EvalPairBestReverbHigh{cell}"] = (min(ours_pf), 1), (max(ours_pf), 1)
        v[f"EvalPairFewConv{cell}"] = (stats([f"{pre}{ks[0]}_pf"])["resends_per_100"], 1)
        v[f"EvalPairFewCand{cell}"] = (ks[0], 0)
    # delay (Figure 13 a, b): the latency simulation of latency_policies
    v["EvalDelayRecoveredPct"] = (lat["delivered_at_recovery_pct"], 1)
    v["EvalDelayConvFinalPct"] = (lat["delivered_final_pct"]["harq_only"], 1)
    v["EvalDelayReverbFinalPct"] = (lat["delivered_final_pct"]["wait"], 1)
    v["EvalRetxFullWait"] = (lat["wait"]["retx_per_100"], 1)
    v["EvalRetxNeutral"] = (lat["select_cb5"]["retx_per_100"], 1)
    v["EvalNeutralWaitPct"] = (lat["neutral_waits_pct_of_failed"], 0)
    conv100 = lat["harq_only"]["retx_per_100"]
    keep_k = [k for k, x in lat["k_sweep"].items() if x["share_min_diff_pp"] >= 0.0]
    cuts = [conv100 / lat["k_sweep"][k]["retx_per_100"] for k in keep_k]
    v["EvalKMax"] = (max(keep_k), 0)
    v["EvalKCutLow"], v["EvalKCutHigh"] = (min(cuts), 2), (max(cuts), 2)
    v["EvalKRetxLow"] = (min(lat["k_sweep"][k]["retx_per_100"] for k in keep_k), 1)
    v["EvalKRetxHigh"] = (max(lat["k_sweep"][k]["retx_per_100"] for k in keep_k), 1)
    v["AblParallelRetx"] = (lat["parallel"]["retx_per_100"], 1)
    v["AblParallelCut"] = (conv100 / lat["parallel"]["retx_per_100"], 2)
    v["AblFirstOnlyRetx"] = (lat["first_only"]["retx_per_100"], 1)
    v["AblFirstOnlyLost"] = (lat["first_only"]["lost_pct"], 2)
    v["AblFullWaitLost"] = (lat["wait"]["lost_pct"], 2)
    v["AblConvLost"] = (lat["harq_only"]["lost_pct"], 2)
    v["EvalNeutralWaitRecPct"] = (lat["neutral_waiting_recovered_pct"], 0)
    # link adaptation (Figure 13 c, d)
    for cell, run in (("Macro", "n6_uma_v3"), ("Small", "n6_umi_v3")):
        o = json.load(open(f"results/{run}{SUFFIX}_olla.json"))["results"]
        snr = sorted({float(k.split("_")[0][3:]) for k in o})
        g = [o[f"snr{x:g}_target0.1"]["ours"]["goodput_vs_ai_and_ran"] for x in snr]
        a = [o[f"snr{x:g}_target0.1"]["ai_for_ran"]["goodput_vs_ai_and_ran"] for x in snr]
        v[f"EvalLa{cell}Low"], v[f"EvalLa{cell}High"] = (min(g), 2), (max(g), 2)
        v[f"EvalLaAi{cell}Low"], v[f"EvalLaAi{cell}High"] = (min(a), 2), (max(a), 2)
        if cell == "Macro":
            best = snr[int(np.argmax(g))]
            v["EvalLaBestSnr"] = (best, 0)
            v["EvalLaMcsConv"] = (o[f"snr{best:g}_target0.1"]["ai_and_ran"]["mean_mcs"], 1)
            v["EvalLaMcsReverb"] = (o[f"snr{best:g}_target0.1"]["ours"]["mean_mcs"], 1)
            v["EvalLaRetxConv"] = (o[f"snr{best:g}_target0.1"]["ai_and_ran"]["resends_per_100"], 1)
            v["EvalLaRetxReverb"] = (o[f"snr{best:g}_target0.1"]["ours"]["resends_per_100"], 1)
    # GPU (Figure 14): the real-time runs, policy tags of make_eval_figures.py
    rec, late = runtime_rows()
    tag = lambda p, g, x="": f"{p}{'g4' if g == 4 else ''}{N16}{x}"
    r4, r1 = rec[tag("allb68", 4, ARRIVAL)], rec[tag("allb68", 1, ARRIVAL)]
    v["EvalGpuFourFew"], v["EvalGpuFourMid"], v["EvalGpuFourMax"] = (r4[4], 1), (r4[16], 1), (r4[24], 1)
    v["EvalGpuOneFew"], v["EvalGpuOneMid"], v["EvalGpuOneMax"] = (r1[1], 1), (r1[4], 1), (r1[8], 1)
    for name, p, x in (("Before", "parab68", ""), ("NoReject", "allb68", "admit0")):
        b4, b1 = rec[tag(p, 4, x)], rec[tag(p, 1, x)]
        v[f"EvalGpuVs{name}FourMid"], v[f"EvalGpuVs{name}FourMax"] = (r4[16] / b4[16], 1), (r4[24] / b4[24], 1)
        v[f"EvalGpuVs{name}OneMid"], v[f"EvalGpuVs{name}OneMax"] = (r1[4] / b1[4], 1), (r1[8] / b1[8], 1)
    v["EvalGpuBeforeNoRejectFourMax"] = (max(rec[tag("parb68", 4)].values()), 1)
    v["EvalGpuTwoFourMid"] = (rec[tag("allb68", 4, "l2")][16], 1)
    v["EvalGpuNoRejectFourMidAbs"] = (rec[tag("allb68", 4, "admit0")][16], 1)
    v["EvalGpuBeforeFourMidAbs"] = (rec[tag("parab68", 4)][16], 1)
    l4 = {k: late[tag(p, 4, x)] for k, p, x in (("reverb", "allb68", ARRIVAL), ("none", "offb68", ""), ("two", "allb68", "l2"),
                                                 ("four", "allb68", "l4"))}
    v["EvalLateReverbMax"] = (max(l4["reverb"].values()), 2)
    v["EvalLateNoneMax"] = (max(l4["none"].values()), 2)
    v["EvalLateDiffMax"] = (max(l4["reverb"][c] - l4["none"][c] for c in l4["reverb"]), 2)
    v["EvalLateTwoMid"], v["EvalLateTwoMax"] = (l4["two"][16], 1), (l4["two"][24], 1)
    v["EvalLateReverbMid"], v["EvalLateNoneMid"] = (l4["reverb"][16], 2), (l4["none"][16], 2)
    v["EvalLateNoRejectMid"] = (late[tag("allb68", 4, "admit0")][16], 2)
    v["EvalLateBeforeMid"] = (late[tag("parab68", 4)][16], 2)
    v["EvalLateFourMid"], v["EvalLateFourMax"] = (l4["four"][16], 1), (l4["four"][24], 1)
    carrier = N16[N16.index("_"):] if "_" in N16 else ""
    d1 = {d: rec[f"allb68d{d}{carrier}{ARRIVAL}"][4] for d in (90, 100, 110, 120, 130)}
    d4 = {d: rec[f"allb68g4d{d}{carrier}{ARRIVAL}"][16] for d in (90, 100, 110, 120, 130)}
    v["EvalDeadlineSpread"] = (max(max(d[k] for k in (100, 110, 120, 130)) - min(d[k] for k in (100, 110, 120, 130)) for d in (d1, d4)), 1)
    v["EvalDeadlineNineOne"], v["EvalDeadlineElevenOne"] = (d1[90], 1), (d1[110], 1)
    v["EvalDeadlineNineFour"], v["EvalDeadlineElevenFour"] = (d4[90], 1), (d4[110], 1)
    eff = json.load(open(f"results/server_effect{SUFFIX}.json"))
    conv = eff["conventional"]["retx_per_100"]
    row = lambda g, c: eff["rows"][f"{g}_{c}"]
    v["EvalServerEvery"] = (conv / eff["rows"]["every_recovery"]["retx_per_100"], 1)
    ld = eff["loads"]
    v["EvalLoadFixed"] = (ld["recovery_runs_per_tb_fixed_mcs14"] / ld["first_failures_per_tb_fixed_mcs14"], 1)
    v["EvalLoadLa"] = (ld["recovery_runs_per_tb_la"] / ld["first_failures_per_tb_fixed_mcs14"], 1)
    v["EvalLoadRetxPct"] = (100.0 * (ld["recovery_runs_per_tb_fixed_mcs14"] / ld["first_failures_per_tb_fixed_mcs14"] - 1), 0)
    spread = [100.0 * (max(r["availability_by_esno"].values()) - min(r["availability_by_esno"].values()))
              for k, r in eff["rows"].items() if k.startswith("four_")]
    v["EvalOnTimeEsnoSpread"] = (max(spread), 0)
    st = ld.get("by_esno_check", {})
    if st:
        v["EvalStratDiffPct"] = (max(100.0 * abs(x["retx_per_100"] - x["single_retx_per_100"]) / x["single_retx_per_100"]
                                     for x in st.values()), 1)
    chk = eff.get("load_matching_check", {})
    if chk:
        v["EvalLoadCheckMinPp"] = (min(100.0 * (c["measured"] - c["predicted"]) for c in chk.values()), 0)
        v["EvalLoadCheckLoadLow"] = (min(c["load"] for c in chk.values()), 1)
        v["EvalLoadCheckLoadHigh"] = (max(c["load"] for c in chk.values()), 1)
        v["EvalLoadCheckMaxPp"] = (max(100.0 * abs(c["measured"] - c["predicted"]) for c in chk.values()), 0)
        v["EvalLoadCheckFailPct"] = (100.0 * float(np.mean([c["failed_per_tb"] for c in chk.values()])), 1)
    tasks = eff.get("per_task_check", {})
    if tasks:   # retransmissions with the recorded outcome of each recovery request, % fewer than with one share
        less = [100.0 * (1 - r[f"lag{lag}"]["retx_per_100"] / r["independent_retx_per_100"]) for r in tasks.values() for lag in (6, 8)]
        v["EvalTaskLessMin"], v["EvalTaskLessMax"] = (min(less), 0), (max(less), 0)
    dec = row("four", 24).get("declining")
    if dec:     # beyond the measured cells, the on-time recoveries falling further at the rate between 20 and 24 cells
        v["EvalDeclineCutFourMax"] = (conv / dec["retx_per_100"], 2)
        v["EvalDeclineGoodFourMax"] = (dec["goodput_gain_la"], 2)
        v["EvalServerFourMaxTwo"] = (conv / row("four", 24)["retx_per_100"], 2)
    v["EvalServerGoodEvery"] = (eff["rows"]["every_recovery"]["goodput_gain_la"], 2)
    for name, g, c in (("FourFew", "four", 4), ("FourEight", "four", 8), ("FourMid", "four", 16), ("FourMax", "four", 24),
                       ("OneFew", "one", 1), ("OneTwo", "one", 2), ("OneMax", "one", 8)):
        v[f"EvalServer{name}"] = (conv / row(g, c)["retx_per_100"], 1)
        v[f"EvalServerGood{name}"] = (row(g, c)["goodput_gain_la"], 2)
        v[f"EvalAvail{name}"] = (100.0 * row(g, c)["availability"], 0)
        v[f"EvalAvailLoad{name}"] = (100.0 * row(g, c)["availability_fixed"], 0)
    return v



def evaluation_more():
    """Macros of Sections 5.7-5.9: AI inference on the same GPUs (results/n22_summary.json against the runs without AI),
    inter-cell interference (the runs of figures/paper/make_eval_figures.py ICI_RUNS that have finished), and the
    sensitivity to resource blocks (N21) and to antennas and users (N5, U1)."""
    v = {}
    n22 = json.load(open("results/n22_summary.json"))
    rows = n22["rows"]
    v["EvalAiSeconds"] = (60 if n22["runs"] == "60 s" else 6, 0)
    rec, _ = runtime_rows()
    keep = []
    for g, cells in ((4, (8, 16, 24)), (1, (1, 2, 4))):
        base = rec[f"allb68{'g4' if g == 4 else ''}{N16}{ARRIVAL}"]
        for r in (4, 8):
            keep += [rows[f"all_g{g}_r{r}_c{c}"]["recovered_pct"] / base[c] for c in cells]
    v["EvalAiKeepLow"], v["EvalAiKeepHigh"] = (100.0 * min(keep), 0), (100.0 * max(keep), 0)
    v["EvalAiRecFourMid"] = (rows["all_g4_r4_c16"]["recovered_pct"], 1)
    v["EvalAiRecFourMidEight"] = (rows["all_g4_r8_c16"]["recovered_pct"], 1)
    v["EvalAiRecFourMidNone"] = (rec[f"allb68g4{N16}{ARRIVAL}"][16], 1)
    allrec = [r for r in rows.values() if r["policy"] == "all"]
    v["EvalAiLateMax"] = (max(r["late_l1_pct"] for r in allrec), 2)
    v["EvalAiLateOffMax"] = (max(r["late_l1_pct"] for r in rows.values() if r["policy"] == "off"), 2)
    v["EvalAiSloLow"] = (min(r["ai_within_slo_pct"] for r in allrec), 1)
    v["EvalAiSloLowFour"] = (min(rows[f"all_g4_r4_c{c}"]["ai_within_slo_pct"] for c in (8, 16, 24)), 1)
    v["EvalAiSloOffLow"] = (min(r["ai_within_slo_pct"] for r in rows.values() if r["policy"] == "off"), 1)
    v["EvalAiRejectMax"] = (max(r["ai_rejected_pct"] for r in allrec), 1)
    v["EvalAiRejectOffMax"] = (max(r["ai_rejected_pct"] for r in rows.values() if r["policy"] == "off"), 1)
    v["EvalAiTtftOffMax"] = (max(r["ai_ttft_p99_ms"] for r in rows.values() if r["policy"] == "off"), 0)
    v["EvalAiTtftMax"] = (max(r["ai_ttft_p99_ms"] for r in allrec), 0)
    v["EvalAiRequestsLow"] = (min(r["ai_requests"] for r in rows.values()), 0)
    v["EvalAiRequestsHigh"] = (max(r["ai_requests"] for r in rows.values()), 0)
    v["EvalAiCompletedLateMax"] = (max(100.0 - r["ai_within_slo_pct"] - r["ai_rejected_pct"] for r in allrec), 1)
    # Receiver Selector at fixed levels: no interference (irc_uma_v3 for the NRx trained without interference, the
    # same slots as ici_uma_v3_iciaware for the NRx trained with it) and 10 dB
    none_pub = stats(["irc_uma_v3"], "ours")["resends_per_100"]
    none_trained = stats(["ici_uma_v3_iciaware"], "ours")["resends_per_100"]
    ten_pub, ten_trained = stats(["ici_uma_inr10"], "ours")["resends_per_100"], stats(["ici_uma_inr10_iciaware"], "ours")["resends_per_100"]
    v["AblSelNonePub"], v["AblSelNoneTrained"] = (none_pub, 1), (none_trained, 1)
    v["AblSelTenPub"], v["AblSelTenTrained"] = (ten_pub, 1), (ten_trained, 1)
    v["AblSelNoneBest"], v["AblSelTenBest"] = (min(none_pub, none_trained), 1), (min(ten_pub, ten_trained), 1)
    # Receiver Selector on cells with mixed interference (N19, src/analysis/selector.py): the best threshold, one NRx for
    # every slot, and the oracle that takes for each TB whichever NRx decodes it
    # Mix: interference in about three quarters of the slots (n19_*_mix); Rare: in a quarter (n19_*_mix75)
    gaps = []
    for cell, c, mix, tag in (("Macro", "uma", "Mix", ""), ("Small", "umi", "Mix", ""), ("Macro", "uma", "Rare", "75"), ("Small", "umi", "Rare", "75")):
        f = f"results/n19_{c}_selector{tag}{SUFFIX}.json"
        if os.path.exists(f):
            d = json.load(open(f))
            ch = d["choices"]
            v[f"AblSel{mix}{cell}"] = (ch[d["best_theta"]]["retx_cut_x"], 2)
            v[f"AblSel{mix}{cell}Trained"] = (ch["trained"]["retx_cut_x"], 2)
            v[f"AblSel{mix}{cell}Pub"] = (ch["published"]["retx_cut_x"], 2)
            v[f"AblSel{mix}{cell}Oracle"] = (ch["oracle"]["retx_cut_x"], 2)
            v[f"AblSel{mix}{cell}ThetaFive"] = (ch["theta_5dB"]["retx_cut_x"], 2)
            if mix == "Mix" and cell == "Macro":
                v["AblSelThetaDb"] = (float(d["best_theta"][6:-2]), 0)   # the threshold of Receiver Selector
            single = max(ch["trained"]["retx_cut_x"], ch["published"]["retx_cut_x"])
            v[f"AblSel{mix}{cell}Single"] = (single, 2)
            v[f"AblSel{mix}{cell}OverSinglePct"] = (100.0 * (ch[d["best_theta"]]["retx_cut_x"] / single - 1.0), 0)
            gaps.append(100.0 * (1.0 - ch[d["best_theta"]]["retx_cut_x"] / ch["oracle"]["retx_cut_x"]))
    if gaps:
        v["AblSelOracleGapPct"] = (max(gaps), 0)   # Receiver Selector against the better NRx for each TB, every mix and cell
    # inter-cell interference: macro and small cells, 10 dB (two users of neighbor cells) and 20 dB with two users
    def load(n):
        f = f"results/{n}{SUFFIX}.json"
        return json.load(open(f))["all"] if os.path.exists(f) else None
    for cell, pre10, pre20 in (("Macro", "ici_uma_inr10", "i1_uma_k2_inr20"), ("Small", "ici_umi_inr10", "i1_umi_k2_inr20")):
        for cond, pre in (("Ten", pre10), ("Twenty", pre20), ("TwentyFour", pre20.replace("_k2_", "_k4_"))):   # four neighbor users
            t, p = load(pre + "_iciaware"), load(pre)
            if t is None:
                continue
            r = lambda d, k: d[k]["resends_per_100"]
            v[f"EvalIci{cell}{cond}Conv"] = (r(t, "ai_and_ran"), 1)
            v[f"EvalIci{cell}{cond}Trained"] = (r(t, "ours"), 1)
            v[f"EvalIci{cell}{cond}Irc"] = (r(t, "ai_and_ran_irc"), 1)
            v[f"EvalIci{cell}{cond}OursIrc"] = (r(t, "ours_irc"), 1)
            v[f"EvalIci{cell}{cond}Cut"] = (r(t, "ai_and_ran") / r(t, "ours"), 1)
            v[f"EvalIci{cell}{cond}CutIrc"] = (r(t, "ai_and_ran_irc") / r(t, "ours_irc"), 1)
            v[f"EvalIci{cell}{cond}Ai"] = (r(t, "ai_for_ran"), 1)
            if p is not None:
                v[f"EvalIci{cell}{cond}Pub"] = (r(p, "ours"), 1)
                v[f"EvalIci{cell}{cond}CutPub"] = (r(p, "ai_and_ran") / r(p, "ours"), 1)
    # the range over every interference condition (two users at 10 and 20 dB, four at 20 dB, both cells)
    cuts = [x[0] for k, x in v.items() if k.startswith("EvalIci") and k.endswith("Cut") and not k.endswith(("IrcCut", "PubCut"))]
    v["EvalIciCutLow"], v["EvalIciCutHigh"] = (min(cuts), 1), (max(cuts), 1)
    # resource blocks: 273 (macro traces), 106 and 52 PRBs, users at 3 km/h
    for name, n in (("Full", ["n3_uma_v3_cb", "n3_uma_v3_more"]), ("Half", ["n21_uma_p106_v3"]), ("Fifth", ["n21_uma_p52_v3"])):
        c, o, a = stats(n), stats(n, "ours"), stats(n, "ai_for_ran")
        v[f"EvalPrb{name}Conv"] = (c["resends_per_100"], 1)
        v[f"EvalPrb{name}Reverb"] = (o["resends_per_100"], 1)
        v[f"EvalPrb{name}Cut"] = (c["resends_per_100"] / o["resends_per_100"], 1)
        v[f"EvalPrb{name}CutAi"] = (a["resends_per_100"] / o["resends_per_100"], 1)
    # antennas and users: failed TBs of the LMMSE receiver decoded by K-best with the LMMSE estimate and the true channel
    for name, sets in (("FourTwo", ["n5_uma_v3", "n5_uma_more1", "n5_uma_more2", "n5_uma_more3"]), ("FourFour", ["u1_uma_r4_u4"]),
                       ("EightFour", ["u1_uma_r8_u4_b1"]), ("SixteenTwo", ["u1_uma_r16_u2_b1"])):
        rs = [json.load(open(f"results/{n}{SUFFIX}.json")) for n in sets]
        w = np.array([r["fail_pct"]["baseline_lslin_lmmse"] * r["slots"] for r in rs])
        for lab, key in (("Kbest", "baseline_lmmse_kbest"), ("True", "baseline_perf_csi_kbest")):
            v[f"EvalUsers{name}{lab}"] = (float(np.dot(w, [r["recovers_pct_of_conventional_failures"][key] for r in rs]) / w.sum()), 0)
        v[f"EvalUsers{name}Fail"] = (float(np.dot(np.array([r["slots"] for r in rs], float), [r["fail_pct"]["baseline_lslin_lmmse"] for r in rs]) / sum(r["slots"] for r in rs)), 1)
    return v

def discussion():
    """Macros of the Discussion (receivers inside and outside the L1), from the macro traces: the real-time NRx
    (small_c*) and the large NRx (nrx_c*) on the same slots and transmissions.  Disc*: retransmissions per 100 TBs
    with the real-time NRx as the recovery receiver (Full Wait, conventional L1) and with the real-time NRx as the L1
    and the large NRx as the recovery receiver; the share of the TBs the real-time NRx fails at the first transmission
    that the large NRx decodes; the share of the failed TBs of the conventional receiver that only the real-time NRx
    decodes; recovery runs per TB under Full Wait with either L1 (one for a failed first transmission, one for each
    failed retransmission of a TB not yet delivered)."""
    g, total = {}, None
    for s_ in MACRO_SETS:
        for f in sorted(glob.glob(f"results/{s_}{SUFFIX}/c*.npz")):
            d = np.load(f)
            total = int(d["total"])
            for k in d.files:
                if k.startswith(("conv_", "sep_", "nrx_", "small_")) or k == "cbfail_t0":
                    g.setdefault(k, []).append(d[k].reshape(-1))
    g = {k: np.concatenate(v) for k, v in g.items()}
    n = len(g["conv_t0"])
    cand = (g["cbfail_t0"] >= 0) & (g["cbfail_t0"] <= 5)

    def run(l1, rec, l1_key, rec_key):
        l1_steps = [None] + [g[f"{l1_key}{t}"] for t in range(1, total)]
        if rec_key is None:
            _, tx = simulate(l1, np.zeros(n, bool), l1_steps, cand, "harq_only", "DDDSU", Timing, l1_steps=l1_steps)
            return float(100.0 * (tx - 1).sum() / n), None
        steps = [None] + [g[f"{l1_key}{t}"] | g[f"{rec_key}{t}"] for t in range(1, total)]
        _, tx = simulate(l1, rec & ~l1, steps, cand, "wait", "DDDSU", Timing, l1_steps=l1_steps)
        runs = (~l1).astype(float)
        alive = ~l1 & ~rec
        for t in range(1, total):
            runs += alive & ~l1_steps[t]
            alive = alive & ~steps[t]
        return float(100.0 * (tx - 1).sum() / n), float(runs.mean())

    conv, small, large = g["conv_t0"], g["small_c0"], g["nrx_c0"]
    conventional, _ = run(conv, None, "sep_c", None)
    ai, _ = run(small, None, "small_c", None)
    small_rec, _ = run(conv, small, "sep_c", "small_c")
    large_rec, runs_conv = run(conv, large, "sep_c", "nrx_c")
    combo, runs_small = run(small, large, "small_c", "nrx_c")
    v = {"DiscSmallRecRetx": (small_rec, 1), "DiscComboRetx": (combo, 1),
         "DiscLargeOfSmallFailPct": (float(100.0 * (large & ~small).sum() / (~small).sum()), 0),
         "DiscSmallOnlyPct": (float(100.0 * (small & ~large & ~conv).sum() / (~conv).sum()), 2),
         "DiscRunsConv": (runs_conv, 3), "DiscRunsSmall": (runs_small, 3), "DiscRunsFewer": (runs_conv / runs_small, 1)}
    raw = {"ai_for_ran_retx_per_100": ai, "large_recovery_retx_per_100": large_rec, "combo_retx_per_100": combo,
           "conventional_retx_per_100": conventional, "small_recovery_retx_per_100": small_rec,
           "recovery_runs_per_100_conv_l1": 100.0 * runs_conv, "recovery_runs_per_100_small_l1": 100.0 * runs_small}
    return v, raw


def receivers(disc):
    """Macros of Section 5.3: recovery runs per 100 TBs with the conventional L1 and with the real-time NRx as the L1
    (from discussion()), and the failed TBs of the LMMSE receiver that each receiver decodes with four receive antennas
    (n5, as Figure 5(a)) and eight (n4b_eval_s30_uma: the large NRx fine-tuned for eight antennas, macro cell), with
    their share of the bound (K-best detection with the true channel)."""
    v = {"EvalRxRunsConv": (100.0 * disc["DiscRunsConv"][0], 1), "EvalRxRunsSmall": (100.0 * disc["DiscRunsSmall"][0], 1)}
    e = json.load(open(f"results/n4b_eval_s30_uma{SUFFIX}.json"))["recovers_pct_of_conventional_failures"]
    v["EvalEightRxNrx"], v["EvalEightRxKbest"], v["EvalEightRxBound"] = (e["nrx"], 0), (e["baseline_lmmse_kbest"], 0), (e["baseline_perf_csi_kbest"], 0)
    # small cells: the same receivers on the failed TBs of an LMMSE receiver (n5_umi sets, four antennas), weighted by
    # the failed TBs of each set as the macro sets of motivation()
    reps = [json.load(open(f)) for f in (f"results/{n}{SUFFIX}.json" for n in ("n5_umi_v3", "n5_umi_more1", "n5_umi_more2"))
            if os.path.exists(f)]
    if reps:
        w = np.array([r["fail_pct"]["baseline_lslin_lmmse"] * r["slots"] for r in reps])
        avg = lambda k: float(np.dot(w, [r["recovers_pct_of_conventional_failures"][k] for r in reps]) / w.sum())
        v["EvalBoundSmallNrx"], v["EvalBoundSmallTrue"] = (avg("nrx"), 0), (avg("baseline_perf_csi_kbest"), 0)
        v["EvalBoundShareSmall"] = (100.0 * avg("nrx") / avg("baseline_perf_csi_kbest"), 0)
    return v


def validation():
    """Macros of the checks of the server results (Sections 5.4, 5.6, 5.7, 5.10), from the stored summaries:
    N26 (n26_load_summary.py: real-time runs at loads above MCS 14 against the load-matched share), N29
    (n29_load_summary.py: cells that transmit in a part of the uplink periods), N25 (n25_la_summary.py: closed-loop link
    adaptation in the real-time runs), N28 (n22_summary.py ai_share: the MPS share of the AI process), the delay of
    Latency-Neutral on a loaded server (delay_under_load.py) and the same runs with the rejection at the latest start."""
    v = {}
    f = f"results/n26_load_summary{SUFFIX}.json"
    if os.path.exists(f):
        pts = [p for d in json.load(open(f)).values() for p in d["points"].values()]
        diff = [100.0 * (p["measured"] - p["predicted"]) for p in pts]
        v["EvalLoadAllPoints"] = (len(pts), 0)
        v["EvalLoadAllMax"] = (max(p["load"] for p in pts), 1)
        v["EvalLoadAllMinPp"], v["EvalLoadAllMaxPp"] = (min(diff), 1), (max(diff), 1)
        v["EvalLoadAllRunMax"] = (max(p["run_load"] for p in pts), 1)   # recovery runs per TB against MCS 14
    f = f"results/n29_load_summary{SUFFIX}.json"
    if os.path.exists(f):
        d = json.load(open(f))
        if d.get("rejection") == "L1 result":
            rows = [(load, r) for load, rs in d.items() if load != "rejection" for r in rs.values() if r["gpus"] == "four"]
            v["EvalPartialDiffMax"] = (max(abs(100.0 * (r["share"] - r["full_load_share_same_transmitting_cells"])) for _, r in rows), 1)
            r48 = d["a25b"]["four_48"]
            v["EvalPartialCellsMax"] = (r48["cells"], 0)
            v["EvalPartialRetx"], v["EvalPartialGood"] = (r48["retx_cut"], 2), (r48["goodput_gain_la"], 2)
            v["EvalPartialLateMax"] = (max(r["late_l1_pct"] for _, r in rows), 2)
    f = "results/n25_la_summary.json"
    if os.path.exists(f):
        sm = json.load(open(f))["summary"]
        if "g4_c4" in sm and "g4_c24" in sm:
            v["EvalLaRtFourFew"], v["EvalLaRtFourMax"] = (sm["g4_c4"]["gain_mean"], 2), (sm["g4_c24"]["gain_mean"], 2)
    n22 = json.load(open("results/n22_summary.json"))
    if "m80" in n22.get("ai_share", {}):
        base, m80 = n22["rows"], n22["ai_share"]["m80"]
        keys = [k for k in m80 if k.startswith("all_g4")]
        v["EvalAiShareRejectMax"] = (max(m80[k]["ai_rejected_pct"] for k in keys), 1)
        v["EvalAiShareRecDiffMax"] = (max(abs(m80[k]["recovered_pct"] - base[k]["recovered_pct"]) for k in keys), 1)
        v["EvalAiShareLateMax"] = (max(m80[k]["late_l1_pct"] for k in keys), 2)
    rec, _ = runtime_rows()
    tag = lambda g, x="": f"allb68{'g4' if g == 4 else ''}{N16}{x}"
    v["EvalArrivalDiffMax"] = (max(abs(rec[tag(g, ARRIVAL)][c] - rec[tag(g)][c]) for g in (1, 4) for c in rec[tag(g, ARRIVAL)]), 1)
    f, fm = f"results/delay_under_load{SUFFIX}.json", f"results/delay_under_load{SUFFIX}_margin015.json"
    if os.path.exists(f) and os.path.exists(fm):
        a, b = json.load(open(f)), json.load(open(fm))
        v["EvalDelayLatestPp"] = (-min(r["latency_neutral"]["at_latest_start_measured"]["min_diff_pp"] for r in a.values()), 1)
        v["EvalDelayArrivalPp"] = (-min(r["latency_neutral"]["at_arrival_runtime"]["min_diff_pp"] for r in b.values()
                                        if "at_arrival_runtime" in r["latency_neutral"]), 1)
    return v


# Section 5.3, antennas and users: the classical receivers (LMMSE receiver, K-best detection with the LMMSE estimate and
# with the true channel) and the large NRx fine-tuned for one to four users (4 RX: nrx_large_mu4 trained to 2026-11-03
# 13:00, U3n "_t"; 8 RX: variant c of U3s) run in separate runs of the same Eb/No points, slots and users.
USERS = {
    "Four": {"classical": {u: f"u3n_uma_r4_u{u}" for u in (2, 3, 4)}, "nrx": {u: f"u3n_uma_r4_u{u}_t" for u in (2, 3, 4)}},
    "Eight": {"classical": {1: "u3s_uma_r8_u1", 2: "u3s_uma_r8_u2", 3: "u3s_uma_r8_u3", 4: "u1_uma_r8_u4_b1"},
              "nrx": {u: f"u3s_nrx_r8_u{u}_c" for u in (1, 2, 3, 4)}},
}


def users_table():
    """{antennas: {users: {conv_fail, nrx, kbest, true}}}: the failed TBs of the LMMSE receiver (%) that each receiver
    decodes, each share of the LMMSE failures of its own run."""
    out = {}
    for ant, d in USERS.items():
        out[ant] = {}
        for u, name in d["classical"].items():
            c = json.load(open(f"results/{name}{SUFFIX}.json"))
            n = json.load(open(f"results/{d['nrx'][u]}{SUFFIX}.json"))
            out[ant][u] = {"conv_fail": c["fail_pct"]["baseline_lslin_lmmse"], "nrx_conv_fail": n["fail_pct"]["baseline_lslin_lmmse"],
                           "nrx": n["recovers_pct_of_conventional_failures"]["nrx"],
                           "kbest": c["recovers_pct_of_conventional_failures"]["baseline_lmmse_kbest"],
                           "true": c["recovers_pct_of_conventional_failures"]["baseline_perf_csi_kbest"]}
    return out


def users():
    """Macros of the antennas and users of Section 5.3 (ranges over the user counts of each antenna count)."""
    v, t = {}, users_table()
    for ant, rows in t.items():
        us = sorted(rows)
        for key, lab in (("nrx", "Nrx"), ("kbest", "Kbest"), ("conv_fail", "Fail")):
            xs = [rows[u][key] for u in us]
            v[f"EvalUsers{ant}{lab}Low"], v[f"EvalUsers{ant}{lab}High"] = (min(xs), 0), (max(xs), 0)
        v[f"EvalUsers{ant}TrueFew"], v[f"EvalUsers{ant}TrueMany"] = (rows[us[0]]["true"], 0), (rows[us[-1]]["true"], 0)
        v[f"EvalUsers{ant}Min"], v[f"EvalUsers{ant}Max"] = (us[0], 0), (us[-1], 0)
        v[f"EvalUsers{ant}SampleDiffMax"] = (max(abs(r["conv_fail"] - r["nrx_conv_fail"]) for r in rows.values()), 1)
    return v


def sensitivity():
    """Macros of Section 5.9 (docs/plan_setup_and_sensitivity.md, E1-E6), from the stored outputs of
    E1a src/analysis/fronthaul_replay.py (results/e1_fronthaul_replay.json): distance to the radio units in the replay,
        one-way delay 5 us/km, L1 result 1.4 + t, lead 4.0 + t, recovery deadline 11.0 - t after the slot;
    E1b src/analysis/fronthaul_realtime.py (results/e1_fronthaul_realtime.json): the real-time runs of Figure 14(d) read
        as distances (recovery deadline 11.0 - 2t after the samples arrive), four GPUs;
    E2, E3, E6 src/analysis/timing_sweeps.py (results/e2e3e6_timing_sweeps.json): TDD patterns, grant leads 1.5-6 ms,
        and the reductions per Eb/No point of the macro traces;
    E4 src/analysis/l1_deadline_recount.py (results/e4_l1_deadline_recount.json): late L1 results at thresholds 2.5-5.0 ms;
    E5 exp/harq_mcs.sh + src/sim/olla.py (results/n6_uma_v3h_f35_olla.json): outer-loop link adaptation with targets
        1-20% at SNR 14-24 dB, MCS 10-16."""
    v, checks = {}, {}
    rp = json.load(open("results/e1_fronthaul_replay.json"))
    rows = {float(k): r for k, r in rp["rows"].items()}
    w0, n0 = rows[0.0]["retx_cut_wait"], rows[0.0]["retx_cut_neutral"]
    same = [k for k in sorted(rows) if abs(rows[k]["retx_cut_wait"] - w0) < 1e-9 and abs(rows[k]["retx_cut_neutral"] - n0) < 1e-9]
    first_change = min(k for k in rows if k not in same)
    checks["e1a_unchanged_up_to_the_limit"] = bool(all(k < first_change for k in same))
    v["SensFhKmSame"], v["SensFhKmBreak"] = (max(same), 0), (first_change, 0)
    v["SensFhBreakCut"] = (rows[first_change]["retx_cut_wait"], 2)
    v["SensFhPerKmUs"] = (rp["params"]["us_per_km"], 0)
    # the recovery path ends this long before the recovery deadline, and the distance limit is the distance whose round
    # trip equals the margin: the last unchanged distance fits the margin, the first changed one does not
    margin = Timing.deadline - (Timing.l1 + Timing.run)
    v["SensRecoveryMarginMs"] = (margin, 1)
    rtt = lambda km: 2.0 * km * rp["params"]["us_per_km"] / 1000.0
    checks["e1a_limit_is_the_recovery_margin"] = bool(rtt(max(same)) <= margin + 1e-6 < rtt(first_change))
    rt = json.load(open("results/e1_fronthaul_realtime.json"))
    base_d = f"{Timing.deadline:g}"
    for d, lab in (("10", "Hundred"), ("9", "TwoHundred")):
        v[f"SensFhRt{lab}Km"] = (rt[d]["km"], 0)
        v[f"SensFhRt{lab}DropPp"] = (max(rt[base_d]["cells"][c]["recovered_on_time_pct"] - rt[d]["cells"][c]["recovered_on_time_pct"]
                                         for c in rt[d]["cells"]), 1)
    v["SensFhRtLateMax"] = (max(r["l1_late_pct"] for d in ("9", "10", base_d) for r in rt[d]["cells"].values()), 2)
    ts = json.load(open("results/e2e3e6_timing_sweeps.json"))
    cuts = [(r["wait"]["cut"], r["select_cb5"]["cut"]) for r in list(ts["e2_pattern"].values()) + list(ts["e3_lead"].values())]
    checks["e2e3_same_reductions_every_pattern_and_lead"] = bool(max(c[0] for c in cuts) - min(c[0] for c in cuts) < 1e-9
                                                                and max(c[1] for c in cuts) - min(c[1] for c in cuts) < 1e-9)
    checks["e2_latency_neutral_no_delay_every_pattern_at_the_reference_lead"] = bool(all(
        r["select_cb5"]["share_min_diff_pp"] >= 0.0 for k, r in ts["e2_pattern"].items() if r["timing"]["lead"] == Timing.lead))
    leads = sorted(r["timing"]["lead"] for r in ts["e3_lead"].values())
    v["SensLeadLow"], v["SensLeadHigh"] = (leads[0], 1), (leads[-1], 0)
    # Latency-Neutral at short grant leads: the smallest lead without a shortfall, and for the longest lead with one the
    # window in which the conventional receiver delivers more (rerun of latency_policies at that lead, DDDSU)
    neutral = {r["timing"]["lead"]: r["select_cb5"]["share_min_diff_pp"] for r in ts["e3_lead"].values()}
    v["SensNeutralLeadMin"] = (min(L for L, x in neutral.items() if x >= 0.0 and all(neutral[k] >= 0.0 for k in neutral if k >= L)), 0)
    short = max(L for L, x in neutral.items() if x < 0.0)
    v["SensNeutralShortLead"], v["SensNeutralShortPp"] = (short, 0), (-min(x for x in neutral.values()), 1)
    saved = (Timing.lead, Timing.deadline)
    Timing.lead, Timing.deadline = short, saved[0] + saved[1] - short
    c_, r_, s_, cs_, k_ = load_traces(MACRO_SETS)
    f_ = ~c_
    d0 = simulate(c_, r_, cs_, k_, "harq_only", "DDDSU", Timing, l1_steps=cs_)[0]
    d1 = simulate(c_, r_, s_, k_, "select_cb5", "DDDSU", Timing, l1_steps=cs_)[0]
    Timing.lead, Timing.deadline = saved
    v["SensNeutralShortRetxMs"] = (float(d0[f_ & np.isfinite(d0)].min()), 1)   # first conventional retransmission decoded
    checks["e3_short_lead_window_opens_at_the_first_retransmission"] = bool(v["SensNeutralShortRetxMs"][0] < Timing.l1 + Timing.run)
    at = np.unique(np.concatenate([[0.0], d0[f_ & np.isfinite(d0)], d1[f_ & np.isfinite(d1)]]))
    diff = delivered_share(d1, f_, at) - delivered_share(d0, f_, at)
    windows, i = [], 0                          # (from, to, largest shortfall) of each interval with a shortfall
    while i < len(at):
        if diff[i] < 0.0:
            j = i
            while j < len(at) and diff[j] < 0.0:
                j += 1
            windows.append((at[i], at[j] if j < len(at) else np.inf, -diff[i:j].min()))
            i = j
        else:
            i += 1
    checks["e3_short_lead_rerun_matches_sweep"] = bool(abs(-max(w[2] for w in windows) - neutral[short]) < 1e-9)
    checks["e3_short_lead_two_windows"] = bool(len(windows) == 2)
    for (a, b, pp), lab in zip(windows, ("One", "Two")):
        v[f"SensNeutralWin{lab}From"], v[f"SensNeutralWin{lab}To"], v[f"SensNeutralWin{lab}Pp"] = (a, 1), (b, 1), (pp, 1)
    v["SensTimingCutWait"], v["SensTimingCutNeutral"] = (cuts[0][0], 1), (cuts[0][1], 1)
    pts = ts["e6_point"]
    fail = {p: r["first_fail_pct"] for p, r in pts.items()}
    wait = {p: r["wait"]["cut"] for p, r in pts.items()}
    neutral = [r["select_cb5"]["cut"] for r in pts.values()]
    v["SensEbnoFailLow"], v["SensEbnoFailHigh"] = (min(fail.values()), 1), (max(fail.values()), 1)
    v["SensEbnoWaitLow"], v["SensEbnoWaitHigh"] = (min(wait.values()), 1), (max(wait.values()), 1)
    v["SensEbnoNeutralLow"], v["SensEbnoNeutralHigh"] = (min(neutral), 1), (max(neutral), 1)
    p_max, p_min = max(wait, key=wait.get), min(wait, key=wait.get)
    v["SensEbnoWaitHighFail"], v["SensEbnoWaitLowFail"] = (fail[p_max], 1), (fail[p_min], 1)
    e4 = json.load(open("results/e4_l1_deadline_recount.json"))
    late = lambda pol, t: max(r[f"late_pct_{t}"] for r in e4[pol].values())
    v["SensLOneThreeReverb"], v["SensLOneThreeNone"], v["SensLOneThreeTwo"] = (late("reverb", "3"), 2), (late("none", "3"), 2), (late("two", "3"), 1)
    v["SensLOneTwoHalfReverb"], v["SensLOneTwoHalfNone"] = (late("reverb", "2.5"), 2), (late("none", "2.5"), 2)
    v["SensLOneTwoHalfTwo"] = (late("two", "2.5"), 1)
    v["SensLOneCellsMax"] = (max(int(c) for c in e4["reverb"]), 0)
    la = json.load(open(f"results/n6_uma_v3h{SUFFIX}_olla.json"))
    res, ladder = la["results"], la["ladder"]
    snr = sorted(la["points"])
    gain = lambda t, s="ours": [res[f"snr{x:g}_target{t}"][s]["goodput_bits_per_tx"] / res[f"snr{x:g}_target{t}"]["ai_and_ran"]["goodput_bits_per_tx"]
                                for x in snr]
    field = lambda t, s, k: [res[f"snr{x:g}_target{t}"][s][k] for x in snr]
    v["SensLaSnrLow"], v["SensLaSnrHigh"] = (snr[0], 0), (snr[-1], 0)
    v["SensLaMcsLow"], v["SensLaMcsHigh"] = (ladder[0], 0), (ladder[-1], 0)
    mid, high = gain("0.03") + gain("0.05"), gain("0.1") + gain("0.2")
    v["SensLaMidLow"], v["SensLaMidHigh"] = (min(mid), 2), (max(mid), 2)
    v["SensLaHighLow"], v["SensLaHighHigh"] = (min(high), 2), (max(high), 2)
    ai_mid = gain("0.03", "ai_for_ran") + gain("0.05", "ai_for_ran")
    v["SensLaAiMidLow"], v["SensLaAiMidHigh"] = (min(ai_mid), 2), (max(ai_mid), 2)
    v["SensLaOneLow"], v["SensLaOneHigh"] = (min(gain("0.01")), 2), (max(gain("0.01")), 2)
    one_fail = field("0.01", "ai_and_ran", "first_fail_pct")
    v["SensLaOneConvFailLow"], v["SensLaOneConvFailHigh"] = (min(one_fail), 1), (max(one_fail), 1)
    checks["e5_conventional_at_lowest_mcs_for_one_percent"] = bool(max(field("0.01", "ai_and_ran", "mean_mcs")) == ladder[0])
    three_fail = field("0.03", "ai_and_ran", "first_fail_pct")
    v["SensLaThreeConvFailLow"], v["SensLaThreeConvFailHigh"] = (min(three_fail), 1), (max(three_fail), 1)
    v["SensLaHighMcsReverb"] = (min(field("0.1", "ours", "mean_mcs") + field("0.2", "ours", "mean_mcs")), 1)
    v["SensLaHighMcsConv"] = (min(field("0.1", "ai_and_ran", "mean_mcs") + field("0.2", "ai_and_ran", "mean_mcs")), 1)
    m3c, m3r = field("0.03", "ai_and_ran", "mean_mcs"), field("0.03", "ours", "mean_mcs")
    v["SensLaThreeMcsConvLow"], v["SensLaThreeMcsConvHigh"] = (min(m3c), 1), (max(m3c), 1)
    v["SensLaThreeMcsReverbLow"], v["SensLaThreeMcsReverbHigh"] = (min(m3r), 1), (max(m3r), 1)
    checks["e5_reverb_higher_mcs_at_three_percent_every_snr"] = bool(all(r > c for r, c in zip(m3r, m3c)))
    # at every SNR, the largest difference in mean MCS and the largest throughput gain are at a target of 3% or 5%
    tg = ("0.01", "0.03", "0.05", "0.1", "0.2")
    dm = {t: [r - c for r, c in zip(field(t, "ours", "mean_mcs"), field(t, "ai_and_ran", "mean_mcs"))] for t in tg}
    gn = {t: gain(t) for t in tg}
    checks["e5_largest_mcs_difference_and_gain_at_three_or_five_percent"] = bool(all(
        max(tg, key=lambda t: dm[t][i]) in ("0.03", "0.05") and max(tg, key=lambda t: gn[t][i]) in ("0.03", "0.05")
        for i in range(len(snr))))
    rx = lambda t, s: field(t, s, "resends_per_100")
    v["SensLaThreeRetxConvLow"], v["SensLaThreeRetxConvHigh"] = (min(rx("0.03", "ai_and_ran")), 1), (max(rx("0.03", "ai_and_ran")), 1)
    v["SensLaThreeRetxReverbLow"], v["SensLaThreeRetxReverbHigh"] = (min(rx("0.03", "ours")), 1), (max(rx("0.03", "ours")), 1)
    return v, checks


def main():
    allstats = json.load(open(f"results/n3_uma_v3_all{SUFFIX}.json"))["all"]
    lat = latency_policies()
    olla = json.load(open(f"results/n6_uma_v3{SUFFIX}_olla.json"))["results"]
    snr = sorted({float(k.split("_")[0][3:]) for k in olla})
    goodput = [olla[f"snr{x:g}_target0.1"]["ours"]["goodput_vs_ai_and_ran"] for x in snr]
    pf16 = json.load(open(f"results/p2b_uma_u16_pf{SUFFIX}.json"))["all"]
    rand = json.load(open(f"results/p2b_uma_u8_random{SUFFIX}.json"))["all"]

    conv_retx, ours_retx = allstats["ai_and_ran"]["resends_per_100"], allstats["ours"]["resends_per_100"]
    checks = {
        "latency_sim_full_wait_equals_harq_stats": abs(lat["wait"]["retx_per_100"] - ours_retx) < 1e-9,
        "latency_sim_conventional_equals_harq_stats": abs(lat["harq_only"]["retx_per_100"] - conv_retx) < 1e-9,
        "latency_neutral_delivers_at_least_conventional_within_every_delay": lat["select_cb5"]["share_min_diff_pp"] >= 0.0,
        "every_wait_adds_the_same_delay": bool(np.isfinite(lat["wait_delay_ms"])),
        "latency_sim_lost_equal_for_both_rules": abs(lat["wait"]["lost_pct"] - lat["select_cb5"]["lost_pct"]) < 1e-9,
    }
    disc, disc_raw = discussion()
    checks["discussion_ai_for_ran_equals_harq_stats"] = bool(abs(disc_raw["ai_for_ran_retx_per_100"] - allstats["ai_for_ran"]["resends_per_100"]) < 1e-9)
    checks["discussion_large_recovery_equals_full_wait"] = bool(abs(disc_raw["large_recovery_retx_per_100"] - ours_retx) < 1e-9)
    mot = motivation(lat)
    four = mot["MotRxLargeNrx"][0] / mot["MotRxBound"][0]
    e8 = json.load(open(f"results/n4b_eval_s30_uma{SUFFIX}.json"))["recovers_pct_of_conventional_failures"]
    eight = e8["nrx"] / e8["baseline_perf_csi_kbest"]
    disc["EvalBoundShareFour"], disc["EvalBoundShareEight"] = (100.0 * four, 0), (100.0 * eight, 0)
    checks["discussion_combination_as_on_the_conventional_l1"] = bool(abs(disc_raw["combo_retx_per_100"] - ours_retx) < 0.05)
    checks["discussion_large_decodes_every_first_tx_tb_the_real_time_nrx_decodes"] = bool(disc["DiscSmallOnlyPct"][0] == 0.0)
    sens, sens_checks = sensitivity()
    checks.update(sens_checks)
    values = {**motivation(lat), **evaluation(lat), **evaluation_more(), **validation(), **disc, **receivers(disc),
        **users(), **sens,
        "RetxCutFullWait": (conv_retx / ours_retx, 1),
        "RetxCutNoDelay": (lat["harq_only"]["retx_per_100"] / lat["select_cb5"]["retx_per_100"], 1),
        "LostCutFullWait": (allstats["ai_and_ran"]["residual_after_all_pct"] / allstats["ours"]["residual_after_all_pct"], 1),
        "ThroughputGainLA": (float(np.mean(goodput)), 2),
        "PairingCutBestSixteen": (pf16["ai_and_ran"]["resends_per_100"] / rand["ours"]["resends_per_100"], 1),
        "NeutralLaterPct": (lat["select_cb5"]["later_pct_of_failed"], 1),
        "NeutralEarlierPct": (lat["select_cb5"]["earlier_pct_of_failed"], 1),
        "FullWaitLaterPct": (lat["wait"]["later_pct_of_failed"], 1),
        "WaitAddedDelayMs": (lat["wait_delay_ms"], 1),
        "FullWaitMaxAddedMs": (lat["wait"]["added_delay_ms_max"], 1),
        "NeutralMaxAddedMs": (lat["select_cb5"]["added_delay_ms_max"], 1),
        "FullWaitBehindMaxPp": (-lat["wait"]["share_min_diff_pp"], 1),
        "FullWaitAheadFromMs": (lat["wait"]["ahead_from_ms"], 1),
    }
    # cost of recovery (Discussion, Table 3): A100 time of the recovery runs of a macro cell, every recovery on time, and
    # the retransmissions it avoids; one run occupies the GPU for the whole recovery path (PATH, as MotGpuTimePerSlot)
    runs = values["EvalRxRunsConv"][0]                        # recovery runs per 100 TBs with the conventional L1
    avoided = values["EvalRetxConvMacro"][0] - values["EvalRetxReverbMacro"][0]
    values["CostGpuPerHundred"] = (runs * PATH["p50_ms"], 0)
    values["CostRetxAvoided"] = (avoided, 1)
    values["CostGpuPerAvoided"] = (runs * PATH["p50_ms"] / avoided, 1)
    full = json.load(open(f"results/n3_uma_v3_all{SUFFIX}.json"))
    meta = json.loads(str(np.load(sorted(glob.glob(f"results/{MACRO_SETS[0]}{SUFFIX}/c*.npz"))[0])["meta"]))
    points = sorted(float(k) for k in full["by_point"])
    setup = {"SetupLOneMs": f"{Timing.l1:.1f}", "SetupRecoveryRunMs": f"{Timing.run:.1f}",
             "SetupRecoveryPathMs": f"{PATH['p50_ms']:.1f}", "SetupRecoveryBoundMs": f"{BOUND_MS:.1f}", "SetupArrivalMarginMs": f"{ARRIVAL_MARGIN_MS:.2f}",
             "SetupBoundLowMs": f"{BOUND_LOW_MS:.1f}", "SetupBoundHighMs": f"{BOUND_HIGH_MS:.1f}",
             "SetupLOneDeadlineMs": f"{L1_DEADLINE_MS:.1f}",
             "SetupRecoveryDeadlineMs": f"{Timing.deadline:.1f}", "SetupLeadMs": f"{Timing.lead:.1f}",
             "SetupRetxSlotMs": f"{Timing.deadline + Timing.lead:.1f}",
             "SetupSlots": f"{full['slots']:,}", "SetupTBs": f"{full['slots'] * full['users']:,}",
             "SetupTbBytes": f"{meta['tb_bytes']:,}", "SetupTransmissions": f"{full['transmissions']}",
             "SetupRetxGapMs": f"{meta['generator']['gap_ms']:.1f}",
             "SetupEbNoLow": f"{points[0]:g}", "SetupEbNoHigh": f"{points[-1]:g}",
             "SetupCarrierGhz": f"{meta['generator'].get('carrier_ghz', 2.14):g}"}
    os.makedirs(OUT, exist_ok=True)
    report = {"values": {k: v for k, (v, _) in values.items()}, "setup": setup,
              "inputs": {"retx_per_100": {"conventional": conv_retx, "reverb_full_wait": ours_retx,
                                          "latency_policies": lat},
                         "lost_pct": {"conventional": allstats["ai_and_ran"]["residual_after_all_pct"],
                                      "reverb_full_wait": allstats["ours"]["residual_after_all_pct"]},
                         "goodput_vs_conventional_by_snr": dict(zip([f"{x:g}" for x in snr], goodput)),
                         "pairing_retx_per_100": {"conventional_best_of_16_sum_rate": pf16["ai_and_ran"]["resends_per_100"],
                                                  "reverb_random_partner": rand["ours"]["resends_per_100"]},

                         "recovery_path": PATH},
              "checks": checks}
    json.dump(report, open(f"{OUT}/numbers.json", "w"), indent=1, sort_keys=True)
    lines = ["% Generated by airan_recovery/src/analysis/paper_numbers.py from the stored results.  Do not edit.",
             "% Values, sources and checks: airan_recovery/results/paper/numbers.json"]
    for k, (v, digits) in values.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v:,.{digits}f}}}")   # thousands separators (WRITING_RULES 5)
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

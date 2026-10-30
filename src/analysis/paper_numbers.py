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
  runtime (src/analysis/recovery_path.py); \\SetupRecoveryBoundMs, \\SetupBoundLowMs and \\SetupBoundHighMs are settings of
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
        v[f"MotEstShare{tag}"] = (100 * (est - true) / est, 0)
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
    r4, r1 = rec[tag("allb68", 4)], rec[tag("allb68", 1)]
    v["EvalGpuFourFew"], v["EvalGpuFourMid"], v["EvalGpuFourMax"] = (r4[4], 1), (r4[16], 1), (r4[24], 1)
    v["EvalGpuOneFew"], v["EvalGpuOneMid"], v["EvalGpuOneMax"] = (r1[1], 1), (r1[4], 1), (r1[8], 1)
    for name, p, x in (("Before", "parab68", ""), ("NoReject", "allb68", "admit0")):
        b4, b1 = rec[tag(p, 4, x)], rec[tag(p, 1, x)]
        v[f"EvalGpuVs{name}FourMid"], v[f"EvalGpuVs{name}FourMax"] = (r4[16] / b4[16], 1), (r4[24] / b4[24], 1)
        v[f"EvalGpuVs{name}OneMid"], v[f"EvalGpuVs{name}OneMax"] = (r1[4] / b1[4], 1), (r1[8] / b1[8], 1)
    v["EvalGpuBeforeNoRejectFourMax"] = (max(rec[tag("parb68", 4)].values()), 1)
    v["EvalGpuTwoFourMid"] = (rec[tag("allb68", 4, "l2")][16], 1)
    l4 = {k: late[tag(p, 4, x)] for k, p, x in (("reverb", "allb68", ""), ("none", "offb68", ""), ("two", "allb68", "l2"),
                                                 ("four", "allb68", "l4"))}
    v["EvalLateReverbMax"] = (max(l4["reverb"].values()), 2)
    v["EvalLateNoneMax"] = (max(l4["none"].values()), 2)
    v["EvalLateDiffMax"] = (max(l4["reverb"][c] - l4["none"][c] for c in l4["reverb"]), 2)
    v["EvalLateTwoMid"], v["EvalLateTwoMax"] = (l4["two"][16], 1), (l4["two"][24], 1)
    v["EvalLateFourMid"], v["EvalLateFourMax"] = (l4["four"][16], 1), (l4["four"][24], 1)
    carrier = N16[N16.index("_"):] if "_" in N16 else ""
    d1 = {d: rec[f"allb68d{d}{carrier}"][4] for d in (90, 100, 110, 120, 130)}
    d4 = {d: rec[f"allb68g4d{d}{carrier}"][16] for d in (90, 100, 110, 120, 130)}
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
    chk = eff.get("load_matching_check", {})
    if chk:
        v["EvalLoadCheckMaxPp"] = (max(100.0 * abs(c["measured"] - c["predicted"]) for c in chk.values()), 0)
        v["EvalLoadCheckFailPct"] = (100.0 * float(np.mean([c["failed_per_tb"] for c in chk.values()])), 1)
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
        base = rec[f"allb68{'g4' if g == 4 else ''}{N16}"]
        for r in (4, 8):
            keep += [rows[f"all_g{g}_r{r}_c{c}"]["recovered_pct"] / base[c] for c in cells]
    v["EvalAiKeepLow"], v["EvalAiKeepHigh"] = (100.0 * min(keep), 0), (100.0 * max(keep), 0)
    v["EvalAiRecFourMid"] = (rows["all_g4_r4_c16"]["recovered_pct"], 1)
    v["EvalAiRecFourMidEight"] = (rows["all_g4_r8_c16"]["recovered_pct"], 1)
    v["EvalAiRecFourMidNone"] = (rec[f"allb68g4{N16}"][16], 1)
    allrec = [r for r in rows.values() if r["policy"] == "all"]
    v["EvalAiLateMax"] = (max(r["late_l1_pct"] for r in allrec), 2)
    v["EvalAiLateOffMax"] = (max(r["late_l1_pct"] for r in rows.values() if r["policy"] == "off"), 2)
    v["EvalAiSloLow"] = (min(r["ai_within_slo_pct"] for r in allrec), 1)
    v["EvalAiSloLowFour"] = (min(rows[f"all_g4_r4_c{c}"]["ai_within_slo_pct"] for c in (8, 16, 24)), 1)
    v["EvalAiSloOffLow"] = (min(r["ai_within_slo_pct"] for r in rows.values() if r["policy"] == "off"), 1)
    v["EvalAiRejectMax"] = (max(r["ai_rejected_pct"] for r in allrec), 1)
    v["EvalAiTtftOffMax"] = (max(r["ai_ttft_p99_ms"] for r in rows.values() if r["policy"] == "off"), 0)
    v["EvalAiTtftMax"] = (max(r["ai_ttft_p99_ms"] for r in allrec), 0)
    # inter-cell interference: macro and small cells, 10 dB (two users of neighbor cells) and 20 dB with two users
    def load(n):
        f = f"results/{n}{SUFFIX}.json"
        return json.load(open(f))["all"] if os.path.exists(f) else None
    for cell, pre10, pre20 in (("Macro", "ici_uma_inr10", "i1_uma_k2_inr20"), ("Small", "ici_umi_inr10", "i1_umi_k2_inr20")):
        for cond, pre in (("Ten", pre10), ("Twenty", pre20)):
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
    values = {**motivation(lat), **evaluation(lat), **evaluation_more(),
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
    full = json.load(open(f"results/n3_uma_v3_all{SUFFIX}.json"))
    meta = json.loads(str(np.load(sorted(glob.glob(f"results/{MACRO_SETS[0]}{SUFFIX}/c*.npz"))[0])["meta"]))
    points = sorted(float(k) for k in full["by_point"])
    setup = {"SetupLOneMs": f"{Timing.l1:.1f}", "SetupRecoveryRunMs": f"{Timing.run:.1f}",
             "SetupRecoveryPathMs": f"{PATH['p50_ms']:.1f}", "SetupRecoveryBoundMs": f"{BOUND_MS:.1f}",
             "SetupBoundLowMs": f"{BOUND_LOW_MS:.1f}", "SetupBoundHighMs": f"{BOUND_HIGH_MS:.1f}",
             "SetupLOneDeadlineMs": f"{L1_DEADLINE_MS:.1f}",
             "SetupRecoveryDeadlineMs": f"{Timing.deadline:.1f}", "SetupLeadMs": f"{Timing.lead:.1f}",
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

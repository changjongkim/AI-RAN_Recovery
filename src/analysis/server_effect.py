#!/usr/bin/env python3
"""Server-level effect of Reverb: the trace-driven results with only the recoveries that finish on time on shared GPUs.

1. On-time share, from the real-time runs (N16, S1; policy allb68[g4]<N16_SUFFIX>, every seed; warm-up periods left out
   as in the run summary): the share of the recoverable failed TBs that a run recovers by the recovery deadline.
   Recoverable TBs are estimated from the slots the run decoded: in those slots, the TBs that the L1 failed
   (conv<cell>.json, ue_crc_mask) and the NRx decoded (lane<gpu>.json, ue_good_mask) over the TBs the L1 failed; the
   rejected slots have the same channels, so the failed TBs of the run times this share is the number of recoverable
   TBs.  The same records give the on-time share per Eb/No of the slot (ring index -> dataset esno_db), to check that a
   recovery finishes on time independently of the channel of its TB.
2. Recovery load.  The real-time runs replay first transmissions at MCS 14 and run only their recoveries.  With Reverb,
   the GPUs also decode the retransmissions that the L1 fails (Full Wait), and link adaptation raises the MCS and with it
   the TBs the L1 fails.  The load factor is the recovery runs per TB of the evaluated operating point over the failed TBs
   per TB of the real-time runs; the on-time share at c cells is the measured share at c x load (interpolated over the
   measured cells, and beyond the largest measured number of cells the fluid limit share(c_max) x c_max / (c x load)).
   Load factors: the timeline at MCS 14 (first failures + recoveries of retransmissions) and link adaptation (the same
   counts at the MCS that outer-loop link adaptation selects, mean over the SNR points of the macro cell).
3. Effect: each recovery of the macro traces (Full Wait) is kept with probability equal to the load-matched on-time share
   (one random draw per TB and transmission, seed 7), the first recovery and the NRx decodes of the retransmissions
   alike, and the timeline simulation (src/sim/harq_latency.py, the timing of paper_numbers.py) and the outer-loop link
   adaptation of src/sim/olla.py (target 10%, SNR points of N6, common random numbers) give retransmissions per 100 TBs,
   lost TBs, and goodput.  The rows also keep the effect with the measured share (no load matching) for comparison.
4. Check of the load matching (N23, when its runs exist): the runs on MCS 16 slots fail more TBs; the share predicted from
   the MCS 14 runs at c x (failures at MCS 16 / failures at MCS 14) against the measured share.
5. Check of the extrapolation: beyond the largest measured number of cells, the on-time recoveries per second fall further
   at the rate between the last two measured numbers of cells (instead of the fluid limit), for the rows that extrapolate.
6. Check with the records of each task: at the cell count c' = c x load (fixed MCS), each TB takes the on-time outcome of a
   recorded recovery request of the c' runs (the first transmission from a random request, transmission t from a random
   request of the period t x lag later, lag 6 and 8 uplink periods: the retransmission slot without and with a wait),
   against one draw per TB and transmission at the on-time share of the same runs.
usage: RESULT_SUFFIX=_f35 N16_SUFFIX=d110_f35 server_effect.py [OUT.json]       (from the repository root)"""
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "src/sim")
sys.path.insert(0, "src/analysis")
import harq_stats  # noqa: E402
import paper_numbers as pn  # noqa: E402
from harq_latency import simulate  # noqa: E402
import olla  # noqa: E402

RAW = "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw"
SUFFIX, N16 = pn.SUFFIX, pn.N16


def run_records(f):
    """Failed TBs, on-time recoveries, and the NRx result of the decoded slots of one run, in total and per Eb/No."""
    d = json.load(open(f))
    cfg, work = d["config"], f[:-5] + "_work"
    skip, deadline_ns = int(cfg["skip_periods"]), float(cfg["rescue_deadline_ms"]) * 1e6
    metas = sorted(glob.glob(os.path.join(cfg["dataset"], "*_meta.json")))   # the meta of the profile the run used
    text = json.dumps(cfg)
    meta = next((m for m in metas if not m.endswith("/nv_mu2_meta.json") and f'"{os.path.basename(m)[:-10]}"' in text),
                os.path.join(cfg["dataset"], "nv_mu2_meta.json"))
    esno = json.load(open(meta))["esno_db"]
    conv = {}
    for cf in glob.glob(f"{work}/conv*.json"):
        cell = int(re.search(r"conv(\d+)\.json$", cf).group(1))
        c = json.load(open(cf))
        col = {k: i for i, k in enumerate(c["columns"])}
        for r in c["records"]:
            if int(r[col["period"]]) >= skip:
                conv[(cell, int(r[col["period"]]))] = (int(r[col["ue_crc_mask"]]), int(r[col["release_ns"]]),
                                                        esno[int(r[col["ring_index"]])])
    tot = {"failed": 0, "on_time": 0, "failed_run": 0, "recoverable_run": 0}
    per = {}
    for mask, _, e in conv.values():
        n = bin(~mask & 0b11).count("1")
        tot["failed"] += n
        per.setdefault(e, dict(tot, failed=0, on_time=0, failed_run=0, recoverable_run=0))["failed"] += n
    for lf in glob.glob(f"{work}/lane*.json"):
        l = json.load(open(lf))
        col = {k: i for i, k in enumerate(l["columns"])}
        for r in l["records"]:
            key = (int(r[col["cell"]]), int(r[col["period"]]))
            if key not in conv:
                continue
            mask, rel, e = conv[key]
            failed = ~mask & 0b11
            good = bin(failed & int(r[col["ue_good_mask"]])).count("1")
            for t in (tot, per[e]):
                t["failed_run"] += bin(failed).count("1")
                t["recoverable_run"] += good
                t["on_time"] += good if int(r[col["done_ns"]]) - rel <= deadline_ns else 0
    return tot, per, d["all"]["tbs"]


def share(t):
    rec = t["recoverable_run"] / max(1, t["failed_run"])
    return min(1.0, t["on_time"] / max(1e-9, t["failed"] * rec))


def availability(tags=None):
    """On-time share per (GPUs, cells), with the failure rate per TB of the runs and the share per Eb/No."""
    tags = tags or {f"allb68{N16}": "one", f"allb68g4{N16}": "four"}
    rows = {}
    for f in glob.glob(f"{RAW}/rec16*_c*_*_none_j*.json"):
        m = re.match(r"rec16(\w+?)s(\d+)_c(\d+)_\w+_none_j(\d+)\.json", os.path.basename(f))
        if not m or m.group(1) not in tags:
            continue
        rows.setdefault((tags[m.group(1)], int(m.group(3))), []).append(run_records(f))
    out = {}
    for key, runs in sorted(rows.items()):
        tot = {k: sum(r[0][k] for r in runs) for k in runs[0][0]}
        per = {}
        for _, p, _ in runs:
            for e, t in p.items():
                acc = per.setdefault(e, {k: 0 for k in t})
                for k in t:
                    acc[k] += t[k]
        tbs = sum(r[2] for r in runs)
        out[key] = {"gpus": key[0], "cells": key[1], "recoverable_share": tot["recoverable_run"] / max(1, tot["failed_run"]),
                    "availability": share(tot), "failed_per_tb": tot["failed"] / tbs,
                    "availability_by_esno": {f"{e:g}": share(t) for e, t in sorted(per.items())}}
    return out


def matched(avail, gpus, cells, load, decline=False):
    """On-time share at `cells` cells with `load` times the recovery load of the real-time runs.  Beyond the measured
    cells: the fluid limit (the on-time recoveries of the largest number of cells), or with `decline` these recoveries
    falling further at the rate between the last two measured numbers of cells."""
    pts = sorted((r["cells"], r["availability"]) for r in avail.values() if r["gpus"] == gpus)
    c = np.array([p[0] for p in pts], float)
    a = np.array([p[1] for p in pts], float)
    x = cells * load
    if x <= c[-1]:
        return float(np.interp(x, c, a))
    n_last = a[-1] * c[-1]
    if decline:
        slope = min(0.0, (a[-1] * c[-1] - a[-2] * c[-2]) / (c[-1] - c[-2]))
        n_last = max(0.0, n_last + slope * (x - c[-1]))
    return float(n_last / x)


def recovery_runs(first_ok, recovered, l1_steps, steps, total):
    """Recovery runs per TB under Full Wait: one for a TB the L1 fails, and one for each retransmission the L1 fails while
    the TB is not delivered."""
    runs = (~first_ok).astype(float)
    alive = ~first_ok & ~recovered
    for t in range(1, total):
        runs += alive & ~l1_steps[t]
        alive = alive & ~steps[t]
    return runs


def effect(avail):
    conv, rec, steps, conv_steps, cand = pn.load_traces(pn.MACRO_SETS)
    rng = np.random.default_rng(7)
    draw = rng.random(len(conv))
    T = pn.Timing
    d0, tx0 = simulate(conv, rec, conv_steps, cand, "harq_only", "DDDSU", T, l1_steps=conv_steps)
    root = Path(f"results/n6_uma_v3{SUFFIX}")
    tab = olla.per_mcs(root)
    ladder = sorted(tab)
    plans_m, runs_m = {}, {}
    for mcs in ladder:
        fs = sorted(glob.glob(str(root / f"m{mcs}_c*.npz")), key=lambda f: int(re.search(r"_c(\d+)\.npz$", f).group(1)))
        good, point, corr, _, _, total, _ = harq_stats.load(fs)
        flat = {k: v.reshape(-1) for k, v in good.items()}
        pl = harq_stats.plans(flat, total)
        plans_m[mcs] = (pl["ours"], pl["ours_conv_harq"][2], total)
        (f, rr, stp), st_conv, _ = plans_m[mcs]
        runs_m[mcs] = recovery_runs(f, rr, st_conv, stp, total)
    points = sorted(set(tab[ladder[0]]["point"].tolist()))
    draw_la = np.random.default_rng(11).random(len(tab[ladder[0]]["point"]))

    # recovery load of the timeline at MCS 14 and of link adaptation, in recovery runs per TB
    load_fixed = float(recovery_runs(conv, rec & ~conv, conv_steps, steps, len(steps)).mean())
    la_runs = []
    for q in points:
        sel = tab[ladder[0]]["point"] == q
        idx, r = np.flatnonzero(sel), np.random.default_rng(5)
        x, up, n, acc = float(ladder[len(ladder) // 2]), 0.25 * 0.1 / 0.9, 0, 0.0
        for k in range(20000):
            m = int(np.clip(round(x), ladder[0], ladder[-1]))
            i = idx[r.integers(0, len(idx))]
            x = float(np.clip(x + (up if tab[m]["first"]["ours"][i] else -0.25), ladder[0] - 0.49, ladder[-1] + 0.49))
            if k >= 2000:
                acc += runs_m[m][i]
                n += 1
        la_runs.append(acc / n)
    load_la = float(np.mean(la_runs))

    n_steps = len(steps)
    draws = [np.random.default_rng(100 + t).random(len(conv)) for t in range(n_steps)]

    def run(a_fix, a_la):
        r = rec & (draw < a_fix)
        st = [None] + [conv_steps[t] | (steps[t] & ~conv_steps[t] & (draws[t] < a_fix)) for t in range(1, n_steps)]
        d, tx = simulate(conv, r, st, cand, "wait", "DDDSU", T, l1_steps=conv_steps)
        for mcs in ladder:
            (f, rr, stp), st_conv, total_m = plans_m[mcs]
            r2 = rr & (draw_la < a_la)
            st2 = [None] + [st_conv[t] | (stp[t] & ~st_conv[t] & (np.random.default_rng(200 + t).random(len(f)) < a_la))
                            for t in range(1, total_m)]
            tab[mcs]["first"]["ours_thin"] = f | r2
            tab[mcs]["n_tx"]["ours_thin"] = harq_stats.transmissions(f, r2, st2, total_m)
        gains = []
        for q in points:
            sel = tab[ladder[0]]["point"] == q
            base = olla.simulate(tab, ladder, sel, "ai_and_ran", 0.1, 20000, 0.25, np.random.default_rng(5))
            ours = olla.simulate(tab, ladder, sel, "ours_thin", 0.1, 20000, 0.25, np.random.default_rng(5))
            gains.append(ours["goodput_bits_per_tx"] / base["goodput_bits_per_tx"])
        return {"retx_per_100": float(100.0 * (tx - 1).sum() / len(tx)), "lost_pct": float(100.0 * (~np.isfinite(d)).mean()),
                "goodput_gain_la": float(np.mean(gains))}

    base = {"retx_per_100": float(100.0 * (tx0 - 1).sum() / len(tx0)), "lost_pct": float(100.0 * (~np.isfinite(d0)).mean())}
    rows = {"every_recovery": run(1.0, 1.0)}
    for (g, c), v in avail.items():
        lf, ll = load_fixed / v["failed_per_tb"], load_la / v["failed_per_tb"]
        a_fix, a_la = matched(avail, g, c, lf), matched(avail, g, c, ll)
        rows[f"{g}_{c}"] = {**v, "load_fixed": lf, "load_la": ll, "availability_fixed": a_fix, "availability_la": a_la,
                            **run(a_fix, a_la), "unmatched": run(v["availability"], v["availability"])}
        d_fix, d_la = matched(avail, g, c, lf, True), matched(avail, g, c, ll, True)
        if (d_fix, d_la) != (a_fix, a_la):
            rows[f"{g}_{c}"]["declining"] = {"availability_fixed": d_fix, "availability_la": d_la, **run(d_fix, d_la)}
    # check of a single share for all TBs: the share of each Eb/No point of the real-time runs (scaled to the load-matched
    # share) applied to the TBs of that point, against one share for every TB
    pts = np.concatenate([np.repeat(np.load(f)["point"], np.load(f)["conv_t0"].shape[1]) for s_ in pn.MACRO_SETS
                          for f in sorted(glob.glob(f"results/{s_}{SUFFIX}/c*.npz"))])
    pts_la = tab[ladder[0]]["point"]
    strat = {}
    for g, c in (("four", 8), ("four", 16), ("one", 2)):
        rw = rows[f"{g}_{c}"]
        by = {float(e): x for e, x in avail[(g, c)]["availability_by_esno"].items()}
        scale = lambda a_m: (lambda pv: np.array([min(1.0, a_m * by.get(float(q), rw["availability"]) / rw["availability"]) for q in pv]))
        r_s = run(scale(rw["availability_fixed"])(pts), scale(rw["availability_la"])(pts_la))
        strat[f"{g}_{c}"] = {"retx_per_100": r_s["retx_per_100"], "goodput_gain_la": r_s["goodput_gain_la"],
                             "single_retx_per_100": rw["retx_per_100"], "single_goodput_gain_la": rw["goodput_gain_la"]}
    loads = {"recovery_runs_per_tb_fixed_mcs14": load_fixed, "recovery_runs_per_tb_la": load_la,
             "recovery_runs_per_tb_la_by_snr": dict(zip([f"{q:g}" for q in points], la_runs)),
             "first_failures_per_tb_fixed_mcs14": float((~conv).mean())}
    loads["by_esno_check"] = strat
    return base, rows, loads


def check_load_matching(avail):
    """N23: on-time share of the runs on MCS 16 slots against the share predicted from the MCS 14 runs."""
    m16 = availability({f"allb68{N16}_m16": "one", f"allb68g4{N16}_m16": "four"})
    out = {}
    for (g, c), v in m16.items():
        f14 = np.mean([r["failed_per_tb"] for r in avail.values() if r["gpus"] == g])
        load = v["failed_per_tb"] / f14
        out[f"{g}_{c}"] = {"gpus": g, "cells": c, "measured": v["availability"], "predicted": matched(avail, g, c, load),
                           "load": load, "failed_per_tb": v["failed_per_tb"]}
    return out


def requests(f):
    """On-time outcome of every failed TB of one run (1 if its slot finished by the recovery deadline), per period."""
    d = json.load(open(f))
    cfg, work = d["config"], f[:-5] + "_work"
    skip, deadline_ns = int(cfg["skip_periods"]), float(cfg["rescue_deadline_ms"]) * 1e6
    rel = {}
    for cf in glob.glob(f"{work}/conv*.json"):
        cell = int(re.search(r"conv(\d+)\.json$", cf).group(1))
        c = json.load(open(cf))
        col = {k: i for i, k in enumerate(c["columns"])}
        for r in c["records"]:
            n = bin(~int(r[col["ue_crc_mask"]]) & 0b11).count("1")
            if int(r[col["period"]]) >= skip and n:
                rel[(cell, int(r[col["period"]]))] = (n, int(r[col["release_ns"]]))
    ok = dict.fromkeys(rel, 0)
    for lf in glob.glob(f"{work}/lane*.json"):
        l = json.load(open(lf))
        col = {k: i for i, k in enumerate(l["columns"])}
        for r in l["records"]:
            key = (int(r[col["cell"]]), int(r[col["period"]]))
            if key in rel and int(r[col["done_ns"]]) - rel[key][1] <= deadline_ns:
                ok[key] = 1
    per = {}
    for (cell, p), (n, _) in rel.items():
        per.setdefault(p, []).extend([ok[(cell, p)]] * n)
    return per


def per_task_check(rows):
    """Check 6: the retransmissions with the recorded outcome of each recovery request against one on-time share."""
    conv, rec, steps, conv_steps, cand = pn.load_traces(pn.MACRO_SETS)
    n, total = len(conv), len(steps)

    def replay(flags):
        st = [None] + [conv_steps[t] | (steps[t] & ~conv_steps[t] & flags[t]) for t in range(1, total)]
        d, tx = simulate(conv, rec & flags[0], st, cand, "wait", "DDDSU", pn.Timing, l1_steps=conv_steps)
        return float(100.0 * (tx - 1).sum() / len(tx)), float(100.0 * (~np.isfinite(d)).mean())

    tags = {"one": f"allb68{N16}", "four": f"allb68g4{N16}"}
    out = {}
    for g, c in (("four", 8), ("four", 16), ("one", 2), ("one", 4)):
        cp = int(round(c * rows[f"{g}_{c}"]["load_fixed"]))
        runs = [requests(f) for f in sorted(glob.glob(f"{RAW}/rec16{tags[g]}s*_c{cp}_*_none_j*.json"))]
        if not runs:
            continue
        flat = [(i, p) for i, per in enumerate(runs) for p, v in per.items() for _ in v]
        on = sum(sum(v) for per in runs for v in per.values()) / len(flat)
        rng = np.random.default_rng(3)
        ind = replay(rng.random((total, n)) < on)
        res = {"cells_run": cp, "on_time_share": on, "independent_retx_per_100": ind[0], "independent_lost_pct": ind[1]}
        for lag in (6, 8):
            pick, flags = rng.integers(0, len(flat), n), np.zeros((total, n), bool)
            for j, k in enumerate(pick):
                i, p = flat[k]
                per = runs[i]
                for t in range(total):
                    q = p + t * lag
                    q = next((q + s for d_ in range(60) for s in (d_, -d_) if q + s in per), q)
                    v = per.get(q, [0])
                    flags[t, j] = v[rng.integers(0, len(v))]
            r = replay(flags)
            res[f"lag{lag}"] = {"retx_per_100": r[0], "lost_pct": r[1], "share_later": float(flags[1:].mean())}
        out[f"{g}_{c}"] = res
    return out


if __name__ == "__main__":
    av = availability()
    base, rows, loads = effect(av)
    check = check_load_matching(av)
    tasks = per_task_check(rows)
    out = sys.argv[1] if len(sys.argv) > 1 else f"results/server_effect{SUFFIX}.json"
    json.dump({"conventional": base, "rows": rows, "loads": loads, "load_matching_check": check, "per_task_check": tasks},
              open(out, "w"), indent=1, sort_keys=True)
    print(f"conventional: retx {base['retx_per_100']:.1f}, lost {base['lost_pct']:.2f}%")
    print("loads (recovery runs per TB):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in loads.items()})
    print("| GPUs | cells | on-time share | load fixed / LA | matched share fixed / LA | cut | goodput (LA) | unmatched cut / goodput |")
    for k, r in rows.items():
        if k == "every_recovery":
            print(f"| every recovery | | 1.00 | | | {base['retx_per_100'] / r['retx_per_100']:.2f}x | {r['goodput_gain_la']:.2f}x | |")
            continue
        u = r["unmatched"]
        print(f"| {r['gpus']} | {r['cells']} | {r['availability']:.2f} | {r['load_fixed']:.2f} / {r['load_la']:.2f} | "
              f"{r['availability_fixed']:.2f} / {r['availability_la']:.2f} | {base['retx_per_100'] / r['retx_per_100']:.2f}x | "
              f"{r['goodput_gain_la']:.2f}x | {base['retx_per_100'] / u['retx_per_100']:.2f}x / {u['goodput_gain_la']:.2f}x |")
    print("on-time share by Eb/No:", {k: r["availability_by_esno"] for k, r in av.items() if k[1] in (8, 16, 24, 4)})
    if check:
        print("load matching check (MCS 16):", {k: (round(v["measured"], 2), round(v["predicted"], 2), round(v["load"], 2)) for k, v in check.items()})
    print("declining beyond the measured cells:", {k: (round(base["retx_per_100"] / r["retx_per_100"], 3),
          round(base["retx_per_100"] / r["declining"]["retx_per_100"], 3), round(r["goodput_gain_la"], 3),
          round(r["declining"]["goodput_gain_la"], 3)) for k, r in rows.items() if "declining" in r})
    print("per-task check:", json.dumps(tasks, indent=1))

#!/usr/bin/env python3
"""Delay of the two Recovery Gate rules on a shared server: the macro traces replayed with the load-matched on-time share
of server_effect.py, where a recovery that is not on time is rejected on the GPU (harq_latency.simulate, rejected=).

A waiting TB whose recovery is rejected receives the grant at the rejection time.  The slot runtime and Section 4.3
reject a slot when its latest start time D - r passes (4.2 ms after the slot), after the last grant time D0 for the
first retransmission slot, so the retransmission moves one uplink period later.  The alternative rejects the slot at
its arrival (the L1 result), when the queue already shows that the slot cannot start by D - r, and the grant keeps the
first retransmission slot.  Latency-Neutral waits only for its candidates (at most 5 of 10 code blocks failed); its
critical recoveries (waiting first transmissions and every retransmission) see 0.082 recovery runs per TB, as the
scheduler serves waiting slots first, and the recoveries of the TBs granted at once are not counted.  Full Wait sees
0.200 runs per TB.  For each server: the smallest difference (percentage points) between the delivered share of the
rule and of the conventional receiver over every delay, and the retransmission reduction.
usage: RESULT_SUFFIX=_f35 N16_SUFFIX=d110_f35 delay_under_load.py [OUT.json]       (from the repository root)"""
import json
import sys

import numpy as np

sys.path[:0] = ["src", "src/sim", "src/analysis"]
import paper_numbers as pn  # noqa: E402
import server_effect as se  # noqa: E402
from harq_latency import simulate  # noqa: E402


def main():
    conv, rec, steps, conv_steps, cand = pn.load_traces(pn.MACRO_SETS)
    total, n, fail = len(steps), len(conv), ~conv
    avail = se.availability()
    d0, tx0 = simulate(conv, rec, conv_steps, cand, "harq_only", "DDDSU", pn.Timing, l1_steps=conv_steps)
    draws = [np.random.default_rng(100 + t).random(n) for t in range(total)]
    loads = {"latency_neutral": 0.082, "full_wait": 0.200}
    reject_times = {"at_latest_start": pn.Timing.deadline - pn.BOUND_MS, "at_arrival": pn.Timing.l1}

    def run(policy, share, candidate, reject_at):
        rej = [np.where(candidate, draws[0] >= share, True)] + [draws[t] >= share for t in range(1, total)]
        d, tx = simulate(conv, rec, steps, candidate, policy, "DDDSU", pn.Timing, l1_steps=conv_steps,
                         rejected=rej, reject_at=reject_at)
        at = np.unique(np.concatenate([[0.0], d0[fail & np.isfinite(d0)], d[fail & np.isfinite(d)]]))
        diff = pn.delivered_share(d, fail, at) - pn.delivered_share(d0, fail, at)
        return {"min_diff_pp": float(diff.min()), "retx_cut": float((tx0 - 1).sum() / max(1, (tx - 1).sum()))}

    out = {}
    for gpus, cells in (("four", (4, 8, 12, 16, 20, 24)), ("one", (1, 2, 3, 4, 6, 8))):
        f14 = np.mean([r["failed_per_tb"] for r in avail.values() if r["gpus"] == gpus])
        for c in cells:
            row = {}
            for rule, runs_per_tb in loads.items():
                share = se.matched(avail, gpus, c, runs_per_tb / f14)
                pol, candidate = ("select_cb5", cand) if rule == "latency_neutral" else ("wait", np.ones(n, bool))
                row[rule] = {"share": share, **{k: run(pol, share, candidate, t) for k, t in reject_times.items()}}
            out[f"{gpus}_{c}"] = row
            print(f"{gpus:4s} {c:2d} cells | " + " | ".join(
                f"{rule} share {v['share']:.2f}: latest {v['at_latest_start']['min_diff_pp']:+.2f} pp, "
                f"arrival {v['at_arrival']['min_diff_pp']:+.2f} pp, cut {v['at_arrival']['retx_cut']:.2f}x"
                for rule, v in row.items()))
    path = sys.argv[1] if len(sys.argv) > 1 else f"results/delay_under_load{pn.SUFFIX}.json"
    json.dump(out, open(path, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()

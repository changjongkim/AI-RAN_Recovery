#!/usr/bin/env python3
"""N7: when does each failed block end, under three ways to combine recovery with HARQ (trace-driven).

Input: the per-block outcomes of decode_harq.py (npz, any number of chunks).  For every block the trace says whether
the conventional receiver decoded it, whether recovery would decode it, and after how many HARQ transmissions the
combined buffer decodes.  This script places those events on a TDD timeline and reports when each block is
delivered (ms after the samples of its first transmission arrive).

Timeline (all in ms after the samples of the first transmission arrive):
  conventional result   L1 (median of the runtime, 1.4)
  recovery result       L1 + recovery run (6.3); recovery has a deadline D (11.5)
  retransmission k      the grant for transmission k may be sent once the decision is known; the UE transmits in the
                        first uplink slot at least LEAD after the grant (MAC lead time + K2 + PUSCH preparation);
                        its samples are decoded L1 after that slot
Policies:
  harq_only     no recovery; the grant follows the conventional result (today's base station)
  wait          recovery first; the grant follows the recovery result (or D if it does not succeed)
  parallel      the grant follows the conventional result as in harq_only, recovery runs at the same time; if
                recovery succeeds first the block ends then, and the retransmission slot is spent anyway
  select_rho    wait only for blocks whose two users' channels are similar (similarity >= --rho, known before decoding
                from the channel estimate: recovery succeeds 2x more often there, N1); others as harq_only
  select_oracle wait only for the blocks recovery will decode (a perfect predictor; a bound)
  select_cbK    wait only when the conventional receiver failed at most K code blocks of the TB (known when L1
                finishes; needs cbfail_t0 in the trace, decode_harq.py from 2026-10-15 06:30)
TDD patterns: DDDSU (one uplink slot every 2.5 ms) and DDDSUUDDDD (two uplink slots every 5 ms), 0.5 ms slots.
usage: harq_latency.py --inputs C0.npz [...] --output OUT.json [--lead 2.0 --l1 1.4 --run 6.3 --deadline 11.5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PATTERNS = {"DDDSU": "DDDSU", "DDDSUUDDDD": "DDDSUUDDDD"}
SLOT = 0.5


def uplink_times(pattern: str, horizon: float) -> np.ndarray:
    """Start times (ms) of uplink slots, the first transmission's slot at time 0."""
    ul = [i for i, c in enumerate(pattern) if c == "U"]
    period = len(pattern) * SLOT
    first = ul[0] * SLOT
    times = []
    k = 0
    while True:
        for i in ul:
            t = k * period + i * SLOT - first
            if t > horizon:
                return np.asarray(times)
            if t >= 0:
                times.append(t)
        k += 1


def simulate(conv_ok, rec_ok, steps_ok, candidate, policy, pattern, a):
    """Delivery time (ms after the first samples arrive; inf if never) and transmissions used, per block.
    Recovery is always attempted here (no NeuralRx contention; that is the subject of the Antiphase paper), so its
    outcome is known at L1 + run.  A retransmission in uplink slot s is decoded at s + SLOT + L1."""
    total = len(steps_ok)
    times = uplink_times(pattern, 300.0)
    rec_time = a.l1 + a.run
    done = np.full(len(conv_ok), np.inf)
    tx = np.ones(len(conv_ok), dtype=int)
    for i in range(len(conv_ok)):
        if conv_ok[i]:
            done[i] = a.l1
            continue
        tries = policy in ("wait", "parallel") or (policy.startswith("select") and candidate[i])
        recovered = tries and rec_ok[i] and rec_time <= a.deadline
        if recovered:
            done[i] = rec_time
            if policy == "wait" or policy.startswith("select"):
                continue
        # time at which the first retransmission grant is decided
        decide = rec_time if (tries and policy != "parallel") else a.l1
        for k in range(1, total):
            if recovered and rec_time <= decide:
                break                               # parallel: recovery ended the block before this grant
            slot = times[np.searchsorted(times, decide + a.lead)]
            result = slot + SLOT + a.l1
            tx[i] = k + 1
            if steps_ok[k][i]:
                done[i] = min(done[i], result)
                break
            decide = result
    return done, tx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lead", type=float, default=4.0, help="ms from the grant decision to the earliest UL slot")
    parser.add_argument("--l1", type=float, default=1.4)
    parser.add_argument("--run", type=float, default=6.3)
    parser.add_argument("--deadline", type=float, default=11.5)
    parser.add_argument("--rho", type=float, default=0.4, help="similarity threshold of select_rho")
    a = parser.parse_args()
    good, total, rho, cbf = {}, None, [], []
    for p in a.inputs:
        d = np.load(p, allow_pickle=False)
        total = int(d["total"])
        users = d["conv_t0"].shape[1]
        rho.append(np.repeat(d["corr"][:, 0], users))       # same order as reshape(-1) of [slot, user]
        cbf.append(d["cbfail_t0"].reshape(-1) if "cbfail_t0" in d.files else np.full(d["conv_t0"].size, -1))
        for k in d.files:
            if k.startswith(("conv_", "sep_", "nrx_")):
                good.setdefault(k, []).append(d[k].reshape(-1))
    good = {k: np.concatenate(v) for k, v in good.items()}
    conv_ok = good["conv_t0"]
    rec_ok = good["nrx_c0"]
    steps = [None] + [good[f"sep_c{t}"] | good[f"nrx_c{t}"] for t in range(1, total)]
    steps_conv = [None] + [good[f"sep_c{t}"] for t in range(1, total)]
    rho = np.concatenate(rho)
    candidates = {"select_rho": rho >= a.rho, "select_oracle": rec_ok.copy()}
    cbf = np.concatenate(cbf)
    have_cb = bool((cbf[~conv_ok] >= 0).all()) and (~conv_ok).any()
    policies = [("harq_only", steps_conv), ("wait", steps), ("parallel", steps), ("select_rho", steps),
                ("select_oracle", steps)]
    if have_cb:
        for k in (1, 2, 3, 5):
            candidates[f"select_cb{k}"] = (cbf >= 0) & (cbf <= k)
            policies.append((f"select_cb{k}", steps))
    out = {"blocks": int(len(conv_ok)), "params": vars(a) | {"inputs": [str(p) for p in a.inputs]}, "policies": {}}
    if have_cb:   # how well the failed-code-block count predicts recovery
        f = ~conv_ok
        out["recovery_by_failed_cbs"] = {str(k): {"failures": int((f & (cbf == k)).sum()),
                                                  "recovered_pct": 100.0 * float((f & (cbf == k) & rec_ok).sum()) / max(1, int((f & (cbf == k)).sum()))}
                                         for k in range(0, int(cbf[f].max()) + 1) if (f & (cbf == k)).any()}
    for pattern in PATTERNS:
        for policy, st in policies:
            done, tx = simulate(conv_ok, rec_ok, st, candidates.get(policy), policy, PATTERNS[pattern], a)
            fail = ~conv_ok
            fin = np.isfinite(done)
            q = lambda m, p: float(np.percentile(done[m & fin], p)) if (m & fin).any() else None
            out["policies"][f"{pattern}/{policy}"] = {
                "failed_blocks": int(fail.sum()),
                "failed_p50_ms": q(fail, 50), "failed_p90_ms": q(fail, 90), "failed_p99_ms": q(fail, 99),
                "unrecovered_p50_ms": q(fail & ~rec_ok, 50), "unrecovered_p99_ms": q(fail & ~rec_ok, 99),
                "recovered_p50_ms": q(fail & rec_ok, 50),
                "all_p99_ms": q(np.ones_like(fail), 99), "all_p999_ms": q(np.ones_like(fail), 99.9),
                "never_pct": 100.0 * (~fin).sum() / len(done),
                "uplink_slots_per_100": 100.0 * (tx - 1).sum() / len(done),
            }
    a.output.write_text(json.dumps(out, indent=2, default=str))
    print("| pattern/policy | failed p50 | failed p90 | failed p99 | not recovered p50/p99 | recovered p50 | all p99.9 | UL slots /100 | never % |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, r in out["policies"].items():
        f = lambda v: "-" if v is None else f"{v:.1f}"
        print(f"| {k} | {f(r['failed_p50_ms'])} | {f(r['failed_p90_ms'])} | {f(r['failed_p99_ms'])} | "
              f"{f(r['unrecovered_p50_ms'])}/{f(r['unrecovered_p99_ms'])} | {f(r['recovered_p50_ms'])} | "
              f"{f(r['all_p999_ms'])} | {r['uplink_slots_per_100']:.1f} | {r['never_pct']:.2f} |")


if __name__ == "__main__":
    main()

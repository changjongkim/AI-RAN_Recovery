#!/usr/bin/env python3
"""Replay of the recovery queue of the real-time runs with two admission rules (the open question of Section 4.3).

The slot runtime (controller5.py) queues each slot with a failed TB at its L1 result, serves the queue in the order of
the latest start time (slot + D - r) on any free GPU, and rejects a slot when a GPU frees after that time.  A rejected
slot then receives its grant at the latest start time (4.2 ms after the slot), after the last grant time D0 (3.5 ms) of
the first retransmission slot, so the retransmission moves one uplink period later.  The alternative rejects at the L1
result: the scheduler predicts the start of the slot from the GPUs and the slots ahead of it in the queue, each run
taking PRED ms, and rejects the slot at once when the predicted start is after the latest start; the grant then keeps
the first retransmission slot.  With PRED = r (the bound) the prediction never starts a slot later than predicted; with
a shorter PRED some admitted slots are still rejected at the latest start.

Per run of the four-GPU and one-GPU servers (allb68[g4]<N16>): the arrivals (L1 result times) and failed slots from the
conventional records, the run time of every recovery the runtime started (done - start), and for the slots the runtime
rejected a run time drawn from the same run.  The replay of the runtime's rule must give the measured on-time recoveries
(check); then each rule gives the on-time recoveries per failed slot and the rejections before and after D0.
usage: RESULT_SUFFIX=_f35 N16_SUFFIX=d110_f35 admission_replay.py [OUT.json]       (from the repository root)"""
import glob
import heapq
import json
import re
import sys

import numpy as np

sys.path[:0] = ["src", "src/sim", "src/analysis"]
import paper_numbers as pn  # noqa: E402
import server_effect as se  # noqa: E402

def run_items(f):
    """Failed slots of one run: arrival, latest start, deadline (ns from the epoch), measured run time or None."""
    d = json.load(open(f))
    cfg, work = d["config"], f[:-5] + "_work"
    skip = int(cfg["skip_periods"])
    deadline = float(cfg["rescue_deadline_ms"]) * 1e6
    bound = float(cfg["nrx_bound_ms"]) * 1e6
    lanes = {}
    for lf in glob.glob(f"{work}/lane*.json"):
        lane = json.load(open(lf))
        col = {k: i for i, k in enumerate(lane["columns"])}
        for r in lane["records"]:
            lanes[(int(r[col["cell"]]), int(r[col["period"]]))] = (int(r[col["start_ns"]]), int(r[col["done_ns"]]))
    items = []
    for cf in glob.glob(f"{work}/conv*.json"):
        cell = int(re.search(r"conv(\d+)\.json$", cf).group(1))
        c = json.load(open(cf))
        col = {k: i for i, k in enumerate(c["columns"])}
        for r in c["records"]:
            k = int(r[col["period"]])
            if k < skip or not (~int(r[col["ue_crc_mask"]]) & 0b11):
                continue
            rel, arr = int(r[col["release_ns"]]), int(r[col["done_ns"]])
            run = lanes.get((cell, k))
            items.append((arr, rel + deadline - bound, rel + deadline, rel, None if run is None else run[1] - run[0],
                          None if run is None else run[1]))
    items.sort(key=lambda x: (x[0], x[1]))
    return items, int(cfg["num_gpus"]) * int(cfg.get("lanes_per_gpu", 1)), bound


def replay(items, lanes, rule, pred_ns, rng, detail=False):
    """On-time runs, rejections at the L1 result and rejections at the latest start, for rule 'lazy' or 'arrival'.
    detail: also per slot (decision, predicted end, end) in ms after the slot, decision 'arrival', 'latest' or 'run'."""
    measured = np.array([x[4] for x in items if x[4] is not None], float)
    run = [x[4] if x[4] is not None else float(rng.choice(measured)) for x in items]
    free = [0.0] * lanes          # time each lane frees
    started = [0.0] * lanes       # start of the run on each lane (for the prediction)
    pending, i, n = [], 0, len(items)
    on_time = early = late = 0
    pred_end, slots = {}, []
    while i < n or pending:
        tf = min(free)
        ta = items[i][0] if i < n else float("inf")
        if pending and tf <= ta:
            latest, j = heapq.heappop(pending)
            lane = free.index(tf)
            t = max(tf, items[j][0])
            if t > latest:
                late += 1
                slots.append(("latest", pred_end.get(j, np.nan), np.nan))
                continue
            free[lane], started[lane] = t + run[j], t
            on_time += t + run[j] <= items[j][2]
            slots.append(("run", pred_end.get(j, np.nan), (t + run[j] - items[j][3]) / 1e6))
            continue
        arr, latest = items[i][0], items[i][1]
        if rule == "arrival":
            # predicted lane free times: a running slot ends PRED after its start (or now, if that is past)
            pl = sorted(max(arr, (started[l] + pred_ns) if free[l] > arr else arr) for l in range(lanes))
            for lat, _ in sorted(pending):
                if lat > latest:
                    break
                s = pl[0]
                if s > lat:      # the scheduler also predicts that this slot is rejected: it frees no lane
                    continue
                pl[0] = s + pred_ns
                pl.sort()
            if pl[0] > latest:
                early += 1
                slots.append(("arrival", np.nan, np.nan))
                i += 1
                continue
            pred_end[i] = (pl[0] + pred_ns - items[i][3]) / 1e6
        heapq.heappush(pending, (latest, i))
        i += 1
    r = {"on_time": on_time, "rejected_at_arrival": early, "rejected_at_latest_start": late}
    return (r, slots) if detail else r


def wait_split(tag, cells, cut_ms):
    """Rejection at the L1 result with the median run as the prediction, over the runs of a tag and number of cells:
    the share of the on-time runs whose predicted end is at most cut_ms (the slots that wait for the recovery when the
    grant waits only for a recovery predicted to end by cut_ms), the end times of these runs (ms after the slot), and
    the rejections at the latest start per failed slot."""
    ends, late, ontime, failed = [], 0, 0, 0
    for f in sorted(glob.glob(f"{se.RAW}/rec16{tag}s*_c{cells}_*_none_j*.json")):
        items, lanes, bound = run_items(f)
        p50 = float(np.median([x[4] for x in items if x[4] is not None]))
        deadline_ms = (items[0][2] - items[0][3]) / 1e6
        _, slots = replay(items, lanes, "arrival", p50, np.random.default_rng(3), detail=True)
        failed += len(items)
        late += sum(1 for d, _, _ in slots if d == "latest")
        on = [(pe, e) for d, pe, e in slots if d == "run" and e <= deadline_ms]
        ontime += len(on)
        ends += [e for pe, e in on if pe <= cut_ms]
    return {"wait_share_of_on_time": len(ends) / max(1, ontime), "wait_end_ms": np.asarray(ends),
            "late_per_failed": late / max(1, failed), "on_time_per_failed": ontime / max(1, failed)}


def main():
    out = {}
    for gpus, tag in (("four", f"allb68g4{pn.N16}"), ("one", f"allb68{pn.N16}")):
        files = sorted(glob.glob(f"{se.RAW}/rec16{tag}s*_c*_*_none_j*.json"))
        cells = sorted({int(re.search(r"_c(\d+)_", f).group(1)) for f in files})
        for c in cells:
            acc = {}
            for f in [f for f in files if f"_c{c}_" in f]:
                items, lanes, bound = run_items(f)
                acc.setdefault("failed_slots", 0)
                acc["failed_slots"] += len(items)
                acc.setdefault("measured_on_time", 0)
                acc["measured_on_time"] += sum(1 for x in items if x[5] is not None and x[5] - x[3] <= x[2] - x[3])
                for name, rule, pred in (("lazy", "lazy", bound), ("arrival_bound", "arrival", bound),
                                         ("arrival_p50", "arrival", None)):
                    p = pred if pred is not None else float(np.median([x[4] for x in items if x[4] is not None]))
                    r = replay(items, lanes, rule, p, np.random.default_rng(3))
                    a = acc.setdefault(name, {k: 0 for k in r})
                    for k in r:
                        a[k] += r[k]
            n = acc["failed_slots"]
            row = {"gpus": gpus, "cells": c, "failed_slots": n, "measured_on_time_per_failed": acc["measured_on_time"] / n}
            for name in ("lazy", "arrival_bound", "arrival_p50"):
                row[name] = {k: v / n for k, v in acc[name].items()}
            out[f"{gpus}_{c}"] = row
            print(f"{gpus:4s} {c:2d} cells: on time per failed slot measured {row['measured_on_time_per_failed']:.3f}, "
                  + ", ".join(f"{name} {row[name]['on_time']:.3f} (rejected at L1 {row[name]['rejected_at_arrival']:.3f}, "
                              f"at latest start {row[name]['rejected_at_latest_start']:.3f})"
                              for name in ("lazy", "arrival_bound", "arrival_p50")))
    path = sys.argv[1] if len(sys.argv) > 1 else f"results/admission_replay{pn.SUFFIX}.json"
    json.dump(out, open(path, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""N13: how many neural-receiver runs one A100 sustains, with K copies of the engine on K CUDA streams ("lanes").

Each lane is its own TensorRT execution context with a captured CUDA graph (the runtime path of decode_harq.py).
For K = 1..8 lanes, every lane launches its graph back to back for --seconds; we record the completed runs per second
(throughput) and the time from launch to completion of each run (latency, p50/p99).  From throughput and the measured
share of uplink slots that need recovery (N9), cells per GPU = throughput / (recoveries per second per cell).
usage: GPU=0 rx python3 /rec/nrx_lanes.py --engine LARGE --engine2 SMALL --output OUT.json
"""
import argparse, json, time
import cupy as cp
import numpy as np
from mu_radio import NvEngine

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True)
ap.add_argument("--engine2", default="")
ap.add_argument("--lanes", default="1,2,3,4,6,8")
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--output", required=True)
a = ap.parse_args()


def run(path, k):
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(k)]
    engines = [NvEngine(path, stream=s) for s in streams]
    for e in engines:
        for x in e.inputs.values():
            if x.dtype.kind == "f":
                x[...] = cp.random.standard_normal(x.shape).astype(x.dtype) * 0.1
        e.inputs["active_dmrs_ports"].fill(1)
        e.capture_graph()
    for e in engines:                     # warm-up
        for _ in range(5):
            e.launch(use_graph=True)
        e.stream.synchronize()
    lat, done = [], 0
    ev = [[None, None] for _ in range(k)]
    t_end = time.perf_counter() + a.seconds
    pending = []
    for i, e in enumerate(engines):
        s, f = cp.cuda.Event(), cp.cuda.Event()
        s.record(e.stream); e.launch(use_graph=True); f.record(e.stream); pending.append((i, s, f))
    t0 = time.perf_counter()
    while time.perf_counter() < t_end:
        nxt = []
        for i, s, f in pending:
            f.synchronize()
            lat.append(cp.cuda.get_elapsed_time(s, f)); done += 1
            e = engines[i]
            s2, f2 = cp.cuda.Event(), cp.cuda.Event()
            s2.record(e.stream); e.launch(use_graph=True); f2.record(e.stream); nxt.append((i, s2, f2))
        pending = nxt
    for i, s, f in pending:
        f.synchronize()
    el = time.perf_counter() - t0
    lat = np.asarray(lat)
    return {"lanes": k, "runs_per_s": done / el, "latency_ms_p50": float(np.percentile(lat, 50)),
            "latency_ms_p99": float(np.percentile(lat, 99)), "latency_ms_mean": float(lat.mean())}


out = {}
for name, path in (("large", a.engine), ("small", a.engine2)):
    if not path:
        continue
    out[name] = []
    for k in [int(x) for x in a.lanes.split(",")]:
        r = run(path, k); out[name].append(r)
        print(name, json.dumps(r), flush=True)
json.dump(out, open(a.output, "w"), indent=1)

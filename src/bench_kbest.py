#!/usr/bin/env python3
"""GPU time of the classical recovery receiver on one slot (Section 5.3 and Discussion of the Reverb paper).

The receivers of Figure 12 (d, e) run in the NVlabs/Sionna 0.18 framework.  This script times their stages alone on
the inputs the PUSCH receiver passes to each stage, one slot (batch 1) at 273 PRBs on one GPU:
  kbest_det   K-best detection, K = 64, of the max_num_tx streams of the configuration (baseline_lmmse_kbest)
  lmmse_est   the LMMSE channel estimate that K-best detection uses (baseline_lmmse_kbest)
  lmmse_det   LMMSE detection (baseline_lslin_lmmse, the conventional receiver of the traces)
  lslin_est   the LS estimate with linear interpolation (baseline_lslin_lmmse)
  nrx         the large NRx of the same configuration up to the LLRs (its channel estimate and the network with the
              iterations of the configuration, without TB decoding), as a reference in the same framework
Each stage is compiled once with tf.function, with XLA when XLA compiles the stage, else as a graph, and is timed
after warm-up over --reps calls, each ending with a reduction copied to the host (the GPU work of the call is done).
The inputs come from one call of the receiver at --ebno (the time does not depend on the values).  The node runs
nothing else.  usage (tfgen of env.sh): python3 bench_kbest.py --nrx-repo DIR --base-config nrx_large_mu4.cfg
       --num-rx 4 --users 2 --output results/kbest_gpu_time/r4_u2.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--nrx-repo", type=Path, required=True)
parser.add_argument("--base-config", required=True)
parser.add_argument("--num-rx", type=int, required=True)
parser.add_argument("--users", type=int, required=True)
parser.add_argument("--channel", default="UMa")
parser.add_argument("--carrier-ghz", type=float, default=3.5)
parser.add_argument("--prbs", type=int, default=273)
parser.add_argument("--mcs", type=int, default=14)
parser.add_argument("--ebno", type=float, default=4.0)
parser.add_argument("--reps", type=int, default=50)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--gpu", default="0")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, str(args.nrx_repo))
os.chdir(args.nrx_repo / "scripts")

import numpy as np                      # noqa: E402
import tensorflow as tf                 # noqa: E402

tf.get_logger().setLevel("ERROR")
for device in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)
tf.random.set_seed(1)
from utils import E2E_Model, Parameters   # noqa: E402


def make_config(base: str) -> str:
    """The configuration of classical_rx.py (channel, speed 3 km/h, MCS, PRBs, antennas, carrier)."""
    text = (Path("../config") / base).read_text()
    label = re.search(r"label\s*=\s*'([^']+)'", text).group(1)
    subs = [(r"(?m)^channel_type_eval\s*=.*", f'channel_type_eval = "{args.channel}"'),
            (r"max_ut_velocity_eval\s*=.*", "max_ut_velocity_eval = 0.83"),
            (r"min_ut_velocity_eval\s*=.*", "min_ut_velocity_eval = 0.83"),
            (r"(?m)^mcs_index\s*=.*", f"mcs_index = [{args.mcs}]"),
            (r"n_size_bwp_eval\s*=.*", f"n_size_bwp_eval = {args.prbs}"),
            (r"(?m)^num_rx_antennas\s*=.*", f"num_rx_antennas = {args.num_rx}"),
            (r"(?m)^carrier_frequency\s*=.*", f"carrier_frequency = {args.carrier_ghz * 1e9:.1f}")]
    for pattern, value in subs:
        text, n = re.subn(pattern, value, text)
        assert n >= 1, pattern
    name = f"bench_{label}_r{args.num_rx}_u{args.users}_p{os.getpid()}.cfg"
    (Path("../config") / name).write_text(text)
    return name


cfg = make_config(args.base_config)
captured: dict[str, list] = {}
stages: dict[str, object] = {}
for system, est_key, det_key in (("baseline_lmmse_kbest", "lmmse_est", "kbest_det"),
                                 ("baseline_lslin_lmmse", "lslin_est", "lmmse_det")):
    params = Parameters(cfg, training=False, system=system, num_tx_eval=args.users)
    params.batch_size_eval_small = 1
    model = E2E_Model(params, training=False, return_tb_status=True)
    pusch = model._receiver._receiver
    det, est = pusch._mimo_detector, pusch._channel_estimator
    stages[det_key], stages[est_key] = det, est

    def det_wrap(inputs, key=det_key, layer=det):
        captured[key] = [tf.identity(x) for x in inputs]
        return layer(inputs)

    def est_wrap(inputs, key=est_key, layer=est):
        captured[key] = [tf.identity(x) for x in inputs]
        return layer(inputs)

    pusch._mimo_detector, pusch._channel_estimator = det_wrap, est_wrap
    model(1, args.ebno)
    pusch._mimo_detector, pusch._channel_estimator = det, est
params = Parameters(cfg, training=False, system="nrx", num_tx_eval=args.users)
params.batch_size_eval_small = 1
model = E2E_Model(params, training=False, return_tb_status=True)
nrx = model._receiver
nrx._neural_rx.num_it = params.num_nrx_iter_eval          # as classical_rx.py (weights do not change the time)


class Capture:
    """Records the inputs of the receiver call; every other attribute is the receiver's."""

    def __init__(self, inner):
        self._inner = inner

    def __call__(self, inputs, *a, **kw):
        captured["nrx"] = [tf.identity(x) for x in inputs]
        return self._inner(inputs, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


model._receiver = Capture(nrx)
model(1, args.ebno)
model._receiver = nrx


def nrx_llr(inputs):
    y, active_tx = inputs
    h_hat = nrx.estimate_channel(y, tf.shape(active_tx)[1])
    return nrx._neural_rx((y, h_hat, active_tx), [0])[0]


stages["nrx"] = nrx_llr
result_nrx_iterations = int(params.num_nrx_iter_eval)
os.remove(Path("../config") / cfg)


def done(out) -> float:
    t = tf.nest.flatten(out)[0]
    return float(tf.reduce_sum(tf.abs(t)).numpy())


result = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
          "device": tf.config.experimental.get_device_details(tf.config.list_physical_devices("GPU")[0]).get("device_name"),
          "tensorflow": tf.__version__, "nrx_iterations": result_nrx_iterations, "stages": {}}
for key in ("lslin_est", "lmmse_det", "lmmse_est", "kbest_det", "nrx"):
    layer, inputs = stages[key], captured[key]
    entry = {"input_shapes": [list(x.shape) for x in inputs]}
    for mode in ("xla", "graph"):
        fn = tf.function(lambda *xs, layer=layer: layer(list(xs)), jit_compile=(mode == "xla"))
        try:
            t0 = time.perf_counter()
            done(fn(*inputs))
            entry["compile_s"] = time.perf_counter() - t0
            break
        except Exception as e:                       # noqa: BLE001  (XLA cannot compile every stage)
            entry[f"{mode}_error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    else:
        raise RuntimeError(f"{key}: neither XLA nor graph mode ran")
    entry["mode"] = mode
    for _ in range(args.warmup):
        done(fn(*inputs))
    ms = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        done(fn(*inputs))
        ms.append(1000.0 * (time.perf_counter() - t0))
    ms = np.asarray(ms)
    entry.update({"median_ms": float(np.median(ms)), "p99_ms": float(np.percentile(ms, 99)), "min_ms": float(ms.min()),
                  "reps": args.reps})
    result["stages"][key] = entry
    print(f"r{args.num_rx} u{args.users} {key:10s} {mode:5s} median {entry['median_ms']:8.2f} ms  p99 {entry['p99_ms']:8.2f} ms"
          f"  inputs {entry['input_shapes']}", flush=True)

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=1))

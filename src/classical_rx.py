#!/usr/bin/env python3
"""N5 and P1 (antennas): several receivers on the same received slots, in the NVlabs/Sionna framework.

Every system is the NVlabs end-to-end model (E2E_Model, return_tb_status) of one receiver; the first system draws the
bits, topology and channel of a batch and records them, the others replay the same bits and received signal, so all
receivers see identical slots.  Receivers (NVlabs names):
  baseline_lslin_lmmse     LS channel estimate, linear interpolation, LMMSE detection   (the conventional receiver)
  baseline_lmmse_lmmse     LMMSE channel estimate, LMMSE detection                        (needs covariance matrices)
  baseline_lmmse_kbest     LMMSE channel estimate, K-best detection (K = 64)              (needs covariance matrices)
  baseline_perf_csi_kbest  perfect channel knowledge, K-best detection                     (a bound)
  nrx / nrx_rt             the large / real-time neural receiver (TensorFlow weights of the NVlabs repository)
  lslin_idd<k>             the conventional channel estimate with iterative detection and decoding, k rounds of
                           decoder feedback to an MMSE-PIC detector (src/idd.py); lslin_idd0 is the LMMSE receiver
  <system>@<n>             that system with n BP iterations instead of --bp-iter (e.g. baseline_lslin_lmmse@50)
Decoding is Sionna's TBDecoder for all (the same number of BP iterations).  Outputs per slot and user whether the block
decoded with the right bits, the GPU time per batch of each receiver, and the similarity of the two users' channels.
usage: classical_rx.py --nrx-repo DIR --systems baseline_lslin_lmmse,baseline_lmmse_kbest,nrx --ebno 6,8,10,12
       --slots 200 --output OUT [--channel UMi --speed 0.83 --num-rx 4]
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
parser.add_argument("--base-config", default="nrx_large.cfg")
parser.add_argument("--rt-config", default="nrx_rt.cfg")
parser.add_argument("--nrx-weights", default="", help="weights file for the nrx system instead of ../weights/<label>_weights")
parser.add_argument("--prbs", type=int, default=273, help="bandwidth in PRBs (273 as in N1 and the runtime receivers)")
parser.add_argument("--systems", default="baseline_lslin_lmmse,baseline_lmmse_kbest,baseline_perf_csi_kbest,nrx,nrx_rt")
parser.add_argument("--channel", default="UMi")
parser.add_argument("--speed", type=float, default=0.83)
parser.add_argument("--num-rx", type=int, default=0)
parser.add_argument("--mcs", type=int, default=14)
parser.add_argument("--users", type=int, default=2, help="active users on the same resources (configs must allow it)")
parser.add_argument("--ebno", default="6,8,10,12")
parser.add_argument("--slots", type=int, default=200)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--bp-iter", type=int, default=20)
parser.add_argument("--seed", type=int, default=3)
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
tf.random.set_seed(args.seed)
from utils import E2E_Model, Parameters, load_weights   # noqa: E402


def make_config(base: str, bp_iter: int) -> tuple[str, str]:
    text = (Path("../config") / base).read_text()
    label = re.search(r"label\s*=\s*'([^']+)'", text).group(1)
    subs = [(r"(?m)^channel_type_eval\s*=.*", f'channel_type_eval = "{args.channel}"'),
            (r"max_ut_velocity_eval\s*=.*", f"max_ut_velocity_eval = {args.speed}"),
            (r"min_ut_velocity_eval\s*=.*", f"min_ut_velocity_eval = {args.speed}"),
            (r"(?m)^mcs_index\s*=.*", f"mcs_index = [{args.mcs}]"),
            (r"n_size_bwp_eval\s*=.*", f"n_size_bwp_eval = {args.prbs}"),
            (r"(?m)^num_bp_iter\s*=.*", f"num_bp_iter = {bp_iter}")]
    if args.num_rx:
        subs.append((r"(?m)^num_rx_antennas\s*=.*", f"num_rx_antennas = {args.num_rx}"))
    for pattern, value in subs:
        text, n = re.subn(pattern, value, text)
        assert n >= 1 or "num_bp_iter" in pattern, pattern
    name = f"cls_{label}_{args.channel}_v{args.speed:g}_m{args.mcs}_r{args.num_rx or 4}_bp{bp_iter}_p{os.getpid()}.cfg"
    tmp = Path("../config") / (name + ".tmp")
    tmp.write_text(text)
    tmp.rename(Path("../config") / name)
    return name, label


systems = args.systems.split(",")
assert args.slots % args.batch == 0, "slots must be a multiple of the batch (fixed-batch LMMSE estimator)"
models, record = {}, {}
for s in systems:
    kind, _, iters = s.partition("@")
    bp = int(iters) if iters else args.bp_iter
    idd = kind.startswith("lslin_idd")
    system = "baseline_lslin_lmmse" if idd else ("nrx" if kind.startswith("nrx") else kind)
    base = args.rt_config if kind == "nrx_rt" else args.base_config
    cfg, label = make_config(base, bp)
    params = Parameters(cfg, training=False, system=system, num_tx_eval=args.users)
    params.batch_size_eval_small = args.batch      # the NVlabs low-complexity LMMSE estimator fixes the batch to this
    model = E2E_Model(params, training=False, return_tb_status=True)
    model(args.batch, 10.0)
    if kind.startswith("nrx"):
        load_weights(model, args.nrx_weights if (args.nrx_weights and kind == "nrx") else f"../weights/{label}_weights")
        model._receiver._neural_rx.num_it = params.num_nrx_iter_eval
    if idd:
        from idd import IddPusch
        model._receiver._receiver = IddPusch(model._receiver._receiver, int(kind[len("lslin_idd"):]), bp,
                                             params.transmitters[0]._num_bits_per_symbol, params.cn_type)
        model(args.batch, 10.0)
    models[s] = model

first = systems[0]
orig_channel, orig_source = models[first]._channel, models[first]._source


def recording_channel(inputs):
    y, h = orig_channel(inputs)
    record["y"], record["h"] = y, h
    return y, h


def recording_source(shape):
    record["b"] = orig_source(shape)
    return record["b"]


models[first]._channel, models[first]._source = recording_channel, recording_source
for s in systems[1:]:
    models[s]._channel = lambda inputs: (record["y"], record["h"])
    models[s]._source = lambda shape: record["b"]


def similarity(h):
    """Largest similarity over the pairs of active users: mean over subcarriers and symbols of
    |h_i^H h_j| / (|h_i||h_j|) (two users: the similarity of the pair)."""
    v = h.numpy()[:, 0, :, :, 0]
    n = min(args.users, v.shape[2])
    if n < 2:
        return np.zeros(v.shape[0], np.float32)
    norm = np.linalg.norm(v, axis=1)
    best = np.zeros(v.shape[0])
    for i in range(n):
        for j in range(i + 1, n):
            num = np.abs(np.sum(np.conj(v[:, :, i]) * v[:, :, j], axis=1))
            best = np.maximum(best, (num / (norm[:, i] * norm[:, j] + 1e-12)).mean(axis=(1, 2)))
    return best


good = {s: [] for s in systems}
seconds = {s: 0.0 for s in systems}
points, corr = [], []
for point in [float(e) for e in args.ebno.split(",")]:
    done = 0
    while done < args.slots:
        n = min(args.batch, args.slots - done)
        for s in systems:
            t0 = time.perf_counter()
            b, b_hat, crc = models[s](n, point)
            ok = tf.logical_and(tf.cast(crc, tf.bool), tf.reduce_all(tf.equal(b, b_hat), axis=-1)).numpy()
            seconds[s] += time.perf_counter() - t0
            good[s].append(ok.reshape(n, -1))
        points += [point] * n
        corr.append(similarity(record["h"]))
        done += n
    print(f"Eb/No {point}: " + ", ".join(f"{s} {np.concatenate(good[s])[-args.slots:].mean():.3f}" for s in systems),
          flush=True)

good = {s: np.concatenate(v) for s, v in good.items()}
args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez(args.output.with_suffix(".npz"), point=np.asarray(points, np.float32), corr=np.concatenate(corr),
         **{f"ok_{s}": v for s, v in good.items()})
conv = good[systems[0]]
report = {"systems": systems, "prbs": args.prbs, "nrx_weights": args.nrx_weights, "users": args.users, "base_config": args.base_config, "channel": args.channel, "speed": args.speed, "num_rx": args.num_rx or 4, "mcs": args.mcs,
          "slots": int(conv.shape[0]), "seconds_per_slot": {s: seconds[s] / conv.shape[0] for s in systems},
          "decoded_pct": {s: 100.0 * good[s].mean() for s in systems},
          "recovers_pct_of_conventional_failures": {s: 100.0 * (good[s] & ~conv).sum() / max(1, (~conv).sum())
                                                    for s in systems[1:]}}
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

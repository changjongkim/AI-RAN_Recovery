#!/usr/bin/env python3
"""N4b: start an 8-RX large neural receiver from the published 4-RX weights.

Every weight whose shape does not depend on the number of receive antennas is copied.  Two places depend on it:
  input of the state initialization (concat of y: 2N, positional encoding: K, LS estimate: 2N, with K = 2 found from
  the shapes although the NVlabs comment says 4N + 3; real parts of all
  antennas first, then imaginary parts)  -> antennas 5-8 get the weights of antennas 1-4; in a regular or pointwise
  kernel both copies are halved, so an input whose antennas 5-8 repeat 1-4 gives the 4-RX model's output (depthwise
  kernels act per channel and are copied as they are)
  channel-estimate readout (2N outputs, real parts first)  -> antennas 5-8 predict what antennas 1-4 predict
The result is the starting point of fine-tuning on 8-RX slots (exp/n4b_train.sh); it is not a trained 8-RX receiver.
usage: nrx_widen.py --nrx-repo DIR --src-config nrx_large.cfg --dst-config nrx_large_8rx.cfg --src-weights W4 --out W8
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--nrx-repo", type=Path, required=True)
parser.add_argument("--src-config", default="nrx_large.cfg")
parser.add_argument("--dst-config", required=True)
parser.add_argument("--src-weights", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--gpu", default="-1")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, str(args.nrx_repo))
os.chdir(args.nrx_repo / "scripts")

import numpy as np                      # noqa: E402
import tensorflow as tf                 # noqa: E402
from utils import E2E_Model, Parameters  # noqa: E402


def build(cfg):
    p = Parameters(cfg, training=True, system="nrx")
    m = E2E_Model(p, training=True)
    m(1, 1.)
    return p, m


p4, m4 = build(args.src_config)
p8, m8 = build(args.dst_config)
n4, n8 = int(p4.num_rx_antennas), int(p8.num_rx_antennas)
assert n8 == 2 * n4, "only doubling the antennas is implemented"
src = pickle.load(open(args.src_weights, "rb"))
m4.set_weights(src)                     # checks that the source weights fit the source model
dst = m8.get_weights()
assert len(src) == len(dst), (len(src), len(dst))


def widen_input(w, scale, k):
    """axis -2 holds [y_re(N), y_im(N), pe(k), h_re(N), h_im(N)]."""
    y_re, y_im = w[..., 0:n4, :], w[..., n4:2 * n4, :]
    pe = w[..., 2 * n4:2 * n4 + k, :]
    h_re, h_im = w[..., 2 * n4 + k:3 * n4 + k, :], w[..., 3 * n4 + k:4 * n4 + k, :]
    d = lambda x: np.concatenate([x * scale, x * scale], axis=-2)
    return np.concatenate([d(y_re), d(y_im), pe, d(h_re), d(h_im)], axis=-2)


def widen_output(w):
    """last axis holds [re(N), im(N)]."""
    re, im = w[..., :n4], w[..., n4:2 * n4]
    return np.concatenate([re, re, im, im], axis=-1)


out, report = [], {"copied": 0, "input": [], "output": []}
for i, (a, b) in enumerate(zip(src, dst)):
    if a.shape == b.shape:
        out.append(a); report["copied"] += 1
    elif a.ndim >= 2 and 0 <= a.shape[-2] - 4 * n4 == b.shape[-2] - 4 * n8 <= 4:
        k = a.shape[-2] - 4 * n4
        depthwise = a.ndim == 4 and a.shape[-1] == 1 and b.shape[-1] == 1
        out.append(widen_input(a, 1.0 if depthwise else 0.5, k)); report["input"].append((i, a.shape, b.shape, depthwise))
    elif a.shape[-1] == 2 * n4 and b.shape[-1] == 2 * n8:
        out.append(widen_output(a)); report["output"].append((i, a.shape, b.shape))
    else:
        raise SystemExit(f"weight {i}: no rule for {a.shape} -> {b.shape}")
    assert out[-1].shape == b.shape, (i, out[-1].shape, b.shape)
m8.set_weights(out)
pickle.dump(m8.get_weights(), open(args.out, "wb"))
print(report)
print("saved", args.out)

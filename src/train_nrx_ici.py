#!/usr/bin/env python3
"""N15: train the NVlabs large neural receiver with inter-cell interference (interference-aware recovery receiver).

The published receiver was trained on one cell without interference (AWGN and the 3GPP UMi channel only).  Here the
NVlabs training loop and data are kept as they are, and the channel of the training model is wrapped: after the
served users' slot is built, two users of a neighbor cell are added on the same resources.  They have their own 3GPP
channels (same model, own random topology every batch), send random 16QAM symbols on every resource element (their
data and DMRS are unknown to this cell; their DMRS scrambling differs), and arrive at an interference over thermal
noise of INR dB.  INR is drawn per slot: no interference with probability --p-none, otherwise uniform in
[--inr-min, --inr-max] dB, so the receiver keeps working without interference.  The evaluation (gen_harq.py
--interferers --inr-db) uses the same model of the interferers.
Run from third_party/neural_rx/scripts like the NVlabs script:
usage: python3 train_nrx_ici.py -config_name nrx_large_ici.cfg -gpu 0 [--p-none 0.25 --inr-min 0 --inr-max 10]
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-config_name", required=True)
parser.add_argument("-gpu", type=int, default=0)
parser.add_argument("--p-none", type=float, default=0.25)
parser.add_argument("--inr-min", type=float, default=0.0)
parser.add_argument("--inr-max", type=float, default=10.0)
parser.add_argument("--interferers", type=int, default=2)
parser.add_argument("--check", action="store_true", help="one forward pass with and without the wrapper, then exit")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np                      # noqa: E402
import tensorflow as tf                 # noqa: E402
tf.get_logger().setLevel("ERROR")
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
sys.path.append("../")
from os.path import exists              # noqa: E402
from sionna.channel import OFDMChannel, gen_single_sector_topology   # noqa: E402
from utils import E2E_Model, training_loop, Parameters, load_weights  # noqa: E402

sys_parameters = Parameters(args.config_name, system="nrx", training=True)
assert sys_parameters.channel_type in ("UMi", "UMa"), "the interferers use the 3GPP topology of the served cell's model"
assert not sys_parameters.channel_norm


class InterferedChannel:
    """Wraps the OFDM channel of the training model: y + the signal of the neighbor-cell users."""

    def __init__(self, base, p):
        self.base, self.p = base, p
        self.model = p.channel_model
        self.ofdm = OFDMChannel(channel_model=self.model, resource_grid=p.transmitters[0]._resource_grid,
                                add_awgn=False, normalize_channel=False, return_channel=False)
        self.scenario = "umi" if p.channel_type == "UMi" else "uma"
        points = p.transmitters[0]._mapper.constellation.points
        self.points = tf.cast(points, tf.complex64) / tf.cast(tf.sqrt(tf.reduce_mean(tf.abs(points) ** 2)), tf.complex64)

    def __call__(self, inputs):
        x, no = inputs
        y, h = self.base([x, no])                       # served users; the topology was set by the model's call
        b = tf.shape(x)[0]
        topo = gen_single_sector_topology(b, args.interferers, self.scenario, min_ut_velocity=self.p.min_ut_velocity,
                                          max_ut_velocity=self.p.max_ut_velocity, indoor_probability=0.)
        self.model.set_topology(*topo)                  # the next call of the model sets the served users' topology again
        shape = tf.concat([[b, args.interferers], tf.shape(x)[2:]], 0)
        idx = tf.random.uniform(shape, 0, tf.shape(self.points)[0], dtype=tf.int32)
        xi = tf.gather(self.points, idx)
        yi = self.ofdm(xi)
        inr = tf.random.uniform([b], args.inr_min, args.inr_max)
        on = tf.cast(tf.random.uniform([b]) >= args.p_none, tf.float32)
        nf = tf.reshape(tf.cast(no, tf.float32), [-1])       # noise variance: one value or one per slot
        no_b = tf.cond(tf.equal(tf.size(nf), b), lambda: nf, lambda: tf.fill([b], nf[0]))
        amp = tf.sqrt(no_b * tf.pow(10.0, inr / 10.0) / args.interferers) * on
        amp = tf.reshape(tf.complex(amp, 0.0), [-1, 1, 1, 1, 1])
        return y + amp * yi, h


sys_parameters.channel = InterferedChannel(sys_parameters.channel, sys_parameters)
label = sys_parameters.label
filename = "../weights/" + label + "_weights"
sys_training = E2E_Model(sys_parameters, training=True)
sys_training(1, 1.)
if exists(filename):
    print("\nWeights exist already - loading stored weights.", flush=True)
    load_weights(sys_training, filename)
if args.check:
    ch = sys_parameters.channel
    x = tf.complex(tf.random.normal([16, 2, 1, 14, 48]), 0.0) * 0     # silent served users: y is noise + interference
    no = tf.constant(0.1)
    y, _ = ch([x, no])
    print("received power / noise (expect about 1 + 10^(INR/10) when on, 1 when off):",
          np.round(np.mean(np.abs(y.numpy()) ** 2, axis=(1, 2, 3, 4)) / 0.1, 2).tolist(), flush=True)
    base = ch.base
    for name in ("with interference", "without"):
        tf.random.set_seed(1)
        out = sys_training(16, 12.)
        print(name, "loss", [float(tf.reduce_mean(v)) for v in (out if isinstance(out, (list, tuple)) else [out])], flush=True)
        sys_training._channel = base
    sys.exit(0)
training_loop(sys_training, label=label, filename=filename, training_logdir="../logs", training_seed=42,
              training_schedule=sys_parameters.training_schedule, eval_ebno_db_arr=sys_parameters.eval_ebno_db_arr,
              min_num_tx=sys_parameters.min_num_tx, max_num_tx=sys_parameters.max_num_tx, sys_parameters=sys_parameters,
              mcs_arr_training_idx=list(range(len(sys_parameters.mcs_index))),
              mcs_training_snr_db_offset=getattr(sys_parameters, "mcs_training_snr_db_offset", None),
              mcs_training_probs=getattr(sys_parameters, "mcs_training_probs", None), xla=sys_parameters.xla)

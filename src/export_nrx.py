#!/usr/bin/env python3
"""Export an NVlabs neural receiver to ONNX for the runtime (copied from vendor/nv_export_mu.py, 2026-10-18): the metadata
also has the DMRS positions that the runtime engine reads, and the export stops if the weights do not load.

Original description: export an NVlabs neural receiver for 1-4 users on the same resources to ONNX (v38).

The published weights know two users.  The network keeps one state per user, so its size does not depend on the
number of users and the weights of two users load into the receiver of any number of users; if they do not, the
receiver keeps its initial weights.  Either way the run time is that of the trained receiver: the computation does
not depend on the values of the weights.  Users 1-4 are on DMRS ports 0, 2, 1, 3 (as nv_dataset_mu.py).
Run from third_party/neural_rx/scripts.
usage: nv_export_mu.py --base-config nrx_rt.cfg --num-tx 4 --output-dir DIR
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--base-config", required=True)
parser.add_argument("--prbs", type=int, default=273)
parser.add_argument("--num-tx", type=int, required=True)
parser.add_argument("--channel", default="UMi", help="UMi: the DoubleTDL models know two users only")
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
assert 1 <= args.num_tx <= 4

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("../")

import numpy as np                      # noqa: E402
import tensorflow as tf                 # noqa: E402
import tf2onnx                          # noqa: E402
import onnx                             # noqa: E402

from utils import Parameters, NeuralReceiverONNX, load_weights      # noqa: E402
from utils import DataGeneratorAerial                               # noqa: E402

n = args.num_tx
ports = [0, 2, 1, 3][:n]
text = (Path("../config") / args.base_config).read_text()
label = re.search(r"label\s*=\s*'([^']+)'", text).group(1)
text = re.sub(r"n_size_bwp_eval\s*=.*", f"n_size_bwp_eval = {args.prbs}", text)
text = re.sub(r"(?m)^channel_type_eval\s*=.*", f'channel_type_eval = "{args.channel}"', text)
text = re.sub(r"(?m)^dmrs_port_sets\s*=.*", f"dmrs_port_sets = {[[p] for p in ports]}", text)
text = re.sub(r"(?m)^n_rntis\s*=.*", f"n_rntis = {[1] * n}", text)
text = re.sub(r"(?m)^n_ids\s*=.*", f"n_ids = {[1] * n}", text)
text = re.sub(r"(?m)^dmrs_nid\s*=.*", f"dmrs_nid = {[[1, 1]] * n}", text)
text = re.sub(r"(?m)^max_num_tx\s*=.*", f"max_num_tx = {n}", text)
text = re.sub(r"(?m)^min_num_tx\s*=.*", "min_num_tx = 1", text)
name = f"mu_{label}_{args.prbs}_{args.channel}_{n}_export.cfg"
(Path("../config") / name).write_text(text)

params = Parameters(name, training=False, system="nrx", num_tx_eval=n)
generator = DataGeneratorAerial(params)
receiver = NeuralReceiverONNX(
    num_it=params.num_nrx_iter, d_s=params.d_s, num_units_init=params.num_units_init,
    num_units_agg=params.num_units_agg, num_units_state=params.num_units_state,
    num_units_readout=params.num_units_readout,
    num_bits_per_symbol=params.transmitters[0]._num_bits_per_symbol,
    layer_type_dense=params.layer_type_dense, layer_type_conv=params.layer_type_conv,
    layer_type_readout=params.layer_type_readout, nrx_dtype=params.nrx_dtype,
    num_tx=params.max_num_tx, num_rx_ant=params.num_rx_antennas)
inputs, _, _, _ = generator(1, 10.0)
receiver(inputs)
try:
    load_weights(receiver, f"../weights/{label}_weights")
    weights = "published"
except ValueError as err:
    raise SystemExit(f"weights of {label} do not load: {str(err).splitlines()[0][:200]}")
receiver._cgnn.num_it = params.num_nrx_iter_eval
llr, h_hat = receiver(inputs)

symbols, ant, num_tx = params.symbol_allocation[1], params.num_rx_antennas, params.max_num_tx
signature = [
    tf.TensorSpec([1, None, symbols, ant], tf.float32, name="rx_slot_real"),
    tf.TensorSpec([1, None, symbols, ant], tf.float32, name="rx_slot_imag"),
    tf.TensorSpec([1, None, num_tx, ant], tf.float32, name="h_hat_real"),
    tf.TensorSpec([1, None, num_tx, ant], tf.float32, name="h_hat_imag"),
    tf.TensorSpec(inputs[4].shape, tf.float32, name="active_dmrs_ports"),
    tf.TensorSpec(inputs[5].shape, tf.int32, name="dmrs_ofdm_pos"),
    tf.TensorSpec(inputs[6].shape, tf.int32, name="dmrs_subcarrier_pos"),
]
model, _ = tf2onnx.convert.from_keras(receiver, signature)
args.output_dir.mkdir(parents=True, exist_ok=True)
stem = f"{label}_{args.prbs}prb_{n}ue"
onnx.save(model, str(args.output_dir / f"{stem}.onnx"))
meta = {"label": label, "prbs": args.prbs, "num_tx": n, "channel": args.channel, "dmrs_ports": ports,
        "num_rx_ant": int(ant), "symbols": int(symbols), "num_it_eval": int(params.num_nrx_iter_eval),
        "weights": weights,
        "dmrs_ofdm_pos": np.asarray(inputs[5]).tolist(), "dmrs_subcarrier_pos": np.asarray(inputs[6]).tolist(),
        "input_shapes": {k: list(np.asarray(x).shape) for k, x in zip(
            ("rx_slot_real", "rx_slot_imag", "h_hat_real", "h_hat_imag", "active_dmrs_ports",
             "dmrs_ofdm_pos", "dmrs_subcarrier_pos"), inputs)},
        "output_shapes": {"llr": list(llr.shape), "h_hat": list(h_hat.shape)}}
(args.output_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta))

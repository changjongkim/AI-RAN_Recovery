#!/usr/bin/env python3
"""Uplink slots with 1-4 users on the same resources (MU-MIMO), for the conventional receiver only.

The published NVlabs weights know two users (two DMRS ports), so this script runs no neural receiver.
It writes the received slots and the payloads in the format of ``nv_dataset.py`` (``nrx_ok`` all False),
with the DMRS port of every user in ``meta["dmrs_ports"]``: users 1-4 on ports 0, 2, 1, 3, so that up to two
users are in different CDM groups and the third and fourth share a group through the frequency OCC.
``nv_conventional.py`` decodes the file with cuPHY.  Run from third_party/neural_rx/scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--base-config", default="nrx_large.cfg")
parser.add_argument("--prbs", type=int, default=273)
parser.add_argument("--num-tx", type=int, required=True, help="users on the same resources, 1-4")
parser.add_argument("--channel", default="UMi", help="UMi or UMa (the DoubleTDL models know two users only)")
parser.add_argument("--mcs", type=int, default=None)
parser.add_argument("--ebno", default="0,2,4,6,8,10")
parser.add_argument("--slots", type=int, default=100, help="slots per Eb/No point")
parser.add_argument("--batch", type=int, default=10)
parser.add_argument("--gpu", default="0")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert 1 <= args.num_tx <= 4

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("../")

import numpy as np                      # noqa: E402
import tensorflow as tf                 # noqa: E402

tf.get_logger().setLevel("ERROR")
for device in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)

from utils import Parameters, DataGeneratorAerial      # noqa: E402

PORTS = [0, 2, 1, 3]
base = Path("../config") / args.base_config
text = base.read_text()
label = re.search(r"label\s*=\s*'([^']+)'", text).group(1)
text = re.sub(r"n_size_bwp_eval\s*=.*", f"n_size_bwp_eval = {args.prbs}", text)
text = re.sub(r"(?m)^channel_type_eval\s*=.*", f'channel_type_eval = "{args.channel}"', text)
text = re.sub(r"(?m)^dmrs_port_sets\s*=.*", "dmrs_port_sets = [[0], [2], [1], [3]]", text)
text = re.sub(r"(?m)^n_rntis\s*=.*", "n_rntis = [1, 1, 1, 1]", text)
text = re.sub(r"(?m)^n_ids\s*=.*", "n_ids = [1, 1, 1, 1]", text)
text = re.sub(r"(?m)^dmrs_nid\s*=.*", "dmrs_nid = [[1, 1], [1, 1], [1, 1], [1, 1]]", text)
text = re.sub(r"(?m)^max_num_tx\s*=.*", "max_num_tx = 4 # four DMRS ports (no trained weights)", text)
if args.mcs is not None:
    text = re.sub(r"(?m)^mcs_index\s*=.*", f"mcs_index = [{args.mcs}]", text)
name = f"mu_{label}_{args.prbs}_{args.channel}_{args.num_tx}{'' if args.mcs is None else '_m' + str(args.mcs)}_p{os.getpid()}.cfg"
(Path("../config") / (name + ".tmp")).write_text(text)
(Path("../config") / (name + ".tmp")).rename(Path("../config") / name)

params = Parameters(name, training=False, system="nrx", num_tx_eval=args.num_tx)
generator = DataGeneratorAerial(params)
kept = {}
_channel = generator._channel


def keep_channel(inputs):
    y, h = _channel(inputs)
    kept["h"] = h
    return y, h


generator._channel = keep_channel


def pair_similarity(h):
    """[batch, 1, rx_ant, users, 1, symbol, subcarrier] -> per slot the mean and the max over user pairs of
    mean_subcarriers,symbols |h_i^H h_j| / (|h_i||h_j|)."""
    v = h.numpy()[:, 0, :, :, 0]                       # batch, rx_ant, users, symbol, subcarrier
    n = v.shape[2]
    if n < 2:
        z = np.zeros(v.shape[0], np.float32)
        return z, z
    norm = np.linalg.norm(v, axis=1)
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            num = np.abs(np.sum(np.conj(v[:, :, i]) * v[:, :, j], axis=1))
            vals.append((num / (norm[:, i] * norm[:, j] + 1e-12)).mean(axis=(1, 2)))
    vals = np.stack(vals, axis=1)
    return vals.mean(axis=1).astype(np.float32), vals.max(axis=1).astype(np.float32)


inputs, _, _, _ = generator(1, 10.0)

tx = params.transmitters[0]
meta = {
    "base_config": args.base_config, "label": label, "prbs": args.prbs, "num_tx": args.num_tx,
    "channel": args.channel, "num_rx_ant": int(params.num_rx_antennas),
    "mcs_index": int(params.mcs_index[0]), "mcs_table": int(params.mcs_table),
    "tb_size_bits": int(tx._tb_size), "num_bits_per_symbol": int(tx._num_bits_per_symbol),
    "target_coderate": float(tx._target_coderate),
    "dmrs_ofdm_pos": np.asarray(inputs[5]).tolist(), "dmrs_subcarrier_pos": np.asarray(inputs[6]).tolist(),
    "dmrs_ports": PORTS[:args.num_tx],
    "n_cell_id": int(params.n_cell_id), "slot_number": int(params.slot_number),
    "ebno_db": [float(e) for e in args.ebno.split(",")], "slots": args.slots, "neural_receiver": None,
}
print(json.dumps(meta), flush=True)

slots, bits, ebnos, corr_mean, corr_max = [], [], [], [], []
for ebno in meta["ebno_db"]:
    done = 0
    while done < args.slots:
        n = min(args.batch, args.slots - done)
        nrx_inputs, _, b, _ = generator(n, ebno)
        slots.append((nrx_inputs[0] + 1j * nrx_inputs[1]).astype(np.complex64))
        bits.append(np.packbits(np.asarray(b).astype(np.uint8), axis=-1))
        ebnos += [ebno] * n
        cm, cx = pair_similarity(kept["h"])
        corr_mean.append(cm); corr_max.append(cx)
        done += n
    print(f"ebno {ebno}: {done} slots", flush=True)

payload = np.concatenate(bits)
args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez(args.output, rx=np.concatenate(slots), payload=payload,
         nrx_ok=np.zeros(payload.shape[:2], dtype=bool), ebno=np.asarray(ebnos, dtype=np.float32),
         meta=json.dumps(meta), nrx_seconds=0.0, corr_mean=np.concatenate(corr_mean), corr_max=np.concatenate(corr_max))
print("saved", args.output, flush=True)

#!/usr/bin/env python3
"""Recovery against retransmission in a realistic setting (v40): the first reception of a slot and its HARQ resend
7.5 ms later on the same, time-correlated channel.

3GPP UMi (TR 38.901, NVlabs configuration: no path loss or shadowing, outdoor users), two users on the same resources,
users at one speed.  For every slot the channel model is sampled once over 7.5 ms + one slot, and the frequency
response of the first slot and of the slot 7.5 ms later are taken from that one realization, so the resend sees the
channel as it has changed at that speed (carrier 2.14 GHz).  The block is sent again with the same coded bits (RV 0).
Writes <prefix>_first.npz and <prefix>_retx.npz in the format of nv_dataset.py (no neural receiver here; the runtime
receivers decode them with probe_retx2.py), and per slot the similarity of the two users' channels:
  corr = mean over subcarriers and symbols of |h1^H h2| / (|h1| |h2|) over the 4 receive antennas (0: orthogonal,
         1: the same direction; hard to separate when high).
Run from third_party/neural_rx/scripts.
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
parser.add_argument("--speed", type=float, required=True, help="user speed in m/s")
parser.add_argument("--gap-ms", type=float, default=7.5, help="first transmission to resend")
parser.add_argument("--ebno", default="6,8,10")
parser.add_argument("--slots", type=int, default=150, help="slots per Eb/No point")
parser.add_argument("--batch", type=int, default=2)
parser.add_argument("--gpu", default="0")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("../")

import numpy as np                      # noqa: E402
import tensorflow as tf                 # noqa: E402

tf.get_logger().setLevel("ERROR")
for device in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)

from sionna.channel import ApplyOFDMChannel, cir_to_ofdm_channel, subcarrier_frequencies   # noqa: E402
from utils import Parameters, DataGeneratorAerial                                          # noqa: E402

text = (Path("../config") / args.base_config).read_text()
label = re.search(r"label\s*=\s*'([^']+)'", text).group(1)
text = re.sub(r"n_size_bwp_eval\s*=.*", f"n_size_bwp_eval = {args.prbs}", text)
text = re.sub(r"(?m)^channel_type_eval\s*=.*", 'channel_type_eval = "UMi"', text)
text = re.sub(r"max_ut_velocity_eval\s*=.*", f"max_ut_velocity_eval = {args.speed}", text)
text = re.sub(r"min_ut_velocity_eval\s*=.*", f"min_ut_velocity_eval = {args.speed}", text)
name = f"retx2_{label}_{args.prbs}_UMi_v{args.speed:g}.cfg"
(Path("../config") / name).write_text(text)

params = Parameters(name, training=False, system="nrx", num_tx_eval=2)
generator = DataGeneratorAerial(params)
assert not params.channel_norm, "the channel must not be normalized per slot"
model = params.channel_model
rg = params.transmitters[0]._resource_grid
symbols = int(rg.num_ofdm_symbols)
symbol_s = float(rg.ofdm_symbol_duration)
gap = int(round(args.gap_ms * 1e-3 / symbol_s))
frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
apply_channel = ApplyOFDMChannel(add_awgn=True)
kept = {}


def time_correlated(channel_inputs):
    """The channel of the generator, sampled over gap + one slot: the first slot and the slot gap symbols later."""
    x, no = channel_inputs
    a, tau = model(tf.shape(x)[0], gap + symbols, 1.0 / symbol_s)
    h1 = cir_to_ofdm_channel(frequencies, a[..., :symbols], tau, False)
    h2 = cir_to_ofdm_channel(frequencies, a[..., gap:gap + symbols], tau, False)
    y1 = apply_channel([x, h1, no])
    kept["y2"] = apply_channel([x, h2, no])
    kept["h1"], kept["h2"] = h1, h2
    return y1, h1


generator._channel = time_correlated
inputs, _, _, _ = generator(1, 10.0)


def aerial(y):
    return tf.transpose(tf.squeeze(y, axis=1), (0, 3, 2, 1)).numpy().astype(np.complex64)


def similarity(h):
    """[batch, 1, rx_ant, tx, 1, symbol, subcarrier] -> per slot, mean |h1^H h2| / (|h1||h2|) of the two users."""
    v = h.numpy()[:, 0, :, :, 0]                       # batch, rx_ant, tx, symbol, subcarrier
    u1, u2 = v[:, :, 0], v[:, :, 1]
    num = np.abs(np.sum(np.conj(u1) * u2, axis=1))
    den = np.linalg.norm(u1, axis=1) * np.linalg.norm(u2, axis=1) + 1e-12
    return (num / den).mean(axis=(1, 2))


tx = params.transmitters[0]
meta = {
    "base_config": args.base_config, "label": label, "prbs": args.prbs, "num_tx": 2, "channel": "UMi",
    "speed": args.speed, "gap_ms": args.gap_ms, "gap_symbols": gap, "carrier_hz": float(params.carrier_frequency),
    "num_rx_ant": int(params.num_rx_antennas), "mcs_index": int(params.mcs_index[0]), "mcs_table": int(params.mcs_table),
    "tb_size_bits": int(tx._tb_size), "num_bits_per_symbol": int(tx._num_bits_per_symbol),
    "target_coderate": float(tx._target_coderate),
    "dmrs_ofdm_pos": np.asarray(inputs[5]).tolist(), "dmrs_subcarrier_pos": np.asarray(inputs[6]).tolist(),
    "n_cell_id": int(params.n_cell_id), "slot_number": int(params.slot_number),
    "ebno_db": [float(e) for e in args.ebno.split(",")], "slots": args.slots, "neural_receiver": None,
}
print(json.dumps(meta), flush=True)

rx1, rx2, bits, ebnos, corr1, corr2, change = [], [], [], [], [], [], []
for ebno in meta["ebno_db"]:
    done = 0
    while done < args.slots:
        n = min(args.batch, args.slots - done)
        nrx_inputs, _, b, _ = generator(n, ebno)
        rx1.append((nrx_inputs[0] + 1j * nrx_inputs[1]).astype(np.complex64))
        rx2.append(aerial(kept["y2"]))
        h1, h2 = kept["h1"].numpy(), kept["h2"].numpy()
        corr1.append(similarity(kept["h1"]))
        corr2.append(similarity(kept["h2"]))
        # how much the channel changed in the gap: 1 - |<h1, h2>| / (|h1||h2|) over everything of a slot
        flat1, flat2 = h1.reshape(n, -1), h2.reshape(n, -1)
        change.append(1 - np.abs(np.sum(np.conj(flat1) * flat2, axis=1)) /
                      (np.linalg.norm(flat1, axis=1) * np.linalg.norm(flat2, axis=1) + 1e-12))
        bits.append(np.packbits(np.asarray(b).astype(np.uint8), axis=-1))
        ebnos += [ebno] * n
        done += n
    c = np.concatenate(corr1)[-args.slots:]
    d = np.concatenate(change)[-args.slots:]
    print(f"ebno {ebno}: user similarity median {np.median(c):.2f} (p90 {np.percentile(c, 90):.2f}), "
          f"channel change in {args.gap_ms} ms median {np.median(d):.3f}", flush=True)

args.output.parent.mkdir(parents=True, exist_ok=True)
payload = np.concatenate(bits)
zeros = np.zeros(payload.shape[:2], dtype=bool)
common = dict(payload=payload, nrx_ok=zeros, ebno=np.asarray(ebnos, dtype=np.float32), nrx_seconds=0.0,
              corr_first=np.concatenate(corr1).astype(np.float32), corr_retx=np.concatenate(corr2).astype(np.float32),
              channel_change=np.concatenate(change).astype(np.float32))
for kind, slots in (("first", rx1), ("retx", rx2)):
    path = args.output.parent / f"{args.output.name}_{kind}.npz"
    np.savez(path, rx=np.concatenate(slots), meta=json.dumps(dict(meta, reception=kind)), **common)
    print("saved", path, flush=True)

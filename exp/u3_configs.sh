#!/usr/bin/env bash
# U3 (user sweep): NVlabs configurations for more than four users on the same resources (no neural receiver weights;
# the classical receivers and the perfect-CSI bound run on them).  Four DMRS ports need one DMRS symbol of type 1, eight
# ports two symbols of type 1, and twelve ports two symbols of type 2 (TS 38.211 6.4.1.1).
#   nrx_large_8rx_mu8    8 RX, up to 8 users, DMRS type 1 with two symbols, ports 0-7
#   nrx_large_16rx_mu12  16 RX, up to 12 users, DMRS type 2 with two symbols, ports 0-11
# The covariance matrices of the LMMSE estimate are those of the same channel and antennas (nrx_large_8rx, _16rx).
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
C=$NRX_REPO/config; W=$NRX_REPO/weights
python3 - "$C" <<'PY'
import re, sys
c = sys.argv[1]
def make(src, label, users, cfg_type, length):
    t = open(f"{c}/{src}.cfg").read()
    sub = lambda pat, rep: re.subn(pat, rep, t, count=1)
    for pat, rep in ((r"label\s*=\s*'[^']+'", f"label = '{label}'"),
                     (r"(?m)^dmrs_config_type\s*=.*", f"dmrs_config_type = {cfg_type}"),
                     (r"(?m)^dmrs_length\s*=.*", f"dmrs_length = {length}"),
                     (r"(?m)^dmrs_nid\s*=.*", "dmrs_nid = " + str([[1, 1]] * users)),
                     (r"(?m)^dmrs_port_sets\s*=.*", "dmrs_port_sets = " + str([[p] for p in range(users)])),
                     (r"(?m)^n_rntis\s*=.*", "n_rntis = " + str([1] * users)),
                     (r"(?m)^n_ids\s*=.*", "n_ids = " + str([1] * users)),
                     (r"(?m)^max_num_tx\s*=.*", f"max_num_tx = {users} # {users} DMRS ports (no trained weights)")):
        t, n = sub(pat, rep)
        assert n == 1, (label, pat)
    open(f"{c}/{label}.cfg", "w").write(t)
make("nrx_large_8rx_mu4", "nrx_large_8rx_mu8", 8, 1, 2)
make("nrx_large_16rx_mu4", "nrx_large_16rx_mu12", 12, 2, 2)
print("configs written")
PY
for k in freq time space; do
  ln -sf nrx_large_8rx_${k}_cov_mat.npy $W/nrx_large_8rx_mu8_${k}_cov_mat.npy
  ln -sf nrx_large_16rx_${k}_cov_mat.npy $W/nrx_large_16rx_mu12_${k}_cov_mat.npy
done
ls -la $W | grep -E "8rx_mu8|16rx_mu12" | awk '{print $9, $10, $11}'

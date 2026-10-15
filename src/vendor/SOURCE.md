# Vendored code

Copied on 2026-10-14 from `airan_cloudlab` at commit 22952643 so that this project does not change when the Antiphase
code changes. Edit these copies here; do not edit the originals for this project.

| File | Original |
|---|---|
| slot_radio.py, mu_radio.py, cell_ring.py, probe_ldpc.py, ul_profiles.py | scripts_for_node/backstop_slot/ |
| persistent_pusch_rx.py | scripts_for_node/softwall_same_gpu/ |
| nrx_trt_direct.py | scripts_for_node/task1/ |
| nv_build_dataset.py, nv_export_mu.py, nv_engine_users.sh, nv_retx_dataset2.py | scripts_for_node/backstop_slot/nv/ |
| probe_retx3.py | scripts_for_node/backstop_slot/ (reference for the v40b decoder) |

Large assets stay in airan_cloudlab and are mounted read-only by env.sh: the Aerial repository
(third_party/aerial-cuda-accelerated-ran), the NVlabs neural_rx repository (third_party/neural_rx), the TensorFlow and
Sionna packages (runtime/softwall_same_gpu/nrx_export_deps), and the TensorRT engines (runtime/softwall_same_gpu/engines).

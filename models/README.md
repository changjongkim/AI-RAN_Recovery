# 보관한 모델

| 파일 | 내용 | 학습 |
|---|---|---|
| nrx_large_ici_240k_weights | 간섭을 아는 Large NRx (2명, 4 RX) | NVIDIA nrx_large에서 셀 간 간섭(이웃 셀 2명, 잡음 대비 0–10 dB, 25% 없음)으로 24만 반복, `exp/n15_train.sh` |
| nrx_large_ici_273prb_2ue.trt / .json | 위 모델의 TensorRT 엔진(FP16, 273 PRB, A100) | `exp/n15_export.sh`; 원 모델을 같은 경로로 내보내 기존 엔진과 블록 단위 100% 일치 확인 |
| nrx_large_8rx_step25_weights | 8 RX Large NRx | 4 RX 가중치를 넓혀(`src/nrx_widen.py`) 약 86만 반복, `exp/n4b_train.sh` (장기 학습 진행 중) |
| nrx_large_mu4_step16_weights | 1–4명 Large NRx (4 RX) | NVIDIA nrx_large에서 약 65만 반복, `exp/n4_train.sh` (장기 학습 진행 중) |

가중치 파일은 NVlabs `load_weights`(pickle)로 읽는다. 장기 학습이 끝나면 마지막 단계를 여기에 더한다.

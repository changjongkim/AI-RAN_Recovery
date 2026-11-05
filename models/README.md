# 보관한 모델

| 파일 | 내용 | 학습 |
|---|---|---|
| nrx_large_ici_240k_weights | 간섭을 아는 Large NRx (2명, 4 RX) | NVIDIA nrx_large에서 셀 간 간섭(이웃 셀 2명, 잡음 대비 0–10 dB, 25% 없음)으로 24만 반복, `exp/n15_train.sh` |
| nrx_large_ici_273prb_2ue.trt / .json | 위 모델의 TensorRT 엔진(FP16, 273 PRB, A100) | `exp/n15_export.sh`; 원 모델을 같은 경로로 내보내 기존 엔진과 블록 단위 100% 일치 확인 |
| nrx_large_8rx_step25_weights | 8 RX Large NRx | 4 RX 가중치를 넓혀(`src/nrx_widen.py`) 약 86만 반복, `exp/n4b_train.sh` (장기 학습 진행 중) |
| nrx_large_mu4_step16_weights | 1–4명 Large NRx (4 RX) | NVIDIA nrx_large에서 약 65만 반복, `exp/n4_train.sh` (장기 학습 진행 중) |
| nrx_large_mu4_20261019_weights | 1–4명 Large NRx (4 RX), N4 학습의 2026-10-19 마지막 상태 | U3n 4 RX 평가에 씀 |
| nrx_large_8rx_mu8_u3n_weights, nrx_large_16rx_mu12_u3n_weights, nrx_large_mu4_u3n_weights | 8 RX 1–8명, 16 RX 1–12명(8 RX 가중치를 16 RX로 넓힘), 4 RX 1–4명 이어 학습 | `exp/u3n_train.sh`, 2026-11-03 13:00까지 학습 뒤 `exp/u3n_side.sh`가 복사 |
| nrx_large_8rx_mu4c_u3s_weights, nrx_large_8rx_mu4d_u3s_weights | 8 RX 1–4명, 1-symbol DMRS, 포트 0,2,1,3 (c: nrx_large_8rx 가중치에서 lr 5e-4, d: 4 RX mu4 U3n 가중치를 8 RX로 넓혀 lr 5e-4) | `exp/u3s_train.sh`, 2026-11-04 학습(c 18:50, d 20:42까지); 여섯 변형 모두 results/u3s/*_final |

가중치 파일은 NVlabs `load_weights`(pickle)로 읽는다. 장기 학습이 끝나면 마지막 단계를 여기에 더한다.

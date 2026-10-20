# 간섭 학습 신경 수신기 관련 연구 조사 (기여 2 확인)

조사일: 2026-10-19. 방법: 웹 검색(arXiv·IEEE·NVIDIA 문서), 초록·본문 확인. 본문까지 읽은 것은 [본문], 초록·요약만 본 것은 [초록]으로 표시.

## 확인할 주장

공개 NVlabs NRx는 한 셀, 셀 간 간섭 없이 학습됨. AI-RAN 서버에서 큰 NRx를 CRC 실패 블록의 복구 경로로 쓸 때, 셀 간 간섭(IoT 3–10 dB)이 있으면 MMSE-IRC 대비 복구 이득이 거의 사라짐. 셀의 간섭에 맞춰 서버 GPU에서 미세 조정하면 이득이 돌아옴(3 dB에서 IRC 대비 복구 3배).

## 1. NVlabs NRx 자체

**Wiesmayr, Cammerer, Ait Aoudia, Hoydis, Zakrzewski, Keller, "Design of a Standard-Compliant Real-Time Neural Receiver for 5G NR"**, arXiv 2409.02912 (2024). NVIDIA 연구 페이지에는 2025-05로 올라 있음. ICMLCN 2025 게재는 arXiv 페이지에서 확인하지 못함(NVIDIA 페이지 날짜와는 맞음). https://arxiv.org/abs/2409.02912 [본문]
- 학습은 3GPP UMi, 평가는 DoubleTDL(TDL-B와 TDL-C), 사이트 특화는 Sionna RT(뮌헨 지도).
- 본문에 inter-cell, co-channel, interference covariance, IRC라는 말이 없음. 셀 안 MU-MIMO 사용자 간 간섭은 GNN으로 함께 검출함. 기준선은 LS/LMMSE 추정에 LMMSE 또는 K-best 검출이며 IRC는 없음.
- 사이트 특화 미세 조정을 함: UMi로 사전 학습한 뒤 ray tracing 데이터로 N_FT 단계 조정. 1,000단계에 약 30초(RTX 3090). 데이터가 적으면 과적합하고, 조정하면 UMi 성능을 잊음(catastrophic forgetting).
- HARQ 언급 없음.
- 우리와의 차이: 조정 대상이 채널(장소)이고 간섭이 아님. 간섭 아래에서 IRC와 비교하지 않음.

**Cammerer et al., "A Neural Receiver for 5G NR Multi-user MIMO"**, IEEE Globecom Workshops 2023, arXiv 2312.02601. [초록] 같은 구조의 앞선 논문. 초록에 셀 간 간섭 언급 없음.

**github NVlabs/neural_rx README** [본문]: interference, inter-cell, multi-cell 언급 없음. 3GPP 채널 모델 학습과 Sionna RT 사이트 특화 학습만 있음.

**Baytekin, Wiesmayr, Cammerer, Dick, Studer, "Site-Specific Finetuning of Neural Receivers with Real-World 5G NR Measurements"**, arXiv 2603.09644 (2026, SPAWC 2026 투고). https://arxiv.org/abs/2603.09644 [초록] ETH 시험망의 실제 PUSCH로 사이트 특화 미세 조정(실험실·사무실·실외 고속). 오류율이 크게 줄고 UE 기기가 달라도 유지됨. 초록에 셀 간 간섭 언급 없음. 쓴 NRx가 NVlabs 것인지 초록으로는 확인 못 함(저자가 같아 가능성은 높음).

판단: "공개 NVlabs NRx는 셀 간 간섭 없이 학습되었고 IRC와 비교되지 않았다"는 확인됨. 다만 "AWGN + UMi"에서 AWGN이라는 표현은 논문에 따로 없음. "UMi(열잡음만)"로 쓰는 것이 안전함. 또 "간섭 없음"이 아니라 "셀 간 간섭 없음"으로 써야 함(셀 안 MU-MIMO 간섭은 다룸).

## 2. 간섭을 넣어 학습·평가한 신경 수신기

**Honkala, Korpi, Huttunen, "DeepRx: Fully Convolutional Deep Learning Receiver"**, IEEE TWC 20(6), 2021, arXiv 2005.01494. [본문]
- SIMO 1x2 상향. 간섭자는 같은 파형, 임의 시간 차, 다른 채널. 간섭 전력은 평균적으로 잡음보다 4 dB 낮음. 간섭을 넣은 데이터로 학습함.
- 기준선은 간섭 처리가 없는 LMMSE. 저자 스스로 "IRC를 쓰면 이 차이가 줄 것"이라고 적음(IRC와 비교하지 않음).
- 우리와의 차이: 처음부터 간섭으로 학습. 간섭 없이 학습한 모델이 간섭에서 무너지는지, 미세 조정으로 회복되는지는 보지 않음. IRC 비교 없음.

**Honkala, Korpi, Raninen, Huttunen, "EqDeepRx: Learning a Scalable and Interference Mitigating MIMO Receiver"**, arXiv 2602.11834 (2026, IEEE 투고). https://arxiv.org/abs/2602.11834 [본문 대부분]
- 이웃 셀 UE 한 명이 겹쳐 보냄(임의 시간 차). INR은 Lognormal(10 dB, 5 dB). 샘플마다 간섭자 0 또는 1명으로 학습.
- 기준선은 간섭+잡음 공분산(INCM)을 shrinkage로 추정하는 LMMSE, 즉 사실상 MMSE-IRC형. 10% BLER에서 약 2 dB(CDL-C·CDL-D), 4 dB 이상(UMa·UMi) 이득.
- 앞선 DeepRx MIMO가 "셀 간 간섭을 다루지 못했다"고 명시함.
- 우리와 가장 많이 겹치는 논문. "간섭으로 학습한 신경 수신기가 IRC형 기준선을 이긴다"는 이미 있음.
- 차이: Nokia 자체 구조이고 공개 NVlabs NRx가 아님. 간섭 없이 학습한 모델의 성능 저하와 미세 조정 회복을 보지 않음. 셀별 간섭에 맞춘 현장 조정이 아님. HARQ나 실패 블록 복구 없음.

**Obeed, Jian (Huawei Canada), "CoNet-Rx: Collaborative Neural Networks for OFDM Receivers"**, arXiv 2510.12739 (2025). [본문 일부] 같은 numerology의 OFDM 간섭자(임의 채널·시간 차)를 넣어 학습. 간섭 비율은 10–90%. 기준선은 DeepRx와 LMMSE이며 IRC는 없음.

**Korpi et al., "DeepRx MIMO"**, ICC 2021, arXiv 2010.16283. [확인 못 함] EqDeepRx에 따르면 간섭 없는 조건용.

**Chen, He, "IntLearner"**, arXiv 2308.02167 (2023). [초록] 다중 셀에서 간섭 특징을 학습해 제거. 상향 채널 추정 최대 7.4배, 하향 1.5 dB. 기준선은 간섭을 백색 잡음으로 보는 방식이며 IRC가 아님. 전체 신경 수신기도 아님.

학습형 IRC(신경망으로 간섭 공분산을 추정해 IRC에 넣는 것)는 특허(US 12136979 등)와 조사 논문(arXiv 2401.01608) 언급만 찾음. 동료 심사 논문은 확인 못 함.

## 3. 현장 적응·온라인 미세 조정

**Fischer, Dörner, Cammerer, Shimizu, Lu, ten Brink, "Adaptive Neural Network-based OFDM Receivers"**, SPAWC 2022, arXiv 2203.13571. [본문]
- FEC로 복원한 라벨로 온라인 재학습("on-the-fly label recovery", Schibisch et al. ISWCS 2018).
- 간섭 사례가 있음. 다만 바깥 네 부반송파의 SNR을 6 dB 낮추는 대역 가장자리 잡음이며 셀 간 공채널 간섭이 아님. 재학습 이득은 0.42 dB. 단일 안테나 RNN이고 IRC 비교는 없음.
- "현장에서 간섭에 맞춰 재학습한다"는 생각 자체는 이미 있음. Cammerer가 공저자.

**"Learning During Detection: Continual Learning for Neural OFDM Receivers via DMRS"** (Huawei), arXiv 2602.20361 (2026). [초록] DMRS를 라벨로 써서 지속 학습. 채널 변화 추적이 목적이며 간섭 언급은 확인 못 함.

**"On Transfer Learning for a Fully Convolutional Deep Neural SIMO Receiver"**, arXiv 2408.16401 (2024). [초록] CDL-C에서 UMi로 옮길 때 미세 조정. 간섭 없음.

**Obeed, Jian, "Hybrid Neural/Traditional OFDM Receiver with Learnable Decider"**, arXiv 2509.18574 (2025). [초록] 블록마다 신경/기존 수신기를 고름. 분포 이동 대책이며 HARQ나 CRC 기반은 아님.

**Elnashar, "When Does a Neural Receiver Help? Calibration-Drift Benchmarking and Detect-and-Rollback"**, arXiv 2605.26157 (2026). [검색 요약만] DeepRx류의 분포 밖 저하를 재고, 복호 전에 기존 수신기로 되돌림. 검색 요약에 따르면 CRC 기반 되돌림은 향후 과제로 남김. 간섭이 drift 조건에 들어가는지는 확인 못 함.

**NVIDIA pyAerial / Aerial Data Lake** (문서): 기지국 fronthaul IQ와 L2 메타데이터를 모아 L1 노드(예: 신경 디매퍼)를 학습하는 도구. GPU 기지국 데이터로 학습하는 경로는 제품 차원에서 있음. 간섭 맞춤 조정을 보인 결과는 찾지 못함.

## 4. 신경 수신기와 HARQ·실패 블록 재복호

- 신경 수신기를 CRC 실패 블록에만 다시 돌려 HARQ 재전송 전에 복구하는 연구는 찾지 못함.
- 가까운 것: E-HARQ(Fraunhofer HHI, 복호 결과 예측), "Neural CRC Prediction for 5G NR URLLC"(arXiv 2608.06230, 결과 예측), BiLSTM으로 HARQ를 결합하는 장(NITK, 세부 확인 못 함), 위의 Elnashar(복호 전 되돌림, CRC 방식은 향후 과제).
- AI-RAN 쪽(ScaleRAN arXiv 2610.23072 등)은 남는 GPU를 AI에 돌리는 데 집중하고, 실패 블록 재복호에 쓰지 않음.
- 같은 작업 디렉터리의 "AI RAN 실패 블록 복구 관련 연구.md"와 함께 볼 것.

## 판단

겹치는 부분
1. 간섭으로 학습한 신경 수신기가 간섭 아래에서 기존 수신기보다 낫다: DeepRx(2021), EqDeepRx(2026, IRC형 INCM 기준선 대비 2–4 dB), CoNet-Rx. 새 주장으로 내세우면 안 됨.
2. 신경 수신기를 현장에 맞춰 미세 조정한다: NVlabs 논문(ray tracing), Baytekin et al.(실측), Fischer et al.(온라인, 대역 가장자리 간섭 포함). 미세 조정 자체는 새롭지 않음.
3. 신경 수신기가 학습 분포 밖에서 무너진다: OOD 연구(arXiv 2505.15570, 2605.26157)와 DeepRx의 자기 언급. 일반론으로는 알려져 있음.

새로운 부분(조사 범위 안에서 선행을 찾지 못함)
- 공개 NVlabs NRx(AI-RAN에서 실제로 쓸 수 있는 공개 모델)를 셀 간 간섭(IoT 3–10 dB) 아래에서 MMSE-IRC와 비교해 이득이 거의 사라짐을 측정한 것.
- 그 이득을 HARQ 재전송 전 CRC 실패 블록 복구 수로 잰 것. 복구 경로라는 쓰임새 자체가 선행을 찾지 못한 부분임.
- 셀의 간섭 조건에 맞춘 미세 조정을 같은 AI-RAN 서버 GPU에서 하는 운영 방식과, 그 결과(3 dB에서 IRC 대비 복구 3배).

주의할 점
- "간섭 학습이 이득을 되살린다"는 메커니즘은 DeepRx·EqDeepRx로 예상 가능함. 기여는 메커니즘이 아니라 "공개 모델을 그대로 쓰면 복구 경로의 값이 IRC 앞에서 사라진다"는 측정과 배치 방식에 둘 것.
- IRC 기준선은 우리가 직접 구현한 것이므로 공분산 추정 방식(DMRS 기반, PRB 단위 등)을 밝힐 것. EqDeepRx의 shrinkage INCM과 비교 가능해야 함.
- 셀 안 MU-MIMO 간섭은 NVlabs NRx가 다룬다는 점을 구분해서 쓸 것.

논문에 넣을 문장(안)
- 영어: "The public NVlabs neural receiver is trained on single-cell 3GPP UMi channels without inter-cell interference. Interference-trained neural receivers are known to outperform covariance-based combining [DeepRx, EqDeepRx], but we find that the public receiver, used as-is, loses nearly all of its recovery gain over MMSE-IRC once neighbor-cell users occupy the same PRBs (IoT 3–10 dB); fine-tuning it on the AI-RAN server to the cell's interference restores the gain (3x more recoveries than IRC at 3 dB)."
- 한국어: "공개 NVlabs NRx는 셀 간 간섭이 없는 단일 셀 UMi로 학습되었다. 간섭으로 학습한 신경 수신기가 공분산 기반 결합보다 낫다는 것은 알려져 있지만[DeepRx, EqDeepRx], 공개 모델을 그대로 쓰면 이웃 셀 간섭(IoT 3–10 dB) 아래에서 MMSE-IRC 대비 복구 이득이 거의 사라지고, AI-RAN 서버에서 셀의 간섭에 맞춰 미세 조정하면 되살아남(3 dB에서 복구 3배)을 보인다."

## 확인하지 못한 것
- 2409.02912의 ICMLCN 2025 게재(NVIDIA 페이지 날짜만 확인).
- 2603.09644가 NVlabs NRx를 쓰는지, 간섭을 다루는지(초록만 봄).
- EqDeepRx 본문의 마지막 약 3%와 결론의 "22 dB/44 dB" 숫자(본문의 2 dB/4 dB와 어긋남. 오타로 보임).
- Elnashar 2605.26157 본문(HTML 404).
- Ericsson·Qualcomm의 신경 수신기 대 IRC 공개 결과(검색에서 찾지 못함. 3GPP Rel-19/20 AI/ML 기고문은 보지 않음).

# Recovering failed uplink blocks before HARQ in AI-RAN servers

논문 A의 작업 디렉터리. Antiphase(논문 B, `airan_cloudlab`)와 분리했다. 이 디렉터리의 코드·결과·큐는 Antiphase 쪽과
섞이지 않는다. 큰 자산(Aerial 이미지와 저장소, TensorRT 엔진, TensorFlow·Sionna 패키지)만 `airan_cloudlab`에서 읽는다
(`env.sh`, `src/vendor/SOURCE.md`).

## 1. 구조의 정의

| 항목 | 정의 |
|---|---|
| AI-RAN 서버 | GPU에서 여러 셀의 L1(cuPHY)과 AI 추론을 함께 실행하는 기지국 서버. 각 상향 슬롯의 수신 신호가 GPU 메모리에 남는다 |
| 세 작업 | L1 기존 수신기(모든 블록, 필수), 복구(CRC가 실패한 블록만, 선택), AI(남는 GPU) |
| 블록의 두 마감 | L1 마감 4.0 ms(모든 블록). 복구 마감: 비동기 상향 HARQ가 허용하는 범위 안에서 고르는 설계값(11.5 ms). 규격은 재전송 grant 시점의 하한(K2, PUSCH 준비 시간)만 정하고 상한은 DRX 타이머뿐이다 (TS 38.300 §5.3.5.4, 38.321 §5.4.2.1, 38.214 §6.1/§6.4) |
| 복구 | CRC가 실패한 블록을 이미 받은 샘플로, 채널 추정부터 다른(더 강한) 수신기로 다시 처리하는 것. 재전송 grant 전에 성공하면 재전송을 요청하지 않는다. 실패하면 표준 HARQ로 간다 |
| 복구와 구별할 것 | HARQ(새로 받음), L1 안의 NRx(모든 블록을 마감 안에), 복호기만 바꾸는 재복호(같은 LLR), 예측 기반 HARQ(Nuberu, Hades) |
| AI-RAN Alliance 분류와의 관계 | AI-for-RAN(NRx), AI-and-RAN(L1과 AI의 GPU 공유), AI-on-RAN(LLM)이 한 GPU에서 만나는 구조 |

## 2. 문제 정의

> In 5G NR uplink, a transport block that fails CRC is handled only by HARQ: each failure costs a scarce uplink slot,
> user transmit energy and a round trip, and failures concentrate where co-scheduled users are hard to separate. Neural
> receivers separate such users better, but the models that help most miss the per-block Layer-1 deadline.
> Asynchronous uplink HARQ gives failed blocks a later deadline, and AI-RAN servers already hold their samples and spare
> GPU time. We ask whether the base station can recover failed blocks before asking for a resend, and what that buys.

서론의 논리와 근거(규격 조항, 운용 측정, 기존 연구의 문장)는 `docs/PLAN.md` 1절, 관련 연구 전체는 `docs/related_work.md`.

## 3. 기여

1. 구조와 두 마감의 정의, 표준(비동기 상향 HARQ) 안의 복구 마감.
2. 복구 경로: CRC가 실패한 블록만 이미 받은 샘플로 Large NRx가 다시 풀고, HARQ와 병렬로 둔다. 어떤 조건에서도 HARQ만 쓰는 것보다 지는 지표가 없다.
3. 실패의 원인과 복구의 값: 실패와 복구는 같은 자원을 쓰는 사용자 신호가 겹치는 쌍(상향 MU-MIMO의 기본 상황)에 몰리고, 대형 기지국(UMa)에서 값이 크다. Large NRx가 고전 강한 수신기를 모두 합친 것보다 많이 살린다.
4. 시스템 수준의 값: 링크 적응 처리량, 짝 고르기의 대체, 지연 꼬리, 끝내 못 받는 블록.
5. 복구 비용과 고르기: GPU 한 장의 실측 처리량, L1 결과만으로 살릴 블록을 고르는 예측기, 작은 NRx → Large NRx 연쇄.

## 4. 핵심 결과 (2026-10-18; 전체 표와 분석은 [docs/RESULTS.md](docs/RESULTS.md), 데이터 표는 `results/summary/`)

| 주장 | 수치 (괄호: 묶음 재표본 95% 구간) |
|---|---|
| 재전송 감소 | 소형 기지국(UMi) 0.75배 [0.74–0.76], 대형(UMa) 0.42–0.44배 [0.40–0.46], 대형·비가시선 0.26–0.27배 |
| 끝내 못 받는 블록 | UMi 2.21 → 1.48%, UMa 2.15 → 0.84%, UMa 비가시선 1.11 → 0.32% |
| 유지되는 범위 | 3–120 km/h, MCS 12/14/16, 재전송 혼자/같은 짝, 가시선/비가시선, Eb/No 2–16 dB: 지는 조건 없음 |
| 효과가 몰리는 곳 | 짝 상관 ≥ 0.4(블록의 66%)에서 실패의 27–29%, 0.4 미만 14% (UMi) |
| 수신기 비교 (같은 슬롯) | 살린 실패 Large NRx 42.5/54.1%, K-best 37.0/42.9, 작은 NRx 30.4/39.1, IDD 19.4/20.6, LDPC 50회 7.0/9.9 (UMi/UMa); 다른 방법만 살리는 블록 0.6–0.9% |
| 지연 | 병렬 복구: 10 ms 안에 못 받는 블록 4.40 → 3.19%, 복구될 블록 9.4 → 7.7 ms |
| 처리량 | 링크 적응 목표 10%: UMi +14–21%, UMa +22–42%; 목표 3%: UMa 최대 +75% |
| 짝 고르기 | 무작위 짝 + 복구 ≈ 8명 중 고른 짝 + 기존 수신기 (UMi 재전송 12.2 대 12.8, UMa 9.1 대 16.4) |
| 고르기 | 실패 조각 1개 98%, 2–5개 84–93% 복구; 조각 ≤ 5면 기다리고 나머지 병렬: 지연 그대로 재전송 17.5 → 14.5 |
| GPU | Large NRx는 A100 한 장을 다 쓰며 초당 233회(4.3 ms, 동시 실행해도 늘지 않음); 기지국 하나가 초당 92–96회 필요 → 전부 복구하면 A100 한 장에 약 2.4개 |
| 약점 | 짧게 추가 학습한 3–4명·8 RX 모델은 K-best보다 약함(8 RX UMa 제외) → "Large NRx가 필요하다"는 원 모델(2명·4 RX)에서만 주장; 추가 학습 중 |

## 5. 실험

| ID | 실험 | 코드 | 상태 |
|---|---|---|---|
| N1·N2 | IR HARQ(RV 0·2·3·1, 최대 4회), UMi 3/30/60 km/h, 조건당 2만 블록 | `src/gen_harq.py`, `src/decode_harq.py`, `src/harq_stats.py`, `exp/harq_ir.sh` | 끝남. 재전송 0.75배, 복구 22–23%, 속도 무관(전대역이라 이득·짝 상관이 22 ms 안에 안 변함) |
| N1b | 재전송은 혼자 보냄 | `gen_harq.py --retx-alone` | 끝남. UMi 0.77배, UMa 0.48배, UMa 비가시선 0.26배 |
| N3 | UMi 120, UMa 3/30/120 km/h, UMa 가시선/비가시선 (UMa는 시드 2000으로 두 배) | `exp/harq_ir.sh` 옵션 | 끝남. UMa 복구 56–58%·0.42–0.44배, 비가시선 72–73%·0.26–0.27배, 가시선 27–28%·0.69–0.70배 |
| P1 | UMi 가시선/비가시선; 안테나 2/4/8 (UMi·UMa, 기존 수신기와 상한) | `exp/harq_ir.sh`, `exp/classical.sh --num-rx` | 끝남. 가시선 0.75배·비가시선 0.62배; 안테나가 늘어도 상한은 실패의 47–79%를 살림. 가시선/비가시선 시드 추가 진행 중 |
| P2 | 사용자 U명 중 짝 고르기(무작위, 이득·상관 기준, 상관만) × 복구 | `gen_harq.py --drop-users --pairing` | 끝남. 무작위 + 복구 ≈ 고른 짝 + 기존 수신기; 상관만 보고 고르면 무작위보다 나쁨(약한 사용자). 첫 계산(표 기반)은 무효 |
| N5 | 같은 슬롯에서 다른 복구 방법: LDPC 50회, IDD(SIC), LMMSE+K-best, 완전 CSI+K-best, Large/작은 NRx | `src/classical_rx.py`, `src/idd.py` | 끝남(UMi 실패 2,797, UMa 233). UMa 시드 추가 진행 중 |
| N6 | 외부 루프 링크 적응(MCS 10–16 같은 채널) | `exp/harq_mcs.sh`, `src/sim/olla.py` | 끝남(UMi, UMa, UMa 가시선/비가시선) |
| N7 | HARQ와 합치는 방식(기다림·병렬·선택)의 지연 | `src/sim/harq_latency.py` | 끝남. 병렬은 지는 지표 없음; 조각 수로 선택 |
| N8 | 셀당 사용자 1–4명: cuPHY 실패, 3–4명 복구(미세 조정 NRx 대 K-best) | `exp/users.sh`, `exp/classical.sh --users` | 끝남. 실패 16→24→36→53%; 3–4명에서는 K-best가 NRx보다 많이 살림 |
| N9 | GPU 수요 | `src/sim/demand.py` | 끝남. 슬롯의 23–24%가 복구 필요(동시 실행 가정은 N13으로 바로잡음) |
| MCS·wide | MCS 12/16, Eb/No 2·4·14·16 dB (UMi·UMa) | `MCS=…`, `--ebno` | 끝남. 같은 경향; 신호가 약하면 효과 작음, 좋으면 큼 |
| N4 | 1–4명 Large NRx 미세 조정(2명 가중치에서, 29만 반복) | `exp/n4_train.sh` | 끝남; 2명 성능은 떨어져 사용자 수별로 모델을 나눔. 53만 반복까지 추가 학습 중(queue_b) |
| N4b | 8 RX Large NRx: 4 RX 가중치를 넓혀 미세 조정(49만 반복) | `src/nrx_widen.py`, `exp/n4b_train.sh` | UMi에서 K-best보다 약간 낮고(38.8 대 41.7%) UMa에서 앞섬(50.2 대 45.5%); 학습률을 바꿔 21만 반복 추가 중(queue) |
| N10·N11 | L1 결과만으로 복구 성공 예측(조각 수, 짝의 성공, 상관, 이득), GPU 예산 대비 복구 가치 곡선 | `src/analysis/predictor.py` | 계산 중 |
| N12 | 수신기 합집합과 연쇄(작은 NRx 먼저) | `src/analysis/union.py` | 끝남. 연쇄는 같은 양을 살리며 Large NRx 실행 30–39% 감소 |
| N13 | Large/작은 NRx의 A100 동시 실행 처리량·지연 | `src/nrx_lanes.py` | 끝남. Large 233회/초(4.3 ms), 작은 740회/초(1.3 ms); 동시 실행해도 처리량은 늘지 않음 |

## 6. 실행

- 큐 둘: `queue/`(노드 1)와 `queue_b/`(노드 2). 각 실행기가 interactive 노드를 스스로 잡고(사용자당 job 2개·job당 4노드·4시간), 다음 단계가 남은 시간에 안 들어가면 놓고 다시 잡는다. `queue/nodes`가 1보다 크면 첫 노드에서 단계를, 나머지 노드에서 부수 작업 묶음(`exp/side_pool.sh`)을 돌린다.
- `bash queue/watchdog.sh start`가 10분마다 두 실행기를 확인해 다시 띄운다(`queue/watchdog.log`). 멈추려면 그 큐에 `STOP` 파일. 단계는 `todo`, 끝난 단계는 `done`, 기록은 `runner.log`, 단계별 출력은 `logs/`.
- 한 조건: `bash exp/harq_ir.sh NAME CHUNKS --channel UMi --speed 0.83 --ebno 6,8,10,12 --slots 64` (GPU 네 장에서 묶음 병렬; `SEED0`로 시드, `MCS`로 MCS).
- 결과 표 다시 만들기: `python3 src/analysis/summarize.py` (nersc-python 3.13), 구간: `src/ci.py`.

## 7. 디렉터리

```
env.sh            컨테이너 경로와 실행 함수(rx: cuPHY·pyAerial·TensorRT, tfgen: TensorFlow·Sionna)
src/              실험 코드; src/vendor는 airan_cloudlab에서 복사한 수신기 코드(SOURCE.md); src/sim 시뮬레이션; src/analysis 분석
exp/              실험 구동 스크립트
queue/, queue_b/  이 프로젝트 전용 큐 실행기
docs/             계획(PLAN.md), 관련 연구(related_work.md), 결과와 분석(RESULTS.md)
results/          결과(json, txt; 묶음별 npz는 git에서 제외), results/summary 모든 결과의 표
data/             생성한 수신 신호(git에서 제외, 묶음이 끝나면 지움)
third_party/      NVlabs neural_rx 사본(설정 파일을 쓰므로 복사; git에서 제외)
```

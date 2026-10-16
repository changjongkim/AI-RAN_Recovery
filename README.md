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

## 3. 기여 (안)

1. 구조와 두 마감의 정의, 표준 안의 복구 마감 유도
2. 실패가 어디서 생기고 무엇이 그것을 푸는가: 사용자 상관·수신 안테나·사용자 수별 실패 분포, 수신기 설계 공간
3. 현실 조건의 값: IR HARQ, OLLA 링크 적응, 공격적 페어링에서 재전송·goodput·상향 셀 용량·지연·단말 송신
4. 복구 경로의 설계와 비용: 어떤 실패를 다시 푸나, HARQ와 합치는 방식, GPU 수요

## 4. 실험

| ID | 실험 | 코드 | 상태 |
|---|---|---|---|
| N1·N2 | IR HARQ(RV 0·2·3·1, 최대 4회), 시간에 따라 변하는 3GPP 채널, 조건당 블록 2만 개 | `src/gen_harq.py`, `src/decode_harq.py`, `src/harq_stats.py`, `exp/harq_ir.sh` | UMi 3 km/h 끝남(2026-10-15, `results/n1_umi_v3.json`, 블록 20,480개): 첫 실패 10.3%, 복구가 실패의 22.8%를 재전송 없이 끝냄, 100블록당 재전송 17.5 → 13.2(0.75배; 복구만 0.86배), 4회 뒤 실패 2.21 → 1.48%(0.67배); 작은 NRx를 모든 블록에 쓰면 재전송 16.8. 상관 0.4 미만/0.4–0.6/0.6 이상(블록 35/50/16%)에서 복구 비율 14/27/29%; 같은 상관 구간 안에서는 약한 사용자(채널 이득 하위 25%)와 나머지의 복구 비율이 같음(13.7/14.0, 27.1/27.1, 27.7/29.9%) → 복구 비율은 사용자 이득이 아니라 두 사용자의 상관을 따름. 상관 0.4 미만의 첫 실패가 높은 것은 그 구간에 약한 사용자가 많기 때문(실패의 55%). UMi 30 km/h 끝남(`results/n1_umi_v30.json`): 재전송 사이 채널 변화가 3 km/h의 0.5–4.8%에서 30–46%로 커져도 결과가 거의 같음(첫 실패 10.3%, 재전송 17.5 → 13.2, 4회 뒤 실패 2.17 → 1.49%). 273 PRB 전대역이라 작은 규모 페이딩은 주파수로 평균되고, TB의 SINR을 정하는 사용자 이득(22 ms 동안 0.26 dB 변화)과 두 사용자의 상관(0.027 변화)은 그대로이기 때문 → 같은 짝으로 다시 보내는 재전송은 속도와 무관하게 같은 조건을 다시 받음. UMi 60 km/h 끝남(채널 변화 42–50%): 역시 같음(재전송 17.4 → 13.2, 0.76배; 4회 뒤 실패 2.16 → 1.50%; 복구 22.3%). 세 속도는 같은 시드라 첫 전송이 같은 짝지어진 비교 |
| N1b | 재전송은 혼자 보냄(짝 사용자 끔): 같은 짝과 다시 보내는 N1이 HARQ를 불리하게 두는지 확인. 첫 전송은 N1과 같고(같은 시드면 비트까지 같음), 재전송은 1명 설정의 기존 수신기와 사용자 1을 가린 NRx로 받음, 사용자 0의 블록만 셈 | `gen_harq.py --retx-alone`, `exp/retx_alone_smoke.sh` | 끝남(블록 10,240개, `results/n1b_umi_v3_alone.json`): 재전송 15.5 → 11.9(0.77배, 같은 짝일 때 0.75배)로 복구의 재전송 절감은 그대로; 4회 뒤 실패는 1.58 → 1.28%(0.81배, 같은 짝 0.67배)로 줄어드는 폭이 작아짐. 혼자 보내도 1.58%가 남음 |
| P1 | 상관 조건(가시선, 속도, 채널 모델) × 수신 안테나 2/4/8 | `exp/harq_ir.sh` 옵션(4 RX), 안테나 수는 TF 파이프라인 | 가시선 강제(블록 10,240): 첫 실패 16.2%, 복구 18.7%, 재전송 29.3 → 22.1(0.75배), 4회 뒤 4.05 → 2.67%; 비가시선 강제: 첫 실패 4.4%, 복구 35.9%, 재전송 5.8 → 3.6(0.62배), 4회 뒤 0.29 → 0.11%. 안테나 2/4/8(273 PRB, 블록 12,288개씩): 상관 0.6 이상 쌍의 비율이 74/16/4%로 줄어도 완전 CSI+K-best 상한은 기존 수신기 실패의 53/47/51%를 살림 — 안테나가 많아도 수신기로 고칠 실패는 절반 |
| P2 | 사용자 배치(UMi·UMa, 4–32명)에서 무작위 쌍 대 가장 잘 갈리는 쌍 × 복구 유무 | `src/pair_drops.py`, `src/sim/pairing.py` | 첫 계산(2026-10-15 10:24)은 쓸 수 없음: 상관 구간별 표만 써서, 상관이 낮은 블록에 몰린 약한 사용자의 실패가 섞여 '가장 잘 갈리는 쌍'이 실패를 늘리는 결과(10.5 → 17–21%)가 나옴. 다시 함: 사용자 U명(4·8·16)을 떨어뜨리고 사용자 0과 무작위 짝 또는 (1−ρ²)·min(이득)이 가장 큰 짝을 골라 N1처럼 생성·복호(`gen_harq.py --drop-users --pairing`), 큐에 있음 |
| N3 | 채널 묶음(UMi, UMa, 가시선/비가시선, 시속 3–120 km) | `exp/harq_ir.sh` 옵션 | 끝남(블록 10,240개씩): UMi 120 km/h 복구 21.8%·재전송 0.76배; **UMa 3/30/120 km/h 복구 56%·재전송 21.5 → 9.4(0.44배)·4회 뒤 실패 2.27 → 1.00%**; UMa 가시선 28%·0.70배; **UMa 비가시선 73%·0.26배**(4회 뒤 0.98 → 0.22%). 작은 NRx도 UMa에서 첫 실패 13.6 → 7.8%(cuPHY가 지연 확산이 큰 UMa에서 약함). NRx는 UMi로 학습됐으므로 UMa는 학습하지 않은 채널 |
| N5 | 같은 슬롯에서 실패 블록을 다시 풀 수 있는 다른 방법: 기존 수신기 + LDPC 50회, 반복 검출–복호(IDD: MMSE-PIC에 CRC를 통과한 code block을 다시 부호화해 넘김 = 복호된 사용자를 빼는 SIC, 3회), LMMSE 추정 + K-best, 완전한 채널 정보 + K-best(상한), 큰/작은 NRx | `src/classical_rx.py`, `src/idd.py`, `exp/n5_smoke.sh` | 끝남(273 PRB, 슬롯 10,240·실패 2,797; 기준선은 NVlabs LS 기존 수신기로 cuPHY보다 약함): 살린 실패 큰 NRx 42.5%, LMMSE+K-best 37.0%, 작은 NRx 30.4%, IDD 19.4%, LDPC 50회 7.0%, 완전 CSI+K-best 50.3%(상한). 고전 셋을 합쳐도 37.6% < 큰 NRx; K-best만 살리는 블록 0.4%. 상관 0.6 이상에서 큰 NRx 48%, 0.4 미만 35% |
| N6 | OLLA 닫힌 루프 + IR HARQ + 복구 (MCS 10–16이 같은 채널 실현 공유) | `exp/harq_mcs.sh`, `src/sim/olla.py` | 끝남(`results/n6_umi_v3_olla.json`): 복구를 성공으로 치면 외부 루프가 MCS를 1–1.5단계 올려 처리량이 목표 10%에서 SNR 6/8/10 dB +14/+16/+21%, 12/14/16 dB +6/+3/+2%(MCS 16에 막힘); 목표 3%에서 +5…+18%. 작은 NRx를 모든 블록에 쓰면 +0–4%. 사다리는 16QAM(MCS 10–16)뿐(NRx 엔진이 16QAM) |
| N7 | HARQ 결합 방식(기다림·병렬·선택)의 지연 분포 | `src/sim/harq_latency.py` | N1 기록으로 계산(`results/n7_*.json`): **HARQ와 병렬로 복구하면 모든 지표에서 HARQ만 쓰는 것과 같거나 나음**(DDDSU, 선행 4 ms: 10 ms 안에 못 받는 블록 4.40 → 3.19%, 20 ms 2.83 → 2.03%, HARQ로 끝내 못 받는 블록 2.21 → 1.48%, 복구될 블록의 전달 p50 9.4 → 7.7 ms, 재전송 17.5 → 15.5; 비용은 GPU 시간과 이미 grant한 재전송 슬롯). 아래는 복구를 기다리는 방식의 맞교환: 큰 NRx 결과가 L1 1.4 + 6.3 = 7.7 ms에 나오므로, DDDSU에서 선행 시간 2 ms면 첫 재전송 결과(6.9 ms)가 더 빠름. 복구를 기다리면(wait) 복구 안 되는 실패(77%)가 UL 주기 하나만큼 늦어짐(p50 6.9 → 11.9, p99 16.9 → 21.9 ms; 선행 4 ms: 9.4 → 14.4, 24.4 → 29.4) 대신 재전송 17.5 → 13.2; 병렬(parallel)은 지연 그대로·재전송 15.5; 상관 기준 선택은 wait와 같음(후보가 66%); 완벽한 예측은 지연 그대로·재전송 13.2 → 실용 예측 신호로 기존 수신기의 실패 code block 수를 기록하도록 고치고 N1 3 km/h를 다시 돌림(n1_umi_v3_cb, 처음과 같은 결과). **실패 CB 수가 복구 성공을 예측**: TB의 10개 중 1개 실패 98%, 2–5개 84–93%, 6–9개 61–90%, 10개 전부 10%(실패의 82%). 실패 CB ≤ 5개면 복구를 기다리고 나머지는 병렬로 돌리면 지연은 HARQ만 쓸 때와 같고 재전송 17.5 → 14.5 |
| N8 | 셀당 사용자 1–4명: (a) 기존 수신기의 실패와 상관(런타임 cuPHY), (b) 3–4명 복구: 기존 수신기 대 미세 조정 NRx(NVlabs 파이프라인) | `exp/users.sh`, `exp/classical.sh --users` | 끝남: (a) cuPHY 실패가 셀당 사용자 1/2/3/4명에서 15.7/23.5/35.9/53.4%(Eb/No 0–12 dB); (b) 미세 조정 NRx가 3명에서 실패의 45.6%, 4명에서 44.4%를 살림(NVlabs LS 기준) |
| N9 | GPU 수요 계산 | `src/sim/demand.py` | 끝남: Eb/No 8 dB에서 상향 슬롯의 23%가 복구 필요, 셀당 0.58 GPU-초/초(큰 NRx 6.3 ms); GPU 하나에서 NRx 4개를 동시에 돌리면 셀 4개가 그 58% |
| N4 | 4명용 큰 NRx 미세 조정(2명용 가중치에서 시작, 4 PRB 학습, 5만 반복 1단계 + 4만 반복 6단계 = 29만 반복, 1000반복당 159초 측정, XLA 끔), 2명 성능 유지 확인. 8 RX 모델은 처음부터 학습해야 해서 범위 밖 | `exp/n4_train.sh` | 끝남(29만 반복): 3–4명은 N8b; 2명 성능은 떨어짐(살린 실패 46.8 → 37.4%) → 사용자 수별로 원래 모델과 미세 조정 모델을 나눠 씀. **N4b(8 RX)**: 4 RX 가중치를 넓혀(`src/nrx_widen.py`: 안테나 1–4의 입력·출력 가중치를 5–8에 복사, 입력은 반으로) 미세 조정 3.5만 × 8단계, 그 뒤 8 RX에서 LMMSE+K-best와 비교 — queue_b에서 진행 중 |

## 5. 실행

- 큐 둘: `queue/`(노드 1: N1–N3, N6–N9, P1의 4 RX 조건, P2)와 `queue_b/`(노드 2: N4 학습은 GPU 0, 같은 시간에 GPU 1–3이 부수 작업 묶음 `queue_b/side_items`를 학습 단계마다 100분씩 처리 — 작업 하나는 Eb/No 한 점·시드 하나·64슬롯이라 끊겨도 다음 단계가 다시 함; 그 뒤 합치기, N4 평가, N8b, 남은 부수 작업). 각 실행기가 interactive 노드를 스스로 잡고, 4시간 할당이 끝나면 다시 잡는다. `bash queue/watchdog.sh start`가 10분마다 두 실행기를 확인해 죽었으면 다시 띄운다(`queue/watchdog.log`). 멈추려면 그 큐에 `STOP` 파일. 단계는 `todo`, 끝난 단계는 `done`, 기록은 `runner.log`, 단계별 출력은 `logs/`.
- 한 조건: `bash exp/harq_ir.sh NAME CHUNKS --channel UMi --speed 0.83 --ebno 6,8,10,12 --slots 64` (GPU 네 장에서 묶음을 병렬로, 결과 `results/NAME.json`·`.txt`).

## 6. 디렉터리

```
env.sh            컨테이너 경로와 실행 함수(rx: cuPHY·pyAerial·TensorRT, tfgen: TensorFlow·Sionna)
src/              실험 코드; src/vendor는 airan_cloudlab에서 복사한 수신기 코드(SOURCE.md)
exp/              실험 구동 스크립트
queue/            이 프로젝트 전용 큐 실행기
docs/             계획(PLAN.md), 관련 연구(related_work.md)
results/          결과(json, txt; 묶음별 npz는 git에서 제외)
data/             생성한 수신 신호(git에서 제외, 묶음이 끝나면 지움)
third_party/      NVlabs neural_rx 사본(설정 파일을 쓰므로 복사; git에서 제외)
```

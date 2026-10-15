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
| N1·N2 | IR HARQ(RV 0·2·3·1, 최대 4회), 시간에 따라 변하는 3GPP 채널, 조건당 블록 2만 개 | `src/gen_harq.py`, `src/decode_harq.py`, `src/harq_stats.py`, `exp/harq_ir.sh` | UMi 3 km/h 끝남(2026-10-15, `results/n1_umi_v3.json`, 블록 20,480개): 첫 실패 10.3%, 복구가 실패의 22.8%를 재전송 없이 끝냄, 100블록당 재전송 17.5 → 13.2(0.75배; 복구만 0.86배), 4회 뒤 실패 2.21 → 1.48%(0.67배); 작은 NRx를 모든 블록에 쓰면 재전송 16.8. 상관 0.4 미만/0.4–0.6/0.6 이상(블록 35/50/16%)에서 복구 비율 14/27/29%; 같은 상관 구간 안에서는 약한 사용자(채널 이득 하위 25%)와 나머지의 복구 비율이 같음(13.7/14.0, 27.1/27.1, 27.7/29.9%) → 복구 비율은 사용자 이득이 아니라 두 사용자의 상관을 따름. 상관 0.4 미만의 첫 실패가 높은 것은 그 구간에 약한 사용자가 많기 때문(실패의 55%). 30·60 km/h 진행 중 |
| N1b | 재전송은 혼자 보냄(짝 사용자 끔): 같은 짝과 다시 보내는 N1이 HARQ를 불리하게 두는지 확인. 첫 전송은 N1과 같고(같은 시드면 비트까지 같음), 재전송은 1명 설정의 기존 수신기와 사용자 1을 가린 NRx로 받음, 사용자 0의 블록만 셈 | `gen_harq.py --retx-alone`, `exp/retx_alone_smoke.sh` | 큐에 있음(30 km/h 뒤 smoke, 60 km/h 뒤 본 실행) |
| P1 | 상관 조건(가시선, 속도, 채널 모델) × 수신 안테나 2/4/8 | `exp/harq_ir.sh` 옵션(4 RX), 안테나 수는 TF 파이프라인 | 4 RX 조건은 queue. 안테나 2/4/8은 queue_b 부수 작업 묶음(`exp/side_pool.sh`): 기존 수신기와 완전 CSI+K-best, Eb/No 격자를 배열 이득만큼 옮김(2 RX 8–18, 4 RX 4–14, 8 RX 0–10 dB), 시드 16 × 64슬롯 |
| P2 | 사용자 배치(UMi·UMa, 4–32명)에서 무작위 쌍 대 가장 잘 갈리는 쌍 × 복구 유무 | `src/pair_drops.py`, `src/sim/pairing.py` | 큐에 있음 |
| N3 | 채널 묶음(UMi, UMa, 가시선/비가시선, 시속 3–120 km) | `exp/harq_ir.sh` 옵션 | 큐에 있음(smoke 뒤) |
| N5 | 같은 슬롯에서 실패 블록을 다시 풀 수 있는 다른 방법: 기존 수신기 + LDPC 50회, 반복 검출–복호(IDD: MMSE-PIC에 CRC를 통과한 code block을 다시 부호화해 넘김 = 복호된 사용자를 빼는 SIC, 3회), LMMSE 추정 + K-best, 완전한 채널 정보 + K-best(상한), 큰/작은 NRx | `src/classical_rx.py`, `src/idd.py`, `exp/n5_smoke.sh` | queue_b 부수 작업 묶음: smoke(IDD 0회 = LMMSE, 3회·LDPC 50회가 더 나쁘지 않음) 통과 뒤, 시드 40 × 64슬롯 × Eb/No 4점 = 10,240슬롯(실패 블록 약 2천 개) 목표 |
| N6 | OLLA 닫힌 루프 + IR HARQ + 복구 (MCS 10–16이 같은 채널 실현 공유) | `exp/harq_mcs.sh`, `src/sim/olla.py` | 큐에 있음 |
| N7 | HARQ 결합 방식(기다림·병렬·선택)의 지연 분포 | `src/sim/harq_latency.py` | 코드 완료(합성 기록으로 시험), N1 뒤 큐에 있음 |
| N8 | 셀당 사용자 1–4명: (a) 기존 수신기의 실패와 상관(런타임 cuPHY), (b) 3–4명 복구: 기존 수신기 대 미세 조정 NRx(NVlabs 파이프라인) | `exp/users.sh`, `exp/classical.sh --users` | (a) queue, (b) queue_b(N4 뒤) |
| N9 | GPU 수요 계산 | `src/sim/demand.py` | 큐에 있음(N1 뒤) |
| N4 | 4명용 큰 NRx 미세 조정(2명용 가중치에서 시작, 4 PRB 학습, 5만 반복 1단계 + 4만 반복 6단계 = 29만 반복, 1000반복당 159초 측정, XLA 끔), 2명 성능 유지 확인. 8 RX 모델은 처음부터 학습해야 해서 범위 밖 | `exp/n4_train.sh` | 승인됨(2026-10-15), queue_b에서 진행 중 |

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

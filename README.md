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
| N1·N2 | IR HARQ(RV 0·2·3·1, 최대 4회), 시간에 따라 변하는 3GPP 채널, 조건당 블록 2만 개 | `src/gen_harq.py`, `src/decode_harq.py`, `src/harq_stats.py`, `exp/harq_ir.sh` | 코드 완료, smoke 뒤 큐에 있음 |
| P1 | 상관 조건(가시선, 속도, 채널 모델) × 수신 안테나 2/4/8 | `exp/harq_ir.sh` 옵션(4 RX), 안테나 수는 TF 파이프라인 | 4 RX 조건은 큐에 있음 |
| P2 | 페어링 기준 × 복구 유무: 상향 셀 용량과 BLER | 준비 중 | |
| N3 | 채널 묶음(UMi, UMa, 가시선/비가시선, 시속 3–120 km) | `exp/harq_ir.sh` 옵션 | 큐에 있음(smoke 뒤) |
| N5 | 고전 강한 수신기(K-best, LDPC 반복)를 같은 복구 경로에 | 준비 중 | |
| N6 | OLLA 닫힌 루프 + IR HARQ + 복구 (여러 MCS가 같은 채널 실현을 공유) | `gen_harq.py --mcs 10,..,16 --snr-db` | 준비 중 |
| N7 | HARQ 결합 방식(기다림·병렬·선택)의 지연 분포 | `src/sim/harq_latency.py` | 코드 완료(합성 기록으로 시험), N1 뒤 큐에 있음 |
| N8 | 셀당 사용자 1–4명 | 준비 중 | |
| N9 | GPU 수요 모델 | 준비 중 | |
| N4 | 사이트별 미세 조정 (GPU 4–8장 × 수일, 제출 전 승인) | 준비 중 | |

## 5. 실행

- 큐: `bash queue/start.sh`로 실행기를 띄운다(로그인 노드, interactive GPU 노드를 스스로 잡는다). 단계는 `queue/todo`, 끝난 단계는 `queue/done`, 기록은 `queue/runner.log`, 단계별 출력은 `logs/`.
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

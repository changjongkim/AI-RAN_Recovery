# 원고 반영 계획: NRx는 AI-RAN 규모(많은 안테나·많은 사용자)에서 보여진 적이 없다

작성 2026-11-04. 사용자 지시: "원고에도 address가 가능해야 함. 계획만 따로 저장해 두고 나중에 한 번에."
원고를 고치기 전에 할 일:

- AI_RAN_Paper를 pull한다(사용자가 Overleaf에서 고침).
- U3s 결과가 나와 있어야 한다(3절).

## 1. 주장 (쓸 것과 쓰지 않을 것)

쓸 것 — 공개된 연구 기준:

- 공개된 full NRx(채널 추정·검출·LLR을 한 모델로)는 시뮬레이션에서 수신 안테나 16개, 4 layer, 1-symbol DMRS(포트 4개)까지만 보여졌다.
  - 근거: Cammerer GC Wkshps'23, Korpi ICC'21, EqDeepRx, FM-Receiver.
- 2-symbol DMRS(type 1에서 포트 4개 초과), 32·64 안테나, 4 layer 초과는 보여진 적이 없다.
- 실시간 동작은 2명 × 4 안테나뿐이다.
  - Wiesmayr ICMLCN'25: A100에서 반복 약 350 µs에 고정 270 µs가 붙어, 1 ms 안에 반복 2번만 들어간다.
  - 논문 스스로 "error rate is limited by its inference latency"라고 쓴다.
- 공유 GPU 위의 AI-RAN 연구는 full NRx를 돌리지 않는다. 기존 PUSCH 또는 AI 채널 추정만 돌린다.
  - 근거: Eunoia, ARCHES, NVIDIA AI Aerial.
- 사용자(layer)가 늘면 NRx는 더 어렵고 더 무겁다. 논문들이 직접 쓴 내용:
  - Korpi: layer 분리는 "an order of magnitude more demanding".
  - EqDeepRx: "poor scaling with increasing spatial multiplexing order". 4 layer로 학습한 모델은 2 layer에서 되지만, 반대로는 성능이 떨어진다.
  - Goutay JSAC'21: 사용자 수가 바뀌면 full retraining이 필요하다.
  - MDX: NVlabs NRx가 16×4에서 397.6 GFLOPs, 4×2에서 78.6 GFLOPs로 5.1배이다. 단위가 "configured as for the TBLER evaluation"이므로 슬롯당인지 원문에서 확인할 것.

쓰지 않을 것:

- "NRx는 사용자가 많으면 안 된다." 16 안테나·4 layer까지는 된다는 논문이 있고, FM-Receiver는 사용자가 늘수록 이득이 커진다고 한다.
- 우리 2-symbol 8/16 RX 학습 실패(U3n·U3n2). 포트 순서, 시작 가중치, 학습량이 겹쳐 원인을 분리할 수 없으므로 결과로 쓰지 않는다.

Reverb와의 연결:

- 실시간 L1은 모든 TB를 마감 안에서 처리해야 하므로, 큰 NRx를 셀 규모의 안테나·사용자 수로 돌릴 수 없다.
- 복구 경로는 L1 마감 밖에서, CRC가 실패한 TB에만, 설정에 맞는 수신기를 돌린다. 학습된 NRx가 있으면 NRx를, 없으면 LMMSE K-best를 쓴다.
- 복구로 풀린 TB의 비트가 저장된 샘플의 라벨이 된다(Discussion에 이미 있음). 이 라벨은 NRx가 아직 없는 설정의 학습 데이터가 된다.

## 2. 고칠 곳 (원고 파일 기준, 2026-11-04 로컬 5b69d96)

| 위치 | 바꿀 내용 |
|---|---|
| 2.Background.tex, `\subsection{L1 Receivers in AI-RAN Servers}`, 문단 "Conventional and Neural Receivers" | 1절의 범위 문장 한두 개를 넣는다(16 RX·4 layer·1-symbol, 실시간은 2×4). 이어서 "L1은 매크로 셀의 안테나·사용자 수에서 이런 수신기를 모든 TB에 돌릴 수 없다"로 맺는다 |
| 같은 절 Table `tab:bg_receivers` | 열 "Shown (RX × layers)"을 추가한다: DeepRx 2×1, MU-MIMO NRx 16×4 (sim), Real-time·Large NRx 4×2, EqDeepRx 16×4, FM-Receiver 16×4. DeepRx MIMO 16×4 행을 추가한다. 표는 살리고 열만 더한다 |
| 5.Evaluation.tex, `\subsection{Receivers in the L1 and in the Recovery Path}` (또는 새 문단 "Number of Users") | 사용자 수 결과(3절)를 넣는다. 4 RX와 8 RX에서 1–4명일 때, 상한이 사용자 수와 함께 커지고 NRx·LMMSE K-best가 살리는 몫을 보인다. 숫자는 paper_numbers.py 매크로로만. 그림은 eval_rx (c)에 패널이나 막대를 더하는 쪽을 우선한다 |
| 7.Discussion.tex, 문단 "Recovery Receivers of Each Cell" | 큰 설정의 NRx는 설정마다 학습이 필요하다(EqDeepRx, Goutay, 우리 8 RX·4명 미세조정). 복구 경로는 설정마다 수신기를 고른다(NRx가 없으면 K-best: 8 RX·4명에서 기존 실패의 44.6%). 복구가 만든 라벨이 그 설정의 학습 데이터가 된다 |
| 6.RelatedWork.tex, 문단 "Neural Receivers" | "이 연구들은 최대 16 RX·4 layer, 실시간은 2×4를 보였다. 공유 GPU AI-RAN 연구(Eunoia, AI Aerial, ARCHES)는 full NRx를 돌리지 않는다"를 넣는다 |
| 1.Introduction.tex | 선택. 문제 문단에 한 구절 정도 |

## 3. 필요한 데이터 (실험 U3s, `exp/u3s_train.sh`, results/u3s/)

- 4 RX, 1–4명:
  - 이미 있음: NRx `_t` 53.2/60.5/53.4%, LMMSE K-best 34.7/40.9/41.4%, 상한 78/90/97% (2–4명).
  - 1명은 공개 NRx 30.1%, 상한 39%.
- 8 RX 1-symbol, 1–4명: 비교 수신기 결과.

  | 사용자 | 기존 실패 | LMMSE K-best | 상한 | 실행 |
  |---|---|---|---|---|
  | 1 | 29.3% | 42.0% | 58.0% | u3s_uma_r8_u1_f35 |
  | 2 | | | | u3s_uma_r8_u2_f35 (진행 중) |
  | 3 | | | | u3s_uma_r8_u3_f35 (진행 중) |
  | 4 | 19.1% | 44.6% | 89.7% | u1_uma_r8_u4_b1_f35 |

- 8 RX NRx, 1–4명: U3s 학습(11-05 07:00까지) 뒤 가장 좋은 변형으로 평가한다.
  - 같은 시드·Eb/No·슬롯·배치, 시스템은 baseline_lslin_lmmse,nrx.
  - 2026-11-04 03:15 곡선: 변형 c가 40분 학습만에 4명에서 22.7%. 학습 전에는 0%였다.
- 논문 숫자는 paper_numbers.py에 추가하고 scripts/update_numbers.sh로 넣는다.

## 4. 추가할 bib 항목 (원문 확인 뒤 메타데이터 작성)

이미 있음: nvidia-mumimo-nrx (Cammerer), nvidia-rt-nrx (Wiesmayr, ICMLCN 2025), deeprx, eqdeeprx, fm-receiver, eunoia, ai-aerial, ts38211.

새로 추가:

- Korpi et al., "DeepRx MIMO", ICC 2021 (arXiv 2010.16283)
- Goutay et al., JSAC 2021 (arXiv 2012.08177)
- Abdollahpour et al., "MDX", GLOBECOM 2025 (arXiv 2508.12892)
- ARCHES (arXiv 2604.23397): 인용할 때만

DMRS 포트 수(type 1: 1-symbol 4개, 2-symbol 8개 / type 2: 6개, 12개)는 ts38211로 인용한다.

## 5. 원고를 고칠 때의 규칙

- WRITING_RULES.md와 메모리의 문체 규칙을 따른다. 미사여구 금지, Thus 남발 금지, its/them 금지, 쉬운 단어.
- 그림·표는 살린다.
- 고친 뒤 확인한다.
  - `bash scripts/build.sh`
  - `python3 scripts/check_style.py`와 `python3 scripts/paragraph_stats.py`가 모두 0을 내야 한다.
  - build/main.log를 확인한다.
- 커밋은 main.pdf를 포함해 경로를 지정해서 한다. 저자 표기는 넣지 않는다. push는 요청이 있을 때만 한다.
- 문헌 근거와 원문 인용문: 메모리 nrx-scaling-literature.
  - 내려받은 PDF는 세션 scratchpad `papers/`에 있다.
  - 임시 위치이므로 원고를 쓸 때 다시 연다.

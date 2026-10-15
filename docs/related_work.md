# 기존 연구는 실패 블록을 HARQ에 맡겼다

2026-10-14까지 확인한 범위(2005년–2026년 10월)에서, 한 GPU에 Layer-1(L1) 수신기·큰 neural receiver(NRx)·AI/LLM 추론을 함께 두고 CRC가 실패한 상향 TB를 이미 받은 샘플로 HARQ 재전송 grant 전에 다시 복호하는 연구는 없다. 가장 가까운 연구는 세 갈래에 있다. RAN–AI GPU 공유 쪽에서는 Eunoia(arXiv, 2026-10-11)와 Weaver(arXiv, 2026-09)가 슬롯 단위로 SM을 나눈다. 두 연구 모두 NRx가 없고 지키는 것은 L1(또는 LDPC)의 슬롯 마감이다. NRx 쪽의 detect-and-rollback(arXiv, 2026-05)은 NRx와 MMSE를 매 슬롯 함께 돌리고 복호 전에 하나를 고른다. 이 연구는 HARQ를 끈 채 평가했고 CRC 기반 사후 검증을 future work로 남겼다. vRAN 쪽의 Nuberu(MobiCom'21)와 Hades(arXiv'25)는 HARQ 응답을 복호 완료와 떼어 놓지만, 복호 가능성을 예측할 뿐 실패 블록을 다른 수신기로 살리지 않는다. "CRC가 실패하면 더 강한 복호기로 다시 푼다"는 원리 자체는 부호 이론(adaptive SCL, BP→OSD/GRAND)과 특허(US 7254769)에 이미 있다. 따라서 노벨티는 세 가지의 조합으로 좁혀야 한다. 첫째는 한 GPU 위의 L1+NRx+AI이고, 둘째는 채널 추정부터 다시 계산하는 수신기 수준의 재처리이며, 셋째는 재전송 grant 전의 복구 마감과 그 마감에 맞춘 AI 실행 결정이다. 동기가 되는 네 사실은 방향은 맞지만 출처를 고쳐야 한다. 3GPP의 BLER 0.1은 하향 PDSCH의 CQI 정의이고, 상향 약 10%는 운용 관행과 상용망 측정(6.9–9.7%)이 받친다. 3.5 GHz TDD에서는 어느 패턴이든 슬롯의 20%만 상향이다. 4 수신 안테나 MU-MIMO의 분리 어려움은 운용 데이터가 아니라 해석 결과다. NR 상향 HARQ는 비동기라 재전송 결정 시각의 상한이 규격에 없으므로, 11.5 ms는 규격값이 아니라 설계값으로 써야 한다.

## AI-RAN 공유 연구는 L1 마감만 지키고 neural receiver를 싣지 않았다

AI-RAN Alliance는 RAN과 AI의 관계를 세 범주로 나눈다. NVIDIA 저자들이 Alliance 백서(2024-12)를 인용한 정의는 다음과 같다. **AI-for-RAN**은 프로토콜·구조 수준에서 스펙트럼·운용 효율을 높이는 AI(PHY의 channel estimation 등)이다. **AI-and-RAN**은 인프라 수준에서 AI와 RAN이 계산 자원을 나눠 자산 이용률을 높이는 것이다. **AI-on-RAN**은 RAN이 서비스 수준에서 새 AI 응용을 올리는 것이다 ([Kundu 외, arXiv 2501.09007](https://arxiv.org/abs/2501.09007)). 이 논문의 GPU에는 세 범주가 한꺼번에 있다. NRx는 AI-for-RAN이고, L1과 LLM이 GPU를 나누는 것은 AI-and-RAN이며, LLM 추론은 AI-on-RAN이다. 아래 연구들은 이 가운데 하나나 둘만 다룬다. Alliance 백서 PDF 원문은 직접 읽지 못했으므로 정의의 1차 인용은 Kundu 외의 문장으로 하는 편이 안전하다.

산업계 공개 자료에서 RAN과 AI가 GPU를 나누는 방식은 두 가지다. 첫째는 MIG 고정 분할이다. NVIDIA·SoftBank PoC는 GH200 서버의 첫 GPU를 MIG로 나눠 Aerial gNB(4T4R, 100 MHz)에 **최대 40%**, LLM chatbot·digital human에 **60%**를 주고 둘째 GPU는 비워 두었다. 근거로 든 문장은 "RAN-only workload that would peak at 40% GPU usage"이다 ([arXiv 2501.09007](https://arxiv.org/abs/2501.09007)). NVIDIA의 ARC-1 설명도 GB200 NVL2를 MIG로 절반씩 나누는 예(100 MHz 64T64R 셀 약 18개와 12.5K tokens/s)를 든다. 이 수치는 업체 주장이다 ([NVIDIA Developer Blog](https://developer.nvidia.com/blog/bringing-ai-ran-to-a-telco-near-you)). 둘째는 orchestrator의 재배분이다. SoftBank AITRAS Orchestrator는 단말 수·처리량·예측 트래픽을 보고, 사용자가 줄면 RAN 몫을 줄여 AI에 GPU를 넘기고 트래픽이 늘면 되돌린다. 다만 MIG/MPS 같은 수단과 재배분 주기는 공개하지 않았다 ([SoftBank Research](https://www.softbank.jp/en/corp/technology/research/topics/153/)). 세 자료 모두 NRx와 외부 AI를 같은 GPU에서 동시에 잰 공개 수치가 없다. Kundu 외의 게재처는 출처마다 다르다. arXiv 원고는 "submitted to the IEEE"라고 적고, NVIDIA 페이지는 IEEE Network라고 적는다 ([NVIDIA Research](https://research.nvidia.com/publication/2025-01_ai-ran-transforming-ran-ai-driven-computing-infrastructure)). Eunoia 참고문헌은 IEEE Communications Magazine 64(1), 2026이라고 적는다.

학계 연구는 AI에 GPU의 일부를 주는 공간 분할이나 우선순위로 모인다. YinYangRAN(INFOCOM 2024)은 A100에서 MPS로 SM 비율을 DU의 LDPC 복호와 ML 추론에 나눈다. MIG 재분할은 "several seconds", MPS 재구성은 "a few hundred milliseconds"가 걸리므로 결정 간격이 초–분 단위이고, 재구성 중에는 DU가 CPU로 물러난다 ([YinYangRAN pre-print](https://neclab.eu/fileadmin/user_upload/YinYangRAN_Resource_Multiplexing_in_GPU-Accelerated_Virtualized_RANs_pre-print.pdf)). Interplay와 CAORA는 강화학습 agent가 MIG 조각을 RAN과 AI에 배정하는 시뮬레이션이다 ([arXiv 2503.07420](https://arxiv.org/abs/2503.07420), [arXiv 2507.09124](https://arxiv.org/abs/2507.09124)). SMEC(NSDI 2026)은 MPS 위에서 마감 긴급도에 따라 CUDA stream 우선순위를 나눈다. 다만 srsRAN은 다른 서버에 있어 무선 L1과 GPU를 나누지는 않는다. 결과는 SLO 만족 90–96%, tail 지연 최대 122배 감소다 ([arXiv 2601.19162](https://arxiv.org/abs/2601.19162)).

슬롯 단위로 결정하는 연구는 둘이다. Weaver는 OAI의 LDPC 복호만 GPU에 둔다. CUDA Green Context로 슬롯마다 RAN의 SM을 예약하고, FM 학습은 GEMM tile 단위로 남는 SM에 넣는다. 결과는 **HARQ deadline-met 99.8%**, LDPC 최악 약 700 µs(예산 1 ms), 학습 처리량 2.1–3.7배다 ([arXiv 2609.35276](https://arxiv.org/abs/2609.35276)). Eunoia는 GH200에서 **Aerial PHY 전체와 LLM**을 함께 돌린다. libsmctrl의 TPC mask로 2 SM 단위·슬롯 단위로 SM을 나누고, MLP ensemble이 다음 슬롯 SM 사용량의 99.8 백분위를 예측한다. 그 결과 **27셀(DL UE 138, UL UE 61)에서 UL P99 지연을 3 ms 아래로** 지킨다. 관리하지 않으면 UL 지연이 3 ms를 넘고 RAN이 crash한다 ([arXiv 2610.15404](https://arxiv.org/abs/2610.15404)). Eunoia는 평가 규모와 결정 주기 모두에서 이 논문과 가장 가까운 경쟁 연구가 됐다. 그러나 NRx와 CRC 실패는 다루지 않는다. HARQ도 "Uplink decoding … must finish within a few milliseconds; otherwise HARQ retransmissions accumulate"라는 동기 문장에만 나온다.

| 연구 | 게재·연도 | 하는 일 | L1 | NRx | 같은 GPU의 다른 AI | CRC 실패 TB를 HARQ 전 재복호 |
|---|---|---|---|---|---|---|
| NVIDIA·SoftBank AI-RAN PoC (Kundu 외) | arXiv 2025-01, 학술지 표기 엇갈림 | GH200 GPU 하나를 MIG로 RAN ≤40% / AI 60% 고정 분할 | 예 (Aerial, 4T4R 100 MHz) | 아니오 | 예 (LLM) | 아니오 |
| NVIDIA ARC-1·AI-RAN 블로그 | 업체 발표 2024-09/10 | GB200 NVL2를 MIG로 절반씩 나누는 예시 | 예 | 지원한다고만 함, 동시 실행 수치 없음 | 예 (LLM) | 아니오 |
| SoftBank AITRAS + Orchestrator | 2024-11, Orchestrator 2025-07 | 트래픽 예측으로 RAN↔AI GPU 재배분, 수단·주기 비공개 | 예 | 공개 자료에 없음 | 예 | 공개 자료에 없음 |
| [YinYangRAN](https://ieeexplore.ieee.org/document/10621380/) | IEEE INFOCOM 2024 | MPS SM 비율을 DU와 ML에, 결정 초–분, 재구성 중 CPU fallback | 부분 (LDPC) | 아니오 | 예 (TES-RNN 추론) | 아니오 |
| [Interplay](https://arxiv.org/abs/2503.07420) | IEEE INFOCOM 2025 Workshops | SAC가 MIG 조각을 배정, 시뮬레이션 | 모델 | 아니오 | 모델 | 아니오 |
| [CAORA](https://arxiv.org/abs/2507.09124) | IEEE OJCOMS 6 (2025) | 예측 + SAC로 MIG 배정, trace 시뮬레이션 | 모델 | 아니오 | 모델 | 아니오 |
| [Next-Gen AI-on-RAN](https://www.tkn.tu-berlin.de/bib/basaran2025next-gen/) | IEEE ICC 2025 | Aerial L1/L2와 트래픽 예측 학습을 같은 GPU에 (수단 미기재, 학습 시간 32% 감소) | 예 | 아니오 | 예 (학습) | 아니오 |
| [SMEC](https://www.usenix.org/conference/nsdi26/presentation/zhang-xiao) | USENIX NSDI 2026 | MPS + 마감 긴급도별 CUDA stream 우선순위 | 아니오 (srsRAN은 다른 서버) | 아니오 | 예 (YOLOv8, Real-ESRGAN) | 아니오 |
| [SLA-aware Device–RAN–Cloud LLM](https://arxiv.org/abs/2602.23722) | IEEE INFOCOM 2026 Workshops | 5G SA RAN과 LLM 추론을 MIG 고정 분할로 | 예 (범위 미기재) | 초록에 없음 | 예 | 아니오 |
| [Agentic AI-RAN](https://arxiv.org/abs/2601.16565) | arXiv 2026-01 | UAV edge에서 MIG + 컨테이너 | 범위 미기재 | 초록에 없음 | 예 | 아니오 |
| [HAF](https://arxiv.org/abs/2605.07547) | arXiv 2026-05 | LLM agent 배치 + 마감 인식 convex GPU/CPU 할당, 이산 사건 시뮬레이션 | 모델 | 아니오 | 예 (모델) | 아니오 |
| [OCUDU dApp](https://arxiv.org/abs/2609.07843) | arXiv 2026-09 | 상용 DU 안 dApp 시간 등급, "conventional path is never displaced" | 예 | neural equalizer | 초록에 없음 | 초록에 없음 |
| [Weaver](https://arxiv.org/abs/2609.35276) | arXiv 2026-09 | Green Context로 슬롯마다 RAN SM 예약, FM 학습 tile을 남는 SM에 | 부분 (OAI, LDPC만) | 아니오 | 예 (FM 학습) | 아니오 |
| [Eunoia](https://arxiv.org/abs/2610.15404) | arXiv 2026-10-11 | TPC mask로 슬롯 단위 SM 분할, MLP로 다음 슬롯 SM 수요 예측 | 예 (Aerial PHY 전체, 27셀) | 아니오 | 예 (LLM) | 아니오 |

표의 열네 연구 가운데 NRx를 무선 작업에 넣은 것은 OCUDU뿐이다. OCUDU의 neural equalizer도 외부 AI와 함께 돌았다는 기록은 초록에 없다. **L1·NRx·외부 AI를 한 GPU에 둔 연구는 없다.** 이 연구들이 지키는 것은 모두 "현재 슬롯의 L1(또는 LDPC) 마감"이다. AI는 줄어든 몫으로 계속 실행된다. 무선 사건, 예컨대 NRx 복구의 시작이나 마감을 기준으로 AI 실행 여부를 정하는 연구는 찾지 못했다.

셀 사이나 다른 작업과 계산을 나누는 vRAN 연구는 CPU에서 먼저 나왔다. 이 연구들은 모두 복호 실패를 PHY의 끝으로 보고 HARQ 재전송에 맡긴다. 가장 분명한 문장은 Slingshot(SIGCOMM 2023)에 있다. PHY를 옮길 때 HARQ buffer를 버려 생기는 CRC 실패에 대해 "this is no different from a normal PHY operation"이라고 쓴다 ([Slingshot PDF](https://utns.cs.utexas.edu/assets/papers/slingshot-sigcomm23.pdf)). Slingshot은 모든 HARQ 재전송 뒤에도 남는 BLER을 **MBB 0.5–2%, URLLC 0.1%**로 인용한다. 재전송이 손실을 다 흡수하지 못한다는 근거로 쓸 수 있는 수치다.

| 연구 | 게재·연도 | 하는 일 | L1 | NRx | 다른 작업 | CRC 실패 TB 재복호 |
|---|---|---|---|---|---|---|
| [CloudIQ](https://amplab.cs.berkeley.edu/publication/cloudiq-a-framework-for-processing-base-stations-in-a-data-center) | MobiCom 2012 | 기지국 처리를 데이터센터에 모아 계산 자원 최소 22% 절감 | 예 | 아니오 | 아니오 | 아니오 |
| [vrAIn](https://e-archivo.uc3m.es/handle/10016/29093) | MobiCom 2019 | DRL로 CPU 할당과 MCS를 함께 결정 | 예 (CPU, LTE) | 아니오 | 아니오 | 아니오 |
| [Agora](https://par.nsf.gov/biblio/10320275) | CoNEXT 2020 | 서버 한 대 CPU에서 64안테나 massive MIMO baseband 실시간 처리 | 예 (CPU) | 아니오 | 아니오 | 아니오 |
| [Concordia](https://www.microsoft.com/en-us/research/?p=766348) | SIGCOMM 2021 | 최악 실행 시간 예측으로 최소 core만 예약(20 µs 주기), 99.999% 신뢰도에서 유휴 CPU cycle 70% 이상 회수 | 예 (FlexRAN) | 아니오 | 예 (일반 CPU 작업) | 아니오 |
| [Nuberu](https://www.asaavedra.net/files/papers/2021_ggarcia_mobicom_nuberu.pdf) | MobiCom 2021 | 공유 플랫폼 DU, 복호 중간 결과로 decodability 예측(정확도 99%, false positive 약 0.1%) | 예 (CPU) | 아니오 | 공유 플랫폼 | 아니오 (예측) |
| [Hydra](https://www.usenix.org/conference/nsdi23/presentation/gong) | NSDI 2023 | 여러 CPU 서버에 massive MIMO baseband 분산 | 예 | 아니오 | 아니오 | 아니오 |
| [Slingshot](https://www.microsoft.com/en-us/research/publication/slingshot/) | SIGCOMM 2023 | PHY failover·이전, HARQ buffer를 버린 CRC 실패는 상위 재전송에 맡김 | 예 (FlexRAN + ACC100) | 아니오 | 아니오 | 아니오 |
| [CloudRIC](https://dspace.networks.imdea.org/bitstream/handle/20.500.12761/1795/mobicom24-final428_authors_v.pdf) | MobiCom 2024 | 여러 DU가 처리기 pool 공유, TB마다 마감 1–3 ms 안 CPU 처리 가능 여부를 계산해 가속기로 보냄 | 부분 (LDPC) | 아니오 | 아니오 | 아니오 |
| [Hades](https://arxiv.org/pdf/2502.00603) | arXiv 2025-02 | LDPC 앞쪽 반복은 edge(HARQ 예측), 뒤쪽 completion 반복은 원격 cloud | 부분 (srsRAN) | 아니오 | 아니오 | 아니오 (같은 복호기) |
| [MegaStation](https://www.usenix.org/conference/nsdi25/presentation/xie) | NSDI 2025 | GPU executor에 baseband 작업을 동적 명령 스케줄링 | 예 (GPU) | 아니오 | 아니오 | 아니오 |

CloudRIC의 정책 C-DA는 "TB마다 마감 안에 처리할 수 있는지 계산해 처리기를 고른다"는 점에서 TB별 마감 판단과 형태가 비슷하다. 차이는 대상에 있다. CloudRIC은 모든 TB가 거치는 필수 LDPC 복호를 어느 처리기에 둘지 정한다. 이 논문은 실패한 TB에만 생기는 선택적 복구 작업과, 같은 GPU에 있는 외부 AI의 실행 여부를 정한다. 관련 연구 문장에서 이 구분을 밝혀야 한다.

GPU 스케줄링 문헌을 보면 "AI를 작게 쪼개 넣고, 중요한 작업이 오면 멈춘다"는 기법 자체는 새롭지 않다. REEF(OSDI 2022)는 µs 규모의 reset 기반 kernel 선점을 한다 ([OSDI'22](https://www.usenix.org/conference/osdi22/presentation/han)). GPreempt(ATC 2025)는 timeslice 양보로 40 µs 이내에 선점한다 ([ATC'25](https://www.usenix.org/conference/atc25/presentation/fan)). XSched(OSDI 2025)는 선점 가능한 command queue를 둔다 ([OSDI'25](https://www.usenix.org/conference/osdi25/presentation/shen-weihang)). Orion(EuroSys 2024)은 연산자 단위로 간섭을 보고 배치한다 ([GitHub](https://github.com/eth-easl/orion)). libsmctrl/nvtaskset(ECRTS 2025)은 TPC 공간 분할을 한다 ([ECRTS'25](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECRTS.2025.21)). 이 기법들에는 무선 마감도 없고 "다음 실패 블록"이라는 판단 기준도 없다. 그러므로 노벨티는 기법이 아니라 판단 기준과 대상에 두어야 한다. Green Context는 Hopper에서 분할당 최소 8 SM이고 8 SM 단위다 ([arXiv 2601.07600](https://arxiv.org/abs/2601.07600)). Eunoia는 Green Context가 "require modifications to each application code"라며 TPC mask를 택했다. 그래서 Green Context만으로 만든 SM 분할 기준선에는 심사자가 "TPC mask 방식은 어떤가"라고 물을 수 있다.

## Neural receiver는 매 블록 inline에 들도록 작아졌다

기존 NRx는 L1을 없애지 않는다. NRx는 L1 안에서 channel estimation·equalization·demapping을 맡아 LLR을 내고, LDPC 복호와 CRC는 기존 L1이 한다. NVIDIA의 MU-MIMO NRx는 CNN과 GNN으로 슬롯 전체의 추정·등화·demapping을 함께 처리한다. 그 성능은 LMMSE 추정 + K-best 검출과 1 dB 이내다 ([arXiv 2312.02601](https://arxiv.org/abs/2312.02601)). 그리고 이 NRx들은 모든 블록에 대해 LDPC 앞에서 inline으로 돈다. 그래서 **L1의 슬롯 마감이 모델 크기를 정한다.** NVIDIA 실시간 NRx(Wiesmayr 외, IEEE ICMLCN 2025)는 "a strict computational latency budget of 1 ms for the NRX using an NVIDIA A100 GPU"를 가정한다. 132 PRB·2 UE에서 TensorRT 지연은 **반복당 약 350 µs에 고정 270 µs**이므로 1 ms 안에는 CGNN 반복 두 번(N_it=2)만 들어간다. 저자들은 "the achievable TBLER performance of the proposed NRX is limited by its inference latency"라고 쓴다. 실시간 모델의 손실은 큰 모델 대비 0.7 dB 미만이다. 다만 64-QAM에서는 격차가 더 커서 실시간 모델을 QPSK·16-QAM에만 쓴다. 이 논문은 버퍼링을 피하려고 여러 샘플을 병렬로 처리하지 않는다고도 적었다 ([arXiv 2409.02912](https://arxiv.org/abs/2409.02912)). 실제 망에 넣은 사례들은 계산을 더 강하게 줄였다. NI/Ettus의 OAI NRx는 parameter를 700k에서 47k로 줄이고 상향 할당을 6 PRB로 묶었는데, "gain vanishes for larger MCS"였다 ([NI](https://www.ni.com/en/solutions/electronics/5g-6g-wireless-research-prototyping/prototyping-real-time-neural-receiver-usrp-openairint.html)). NVIDIA AI Aerial은 cuPHY PUSCH의 채널 추정만 220k parameter CNN으로 바꾸고 셀마다 MIG를 썼다. 그 결과 MMSE 대비 UL throughput이 40% 이상 늘었다 ([arXiv 2510.01533](https://arxiv.org/abs/2510.01533)).

큰 모델은 마감에 들어가지 않는다는 것이 업계와 학계의 공통 판단이다. Ericsson Technology Review(2026-04)는 mid-band의 L1 실행 예산을 심볼 관련 약 30 µs, 슬롯 관련 약 400 µs로 잡는다. 그리고 GH200과 Xeon에서 잰 결과로 "inference for mildly expressive models (≈1M parameters) largely exceeds the execution-latency budget of most L1-L2 RAN functionalities"라고 쓴다. 추론을 PHY 시간축에서 떼어 내도 얻는 여유는 "typically limited to a few tens or hundreds of microseconds"라고 보고, 결론으로 크게 학습하고 작게 배포하는 distillation을 권한다 ([Ericsson Technology Review](https://www.ericsson.com/497720/assets/local/reports-papers/ericsson-technology-review/docs/2026/when-ai-has-no-time-to-think.pdf)). 큰 모델의 계산 비용은 다음과 같다. MathWorks DeepRx_2M(1.23M parameter)은 SIMO 26 PRB에서도 Quadro RTX 6000에서 슬롯당 **24.10 ms**가 걸린다 ([arXiv 2605.26157](https://arxiv.org/abs/2605.26157)). 38.98M parameter Transformer인 FM-Receiver는 RTX 4090에서 2 user·배치 8 하나에 5.511 ms가 걸린다 ([arXiv 2607.12555](https://arxiv.org/abs/2607.12555)). EqDeepRx는 4배 큰 모델이 약 0.5 dB 더 좋다고 보고한다 ([arXiv 2602.11834](https://arxiv.org/abs/2602.11834)). Wiesmayr 외의 본문 수치로 계산하면 큰 모델(N_it=8)은 132 PRB에서 약 0.27 + 8×0.35 ≈ 3.07 ms다. PRB에 선형이라고 가정하면 273 PRB에서 약 6.3 ms가 되며, 이는 이 논문의 A100 측정값(273 PRB, 2 UE, 약 6.3 ms)과 맞는다. 다만 PRB 선형성은 원 논문이 밝힌 것이 아니므로 논문에는 우리 측정값으로 인용해야 한다. 정리하면, 기존 연구는 정확도와 지연의 맞교환을 "마감 안에 드는 작은 모델 하나"로만 풀었다. 수단은 축소, site-specific 미세조정(NVIDIA "up to 2.2 dB", DeepSig 1.85 dB), distillation 셋이다. **큰 모델을 마감이 더 늦은 별도 경로에 쓰는 방향은 보고되지 않았다.**

NRx와 기존 수신기를 고르는 연구도 HARQ와 CRC를 판단에 쓰지 않는다. Elnashar의 detect-and-rollback(arXiv 2605.26157, 2026-05, 단독 저자)이 가장 가깝다. 이 연구는 DeepRx_2M과 MMSE를 같은 resource grid에 매 슬롯 병렬로 돌리고, 슬롯별 검출기가 LDPC 앞에서 LLR 하나를 고른다. 설정은 SIMO 1×2, 26 PRB, 단일 사용자이고, "Hybrid automatic repeat request is disabled (single transmission with redundancy version zero)"이다. 16개 조건 가운데 NRx는 3개에서 1.0–2.0 dB 이득, 10개에서 동률, QPSK에서 약 2 dB 열세였고, DMRS additional position 2에서는 BLER 100%로 "fails silently"했다. 저자는 CRC를 쓰는 대안을 이렇게 평가한다. "The CRC outcome provides a stronger correctness signal but requires decoding both candidate LLR streams or delaying the decision until after decoding". 최악의 경우 복호 비용이 두 배가 된다는 이유로 "A hybrid design combining pre-decoding rollback with CRC-assisted post-decoding validation is an important extension and is left as future work"라고 남겼다. 한계 절에는 "multi-user interference and hybrid automatic repeat request (HARQ) retransmissions introduce additional failure modes that are not characterized here"라고 썼다 ([arXiv 2605.26157](https://arxiv.org/abs/2605.26157)). ARCHES(arXiv 2604.23397)는 X5G(Aerial+OAI)에서 상향 채널 추정을 AI 추정기와 MMSE 사이에서 슬롯 경계마다 바꾼다(제어 루프 약 140 µs). 결과는 UL PHY throughput 중앙값 +5.32%/+7.23%, GPU 전력 15.8 W 절감이다. 초록에는 HARQ/CRC를 전환 입력으로 쓴다는 말도, 외부 AI도 없다 ([arXiv 2604.23397](https://arxiv.org/abs/2604.23397)). OCUDU와 DeepSig는 NRx를 Class A inline dApp으로 분류하고 "a learned receiver has to produce its output before the decoder needs it"라고 쓴다 ([DeepSig](https://www.deepsig.ai/building-an-open-platform-for-ai-native-ran-with-ocudu/)). HARQ를 다루는 방식은 연구마다 비슷하다. DeepRx는 "completely transparent to the processes of the higher layers"라며 HARQ를 뺐다 ([arXiv 2005.01494](https://arxiv.org/abs/2005.01494)). MathWorks 예제는 HARQ가 LLR 이후에만 영향을 준다며 HARQ를 끈다 ([MathWorks](https://www.mathworks.com/help/5g/ug/ai-native-fully-convolutional-receiver.html)). 결국 **NRx 문헌은 NRx를 HARQ 앞의 조건부 단계로 평가한 적이 없다.**

표준 쪽에서도 AI receiver는 아직 범위 밖이다. Rel-18 TR 38.843의 use case는 CSI feedback, beam management, positioning이다 ([3GPP 38.843](https://3gpp.org/DynaReport/38843.htm)). Rel-19는 beam management, positioning, CSI prediction을 표준화했고 Rel-20은 two-sided CSI compression에 집중한다 ([Lin, arXiv 2506.11828](https://arxiv.org/abs/2506.11828)). 2026-10 개관 논문은 "the underlying AI/ML models themselves are not subject to 3GPP standardization"라고 정리한다 ([arXiv 2610.05694](https://arxiv.org/abs/2610.05694)). 6G Radio 연구(RP-251881)의 설계 원칙에는 "AI/ML-dependent functions should have fallback procedures"가 들어 있다. 이 원칙은 기존 수신기를 항상 돌리고 NRx를 덧붙이는 이 논문의 구조와 맞으므로 인용할 수 있다. gNB 수신기 알고리즘이 구현 영역이라 NRx 복구에 규격 변경이 필요 없다는 판단은 일반 원칙에서 나온 추론이며, 3GPP 문장을 직접 인용한 것은 아니다.

| 연구 | 게재·연도 | 하는 일 | 크기·지연 | L1 | NRx | 같은 GPU의 다른 AI | CRC 실패 후 HARQ 전 재복호 |
|---|---|---|---|---|---|---|---|
| [DeepRx](https://arxiv.org/abs/2005.01494) | IEEE TWC 2021 | SIMO, 주파수 영역 → LLR fully conv, 2 pilot LMMSE 대비 약 2 dB | 1.2M param, 지연 보고 없음 | 시뮬레이션 | 예 | 아니오 | 아니오 (HARQ 뺌) |
| [DeepRx MIMO](https://arxiv.org/abs/2010.16283) | IEEE ICC 2021 | 4 layer, 16 RX, 1 pilot NRx가 2 pilot LMMSE보다 약 2 dB | 보고 없음 | 시뮬레이션 | 예 | 아니오 | 아니오 |
| [HybridDeepRx](https://arxiv.org/abs/2106.16079) | IEEE PIMRC 2021 | 높은 EVM 상향 신호 복조 | 보고 없음 | 시뮬레이션 | 예 | 아니오 | 아니오 |
| [NVIDIA MU-MIMO NRx](https://arxiv.org/abs/2312.02601) | IEEE GLOBECOM Workshops 2023 | CNN+GNN, 1–4 layer, HIL 2 UE·4 RX | 730k weight, 실시간은 future work | HIL | 예 | 아니오 | 아니오 |
| [NVIDIA real-time NRx](https://arxiv.org/abs/2409.02912) | IEEE ICMLCN 2025 | 1 ms 안 inline, MCS 변경에 재학습 없음, 미세조정 최대 2.2 dB | A100: 반복당 350 µs + 270 µs (132 PRB, 2 UE), N_it=2 | 표준 호환 PUSCH 체인 (지연 측정) | 예 | 아니오 | 아니오 |
| [NI/Ettus OAI NRx](https://www.ni.com/en/solutions/electronics/5g-6g-wireless-research-prototyping/prototyping-real-time-neural-receiver-usrp-openairint.html) | 업체 자료 2024-12 | OAI gNB 안에서 기존 수신기와 설정으로 전환 | 700k → 47k param, 6 PRB만 | 예 (OAI) | 예 | 아니오 | 아니오 |
| [Sionna Research Kit](https://arxiv.org/abs/2505.15848) | arXiv 2025-05 | Jetson AGX Orin + OAI에 TensorRT NRx | 수치 없음 | 예 | 예 | 아니오 | 아니오 |
| [Abdollahpour 외](https://arxiv.org/abs/2508.12892) | IEEE GLOBECOM 2025 | model-driven MU-MIMO NRx, FLOPs 66배·parameter 396배 감소 | 지연 수치 없음 | 시뮬레이션 | 예 | 아니오 | 아니오 |
| [NVIDIA AI Aerial](https://arxiv.org/abs/2510.01533) | arXiv 2025-10 | cuPHY PUSCH 채널 추정만 CNN, 셀마다 MIG, UL throughput 40% 이상 | 220k param, 지연 수치 없음 | 예 (Aerial) | 부분 (채널 추정) | 아니오 | 아니오 |
| [EqDeepRx](https://arxiv.org/abs/2602.11834) | arXiv 2026-02 | LMMSE/RZF 위의 작은 NN, 10% BLER에서 약 2 dB | 116k param, 0.37 GFLOPs/16 PRB/layer | 시뮬레이션 | 예 | 아니오 | 아니오 |
| [ARCHES](https://arxiv.org/abs/2604.23397) | arXiv 2026-04 | AI 채널 추정기와 MMSE를 슬롯마다 전환 | 제어 루프 약 140 µs | 예 (Aerial+OAI) | 부분 (채널 추정) | 아니오 | 아니오 |
| [Detect-and-rollback](https://arxiv.org/abs/2605.26157) | arXiv 2026-05 | DeepRx_2M ‖ MMSE 매 슬롯 병렬, LDPC 앞에서 선택, HARQ 끔 | 24.10 ms (RTX 6000), 72.10 ms (CPU) | 시뮬레이션 (MATLAB) | 예 | 아니오 | 아니오 (CRC 사후 검증은 future work) |
| [FM-Receiver](https://arxiv.org/abs/2607.12555) | arXiv 2026-07 | Transformer FM이 복호까지 | 38.98M param, 배치(8) 5.511 ms (RTX 4090) | 시뮬레이션 | 예 | 아니오 | 아니오 |
| [OCUDU](https://arxiv.org/abs/2609.07843) / [DeepSig OmniPHY](https://arxiv.org/abs/2410.19565) | arXiv 2026-09 / 2024 | neural equalizer를 inline dApp으로, 측정 CIR 미세조정 1.85 dB | 호출 예산 < 1 µs (추론 제외) | 예 | 예 | 초록에 없음 | 초록에 없음 |

MU-MIMO에 특화된 지연 데이터는 거의 없다. 사용자·layer 수에 따른 NRx 추론 지연으로 공개된 것은 132 PRB·2 UE의 한 점(반복당 350 µs)뿐이다. EqDeepRx는 layer 수에 "near-linear"라고 주장하지만 FLOPs 기준이다. NVIDIA GNN은 반복마다 모든 UE 상태를 갱신하고 UE 사이 메시지를 합친다 ([arXiv 2409.02912](https://arxiv.org/abs/2409.02912)). 따라서 사용자 수가 늘면 비용도 늘 가능성이 크지만, 이것은 구조에서 나온 추정이다. 이 논문의 사용자 수별 NRx 지연 측정은 그 자체로 새 데이터가 된다.

## 재전송을 피한 기존 방법은 피드백·수신기 수·복호기를 바꿨다

Wi-Fi 시대의 부분 패킷 복구는 재전송을 없애지 않고 작게 만들거나, 다른 수신기의 사본을 빌렸다. MRD(MobiCom 2005)는 여러 radio가 받은 오류 있는 사본을 frame combining으로 합친다. 저자 요약 페이지 기준으로 frame loss를 최대 26% 줄였다 ([CSAIL](https://publications.csail.mit.edu/abstracts/abstracts06/aklmiu/aklmiu.html)). PPR(SIGCOMM 2007)은 PHY의 비트별 신뢰도(SoftPHY hint)로 틀렸을 가능성이 높은 비트만 재전송을 요청한다. 그 결과 중간 부하에서 end-to-end capacity가 약 2배가 됐다 ([PPR PDF](https://courses.cs.washington.edu/courses/csep561/08au/papers/jamieson-sigcomm07.pdf)). SOFT(MobiCom 2007)는 같은 전송을 들은 여러 AP가 Ethernet으로 비트별 신뢰도를 모아 loss rate를 최대 10배 줄인다 ([MIT DSpace](https://dspace.mit.edu/handle/1721.1/37587)). ZipTx(MobiCom 2008)는 깨진 패킷까지 받아 오류정정부호로 고치고, BER이 높은 패킷은 재전송한다. 이득은 실외 저SNR에서 중앙값 2–3배였다. 저자들은 이전 연구의 큰 이득이 rate adaptation 없는 고정 MCS에서 잰 값이라고 지적했다 ([ZipTx PDF](https://www.csd.uoc.gr/~hy439/reading/list_2011/ziptx.pdf)). Maranello(NSDI 2010)는 블록별 체크섬을 NACK에 담아 틀린 블록만 재전송한다 ([USENIX](https://www.usenix.org/conference/nsdi10-0/maranello-practical-partial-packet-recovery-80211)). Chorus(INFOCOM 2010)는 같은 데이터를 담은 broadcast의 충돌을 반복 복호로 푼다 ([PDF](https://kabru.eecs.umich.edu/rtclweb/assets/publications/2010/xyzhang-infocom10-chorus-final.pdf)). "재전송 대신 이미 받은 정보로 복구한다"는 동기는 SOFT·MRD가 이 논문과 가장 가깝다. 그러나 그들은 **공간 다이버시티**(다른 수신기가 받은 신호)를 쓰고, 이 논문은 같은 샘플에 대한 **수신 알고리즘 다이버시티**를 쓴다. ZipTx의 지적은 이 논문에도 해당한다. 그러므로 평가가 닫힌 루프 link adaptation(OLLA)을 켠 상태였는지 밝혀야 한다.

셀룰러 쪽에는 "실패할지 빨리 아는" 연구가 많다. Strodthoff 외(IEEE JSAC 2019)는 ML로 복호 결과를 미리 예측해 HARQ 피드백을 앞당긴다 ([arXiv 1807.10495](https://arxiv.org/pdf/1807.10495)). Rost & Prasad(IEEE WCL 2014)의 opportunistic HARQ에서는 RAP가 복호 오류 확률을 추정해 응답하고 실제 복호는 중앙 처리기가 한다 ([KIT](https://publications.cel.kit.edu/publications/export/RostWCL2014)). Khalili & Simeone은 C-RAN에서 재전송 결정은 RRH/UE에, 복호는 BBU에 둔다 ([arXiv 1508.06570](https://arxiv.org/abs/1508.06570)). Qualcomm 특허 US 12568503(2026-03 등록)에서는 UE가 PDSCH 복호 실패를 예측해 K1 전에 NACK을 보낸다 ([Google Patents](https://patents.google.com/patent/US12568503B2/en)). Murthy(arXiv 2608.06230)의 CNN은 수신 grid에서 CRC 결과를 직접 예측해 조기 link adaptation에 쓴다 ([arXiv 2608.06230](https://arxiv.org/abs/2608.06230)). 이 연구들은 실패를 미리 알아내는 쪽이고, 이 논문은 실패한 블록을 살리는 쪽이다. 두 방향은 서로 보완한다. 예측기가 "곧 실패"라고 판단하면 그 TB를 복구 후보로 일찍 넘길 수 있다.

vRAN의 Nuberu와 Hades는 HARQ 결정 시점과 복호 완료 시점을 떼어 놓는다는 점에서 이 논문의 복구 마감과 가장 닮았다. Nuberu는 MVSF 마감에 worker의 extrinsic 정보로 아직 끝나지 않은 TB의 decodability를 예측한다. 그 이유를 "This allows us to use promising UL TBs that otherwise would have to be dropped because they were not decoded on time"라고 쓰고, 예측 뒤의 늦은 TB는 "now do not have a hard deadline to be processed"라고 쓴다. 설정 {γ1=400, γ2=200}에서 **정확도 99%, false positive 약 0.1%**였다 ([Nuberu PDF](https://www.asaavedra.net/files/papers/2021_ggarcia_mobicom_nuberu.pdf)). Hades는 LDPC의 앞쪽 반복을 edge에서 돌려 HARQ를 예측하고, 뒤쪽 completion 반복은 원격 cloud로 보낼 수 있게 한다. 예측이 불확실하면 마감에 NACK을 보내고, 계산이 모자라면 BLER이 목표(5%)를 넘게 둔 뒤 BLER 기반 link adaptation으로 부하를 줄인다 ([Hades PDF](https://arxiv.org/pdf/2502.00603)). 두 연구와의 차이는 세 가지다. 첫째, 그들의 늦은 복호는 **예측상 성공할 TB를 같은 복호기로 마저 푸는 것**이다. 이 논문은 **이미 CRC가 실패한 TB를 다른 수신기로** 살린다. 둘째, 그들은 오예측 위험을 진다. 이 논문은 실제 CRC 통과를 보고 grant를 정한다. 셋째, 그들의 동기는 계산 부족이고 이 논문의 동기는 신호 품질 부족, 곧 기본 수신기로는 못 푸는 블록이다.

"실패하면 더 강한 단계로 다시 푼다"는 원리 자체는 부호 이론과 특허에 있다. Li·Shen·Tse의 adaptive SC-List decoder는 polar code + CRC에서 CRC를 통과하는 경로가 나올 때까지 list 크기를 키운다 ([arXiv 1208.3091](https://arxiv.org/abs/1208.3091)). BP+OSD는 BP가 실패하면 OSD 후처리로 넘긴다 ([arXiv 2005.07016](https://alphaxiv.org/abs/2005.07016)). LDPC–CRC 두 단계 복호는 짧은 BP가 실패하면 CRC에 GRAND를 적용한다 ([Entropy 27(9), 2025](https://www.mdpi.com/1099-4300/27/9/899)). 특허 US 7254769는 재전송 데이터가 CRC에 실패하면 결과를 송신기에 보내기 전에 두 번째 LDPC 복호기를 거쳐 다시 복호한다. 출원인은 확인하지 못했다 ([USPTO](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7254769)). Ericsson US 11159283은 복호가 덜 끝났으면 "incomplete decoding"을 표시하고 피드백을 다음 기회로 미룬다 ([Google Patents](https://patents.google.com/patent/US11159283B2/en)). 고전 수신기에는 CRC가 실패하면 복호기의 soft 출력을 MIMO 검출기로 되돌리는 반복 수신기 특허(US 8621319)가 있다 ([USPTO](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8621319)). one-bit ADC MU-MIMO에서 CRC를 통과한 사용자로 나머지를 다시 검출하는 방식도 있다 ([IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/8885762)). 이들은 같은 LLR 위에서 **복호기만** 바꾸거나 같은 수신기 안에서 반복한다. 채널 추정·등화·demapping은 다시 하지 않는데, NRx의 이득은 바로 그 앞단에서 나온다. 또 이들은 HARQ grant 마감이나 GPU 공유와 무관하다. DUIDD(ACSSC 2022) 같은 검출–복호 반복은 모든 블록의 기본 경로에서 돌아가므로 CRC로 켜지는 조건부 처리가 아니다 ([arXiv 2212.07816](https://ar5iv.labs.arxiv.org/html/2212.07816)). 다만 "두 번째 패스가 더 좋은 LLR을 만든다"는 점은 같으므로 심사자가 물을 수 있다.

| 연구 | 게재·연도 | 하는 일 | 재전송과의 관계 | CRC 실패 블록 재복호 | NRx / 같은 GPU의 AI |
|---|---|---|---|---|---|
| MRD | MobiCom 2005 | 여러 radio의 사본을 frame combining | 실패 시 RFA 재전송 | 아니오 (다른 수신기의 사본) | 아니오 / 아니오 |
| PPR | SIGCOMM 2007 | SoftPHY hint, 틀렸을 비트만 PP-ARQ | 재전송을 작게, 피드백 형식 변경 | 아니오 | 아니오 / 아니오 |
| SOFT | MobiCom 2007 | 여러 AP가 비트별 신뢰도를 합침 | 다중 수신기로 회피 | 아니오 (공간 다이버시티) | 아니오 / 아니오 |
| ZipTx | MobiCom 2008 | 깨진 패킷을 송신 쪽 FEC로 정정 | 부호 추가, 고BER은 재전송 | 아니오 | 아니오 / 아니오 |
| Maranello | NSDI 2010 | 블록 체크섬 NACK, 틀린 블록만 재전송 | 재전송을 작게 | 아니오 | 아니오 / 아니오 |
| Chorus | INFOCOM 2010 | 같은 데이터의 충돌을 반복 복호 | 여러 송신 전제 | 아니오 (충돌 복구) | 아니오 / 아니오 |
| Adaptive SC-List | arXiv 2012 | CRC 통과까지 list 크기를 키움 | 해당 없음 | 부분 (복호기 수준) | 아니오 / 아니오 |
| Rost & Prasad | IEEE WCL 2014 | RAP가 복호 오류 확률 추정, 복호는 중앙 | 예측으로 HARQ 결정 | 아니오 | 아니오 / 아니오 |
| Khalili & Simeone | IEEE TVT 2017 | 재전송 결정 RRH/UE, 복호 BBU | 예측 | 아니오 | 아니오 / 아니오 |
| Strodthoff 외 E-HARQ | IEEE JSAC 2019 | ML로 복호 결과 조기 예측 | 피드백을 앞당김 | 아니오 | 예측기만 / 아니오 |
| Nuberu | MobiCom 2021 | decodability 예측, 늦은 TB는 마감 없이 마저 복호 | 예측 HARQ | 아니오 (같은 복호기) | 아니오 / 아니오 |
| DUIDD | ACSSC 2022 | 검출–복호 반복을 deep unfolding | 모든 블록의 기본 경로 | 아니오 (조건부 아님) | 학습형 / 아니오 |
| Hades | arXiv 2025 | early/completion 반복 분리, 불확실하면 NACK | 예측 HARQ | 아니오 (같은 복호기) | 아니오 / 아니오 |
| BP→OSD, LDPC–CRC GRAND | arXiv 2020, Entropy 2025 | BP 실패 시 OSD 또는 CRC-GRAND | 해당 없음 | 부분 (복호기 수준) | 아니오 / 아니오 |
| US 7254769 | 특허 | CRC 실패 시 두 번째 LDPC 복호기로 재복호 후 보고 | 피드백 전 재복호 | 부분 (복호기 수준) | 아니오 / 아니오 |
| Ericsson US 11159283 | 특허 (우선일 2017) | 복호 미완료 표시, 실패 확정 CBG만 NACK, 피드백 연기 | 피드백 연기 | 아니오 | 아니오 / 아니오 |
| [Semantic HARQ](https://arxiv.org/html/2504.14615v1) | arXiv 2025 | NRx가 현재 신호와 이전 실패 신호를 결합 | 재전송 전제 | 아니오 | 예 / 아니오 |
| Detect-and-rollback | arXiv 2026-05 | NRx ‖ MMSE, 복호 전 선택 | HARQ 끔 | 아니오 (future work) | 예 / 아니오 |

이름이 비슷한 NVIDIA Aerial의 "early-HARQ"는 다른 기능이다. PUSCH에 실린 UCI의 HARQ-ACK 비트를 먼저 L2에 올리는 기능으로, TB 재복호와 관계없다 ([Aerial limitations](https://docs.nvidia.com/aerial/cuda-accelerated-ran/25-2/aerial_cubb/release_notes/limitations.html)). 논문에서 혼동이 생기지 않게 해야 한다.

## 동기의 네 사실은 맞지만 출처를 갈라 써야 한다

### 상향 BLER 10%는 규격이 아니라 운용점이다

규격의 0.1은 하향링크 정의다. TS 38.214 §5.2.2.1(Channel quality indicator)에서 UE는 "a single PDSCH transport block … could be received with a transport block error probability not exceeding: 0.1"을 만족하는 가장 높은 CQI를 보고한다. 0.00001은 URLLC용 table3에 쓴다 ([TS 38.214 V17.6.0](https://www.etsi.org/deliver/etsi_ts/138200_138299/138214/17.06.00_60/ts_138214v170600p.pdf)). PUSCH의 MCS 표 선택(§6.1.4.1)에는 **BLER 목표 문구가 없다.** 상향 약 10%의 근거는 운용 관행과 측정이다. Intel의 상향 link adaptation 논문은 MCS를 "an existing MI2MCS table calibrated to yield around 10% BLER"로 정하고, 시뮬레이션 표에 "Target BLER 10%"를 둔다 ([Sun·Moon·Fwu, IEEE WCL 2020](https://arxiv.org/pdf/2009.08494)). 측정 근거로는 SoftBank 상용 5G SA 현장 시험이 있다. 64QAM 표에서 상향 BLER은 **n3 6.92%, n28 8.62%**였고, UL-256QAM을 켜면 **8.27%, 9.72%**로 올랐다 ([arXiv 2511.19868](https://arxiv.org/pdf/2511.19868)). 이 측정은 UE 쪽 진단 로그에서 얻었고, 첫 전송 BLER인지 전체 BLER인지는 정의를 확인하지 못했다. 학회 측정 연구로 이 수치를 대신할 수는 없다. SIGCOMM'21 "A Variegated Look at 5G in the Wild"에는 BLER·HARQ 언급이 없고 ([PDF](https://www-users.cse.umn.edu/~fengqian/paper/5g_sigcomm21.pdf)), SIGCOMM'24 중대역 연구는 하향 위주다. 목표값도 구현마다 다르다. Hades는 목표를 5%로 둔다. 그래서 "보통 10% 근처"라고 쓰는 것이 정확하다.

### 3.5 GHz TDD에서 상향 슬롯은 20%다

GSMA는 5G끼리 공존할 때 DDDSU를, TD-LTE와 공존할 때 DDDDDDDSUU 또는 DDDSUUDDDD를 권고한다 ([GSMA](https://www.gsma.com/spectrum/wp-content/uploads/2021/03/5G-TDD-Synchronisation-QA.pdf)). SIGCOMM'24 중대역 측정에 따르면 "only four operators in Europe use (1) [DDDSU], while all others use either (2) or (3)"이다 ([arXiv 2310.11000](https://arxiv.org/pdf/2310.11000)). BLER=0일 때 PHY 사용자 평면 지연은 Vodafone Italy(DDDDDDDSUU) 6.93 ms, Vodafone Germany(DDDSU) 2.13 ms로 TDD 구조를 따른다 ([SIGCOMM'24](https://par.nsf.gov/servlets/purl/10536692)). SoftBank·AIS의 n77/n41 설정도 5 ms에 상향 슬롯 두 개인 DDDSUUDDDD형이다 ([arXiv 2511.19870](https://arxiv.org/pdf/2511.19870)). 어느 패턴이든 **상향 전용 슬롯은 20%**이고, 특수 슬롯의 UL 심볼을 넣어도 약 22.9%다. 차이는 기회의 간격이다. DDDSU는 2.5 ms마다 한 슬롯이고, 5 ms 패턴은 5 ms마다 연속 두 슬롯이다. 따라서 재전송 한 번의 최소 지연 비용은 DDDSU에서 2.5 ms, 5 ms 패턴에서 최대 5 ms다. 단말 출력도 제약이다. power class 3은 모든 FR1 대역에서 23 dBm이고, n41/n77/n78/n79에만 PC2 26 dBm과 PC1.5 29 dBm이 있다 ([TS 38.101-1 §6.2.1](https://www.etsi.org/deliver/etsi_ts/138100_138199/13810101/17.10.00_60/ts_13810101v171000p.pdf)). Ericsson은 "UL performance is coverage-limited because smartphones have limited output power (typically 200mW)"라고 쓴다 ([Ericsson Technology Review](https://www.ericsson.com/en/reports-and-papers/ericsson-technology-review/articles/achieving-networks-with-high-performing-uplinks)). SIGCOMM'24 측정에서도 상향 PHY throughput은 좋은 채널에서조차 평균 100 Mbps에 크게 못 미쳤다.

### 4 수신 안테나에서 두 사용자 상관 평균은 0.46이다

수신 안테나가 N개인 i.i.d. Rayleigh 채널에서 두 사용자 채널의 정규화 상관 ρ=|h₁ᴴh₂|/(‖h₁‖‖h₂‖)는 **ρ² ~ Beta(1, N−1)**을 따른다. 등방 단위 벡터에 대한 이 결과는 Jindal(IEEE Trans. IT 2006)에 있다 ([arXiv cs/0603065](https://arxiv.org/pdf/cs/0603065)). 두 사용자 zero-forcing에서 각 사용자의 후처리 SNR은 1−ρ²배로 줄어든다. 아래 값은 해석식에서 나왔고 몬테카를로 40만 쌍과 일치한다.

| 수신 안테나 N | E[ρ] | 중앙값 | P(ρ>0.7) | ZF에서 3 dB 넘게 잃을 확률 P(ρ²>0.5) |
|---|---|---|---|---|
| 2 | 0.667 | 0.707 | 51.0% | 50.0% |
| 4 | **0.457** | 0.454 | **13.3%** | **12.5%** |
| 8 | 0.318 | 0.307 | 0.90% | 0.78% |
| 16 | 0.223 | 0.213 | ≈0 | ≈0 |
| 64 | 0.111 | 0.105 | ≈0 | ≈0 |

이 근거는 해석 결과이고 운용 데이터가 아니다. 상용 상향 MU-MIMO의 공개 사례는 대규모 MIMO 사이트에서 나왔다. Verizon n77(Samsung vRAN)에서는 셀이 상향 4 layer MU-MIMO를 지원했고, 스마트폰 4대가 약 300 Mbps를 내 SU-MIMO 86 Mbps의 **3.3배**였다. 위치가 나쁜 폰은 페어링되지 않았다 ([SRG/Samsung 2024](https://images.samsung.com/is/content/samsung/assets/global/business/networks/insights/white-paper/5g-multi-user-mimo-it-isnt-just-for-downlink-anymore/srg_samsung_mu_mimo_study_final.pdf)). 수신 안테나가 적은 사이트도 흔하다. SoftBank n77(3.4 GHz)은 **4T4R 16.7%, 8T8R 78.0%, 대규모 MIMO 5.2%**이고, n3·n28은 2T2R이 95% 안팎이다. 수신 안테나가 줄면 같은 MCS에 필요한 SNR이 오른다(UL-256QAM MCS 24에서 2T2R·4T4R·8T8R 각각 21.29, 17.23, 13.87 dB) ([arXiv 2511.19870](https://arxiv.org/pdf/2511.19870)). 상용망에서 상향 MU-MIMO가 쓰이는 비율이나, 페어링된 TB의 BLER을 보고한 공개 측정은 없다. 스케줄러는 상관이 큰 쌍을 피한다. 그러므로 i.i.d. 무작위 쌍은 실제보다 비관적일 수 있다. 반대로 가시선이나 좁은 각도 분산에서는 상관이 i.i.d.보다 크다.

### NR 상향 HARQ는 재전송 결정 시각에 상한을 두지 않는다

아래 조항은 ETSI의 Rel-17 사본에서 본문을 직접 확인했다. NR은 LTE의 고정 PHICH 시점을 없앴고, gNB가 재전송 grant의 시점을 고른다. 규격이 두는 것은 하한뿐이다. 30 kHz(µ=1)에서 capability 1의 Tproc,2 ≈ (12+1)×35.7 µs ≈ 0.46 ms이고 capability 2는 약 0.23 ms다. K2 상한 32 슬롯은 16 ms다. 같은 HARQ process는 이전 PUSCH가 끝나기 전에 다시 스케줄할 수 없다. 상한은 DRX를 쓸 때의 drx-RetransmissionTimerUL뿐이다(최대 sl320, 30 kHz에서 160 ms). 결정을 늦추면 재전송이 다음 상향 슬롯으로 밀리고, 그 대가는 지연과 HARQ process 점유다. DDDSU에서 한 UE가 매 상향 슬롯에 스케줄되고 각 TB가 12–15 ms 미결이라면 점유되는 process는 5–6개로, 기본값 16개보다 적다. 이 수치는 측정이 아니라 추론이다.

| 사실 | 조항 | 내용 |
|---|---|---|
| 비동기 IR HARQ, gNB가 재전송을 DCI로 스케줄 | [TS 38.300 §5.3.5.4](https://www.etsi.org/deliver/etsi_ts/138300_138399/138300/17.06.00_60/ts_138300v170600p.pdf) | "Asynchronous Incremental Redundancy Hybrid ARQ is supported. The gNB schedules each uplink transmission and retransmission using the uplink grant on DCI." |
| NDI 토글이면 새 전송, 아니면 재전송; process당 TB 하나 | [TS 38.321 §5.4.2.1](https://www.etsi.org/deliver/etsi_ts/138300_138399/138321/17.06.00_60/ts_138321v170600p.pdf) | 서빙 셀마다 HARQ entity 하나, "Each HARQ process supports one TB" |
| 상향 HARQ process 수, 같은 process 재스케줄 하한 | [TS 38.214 §6.1](https://www.etsi.org/deliver/etsi_ts/138200_138299/138214/17.06.00_60/ts_138214v170600p.pdf) | 기본 16개, UE capability로 최대 32개; 이전 PUSCH가 끝나기 전 같은 process 재스케줄 불가 |
| DCI–PUSCH 간격 K2 | TS 38.214 §6.1.2.1, [TS 38.331](https://www.etsi.org/deliver/etsi_ts/138300_138399/138331/17.06.00_60/ts_138331v170600p.pdf) | `k2 INTEGER(0..32)`, 없으면 15/30 kHz에서 1 |
| 최소 PUSCH 준비 시간 | TS 38.214 §6.4 | Tproc,2, N2 = 10/12/23/36 심볼 (capability 1, µ=0–3) |
| DCI 필드 | [TS 38.212 §7.3.1.1.1–2](https://www.etsi.org/deliver/etsi_ts/138200_138299/138212/17.05.00_60/ts_138212v170500p.pdf) | NDI 1 bit, RV 2 bits, HARQ process number 4 bits (32 process면 5 bits) |
| DRX를 쓸 때의 상한 | TS 38.321 §5.7, TS 38.331 DRX-Config | drx-RetransmissionTimerUL: "the maximum duration until a grant for UL retransmission is received" |
| LTE 동기 HARQ(대비) | [TS 36.213 §8.0, §9.1.2](https://www.etsi.org/deliver/etsi_ts/136200_136299/136213/17.05.00_60/ts_136213v170500p.pdf) | FDD에서 PUSCH n → PHICH n+4 → 재전송 n+8, process 8개 |

## 차이는 조합에 있고 주장은 범위를 밝혀 좁혀야 한다

### 구조와 복구 경로가 갈리는 여덟 축

이 논문(Antiphase)의 구조는 다음과 같다. 한 GPU에서 L1 기존 수신기가 모든 블록을 슬롯 마감 안에 복호하고, CRC가 실패한 TB만 큰 NRx가 이미 받은 샘플로 다시 처리한다. 복구 마감 안에 CRC를 통과하면 재전송 grant를 내지 않는다. AI/LLM 추론은 다음 실패 블록의 복구 마감을 기준으로 실행 여부가 정해진다. 아래 표는 이 구조가 각 축에서 기존 연구와 어떻게 다른지 보인다.

| 축 | 기존 연구 | 이 논문 |
|---|---|---|
| 한 GPU의 작업 | L1(또는 LDPC)+AI (PoC, YinYangRAN, Weaver, Eunoia) 또는 L1+NRx (NVIDIA 실시간 NRx, ARCHES, OCUDU) | L1 + 큰 NRx + AI/LLM |
| NRx를 부르는 조건 | 모든 블록에 LDPC 앞 inline (NVIDIA, DeepRx, AI Aerial), 또는 매 슬롯 MMSE와 병렬 (detect-and-rollback) | CRC가 실패한 TB만 |
| NRx 크기와 마감 | L1 슬롯 마감(A100에서 1 ms)에 맞춰 축소 (N_it=2, 47k param, 6 PRB) | 큰 모델(A100, 273 PRB, 2 UE에서 약 6.3 ms 측정), 복구 마감 11.5 ms(설계값) |
| 두 번째 처리의 신호 | 없음, 복호 전 LLR 통계 (detect-and-rollback), 예측한 decodability (Nuberu, Hades, E-HARQ) | 실제 CRC 결과 |
| 다시 계산하는 범위 | 같은 LLR 위 복호기만 (adaptive SCL, BP→OSD/GRAND, US 7254769), 같은 복호기의 남은 반복 (Nuberu, Hades), 다른 수신기의 사본 (SOFT, MRD) | 같은 샘플에서 채널 추정·등화·demapping부터 다른 수신기로 |
| HARQ와의 관계 | 끄거나 다루지 않음 (NRx 연구), 예측으로 응답 (vRAN), 재전송을 작게 (PPR, Maranello) | HARQ 앞의 단계: 성공하면 재전송 grant를 내지 않고, 실패하거나 마감을 넘기면 표준 HARQ로 |
| AI 실행 결정 | 공간 몫(MIG, MPS, Green Context, TPC mask) 또는 stream 우선순위, 기준은 L1 수요·트래픽 | 다음 실패 TB의 복구 마감을 기준으로 AI 실행 여부를 정함 |
| 공중 인터페이스 | PPR·Maranello는 피드백 형식 변경, ZipTx는 송신 쪽 부호 추가 | 변경 없음 (비동기 HARQ 안에서 grant 시점만 조절) |

Detect-and-rollback이 CRC 경로를 뺀 이유는 "모든 슬롯에서 복호 비용이 두 배"라는 것이었다. 이 논문에서는 CRC가 실패한 TB에만 두 번째 처리가 생기므로 평균 추가 비용이 실패율(약 10%)에 비례한다. 이 대비는 related work 문장으로 바로 쓸 수 있다. Ericsson이 본 "PHY 시간축과 분리할 때의 여유"는 수십–수백 µs다. 반면 이 논문의 창은 HARQ 재전송을 기다리는 수 ms이고, 이는 그 글이 다루지 않은 영역이다. 이 차이도 인용할 때 분명히 해야 한다.

### 써도 되는 주장

모든 문장에 "to our knowledge"와 확인 범위(이 보고서의 문헌, 2026-10 기준)를 함께 적는다. 영문은 "existing work"로 쓴다.

| 주장 (영문 제안) | 근거 |
|---|---|
| To our knowledge, no existing AI-RAN system runs the L1 receiver, a neural receiver, and AI inference on the same GPU. | 첫 표의 열네 연구와 NRx 연구 표 (NRx가 있으면 외부 AI가 없고, 외부 AI가 있으면 NRx가 없음) |
| Existing RAN–AI GPU sharing gives AI a spatial share (MIG, MPS, SM or TPC masks) or a stream priority and protects the L1 deadline of the current slot; none decides AI execution by the deadline of recovering a failed block. | PoC, YinYangRAN, Weaver, Eunoia, SMEC |
| Existing neural receivers run inline on every block before LDPC decoding and are reduced to meet the L1 deadline; NVIDIA's real-time receiver fits only two iterations in 1 ms on an A100, and its authors note that its error rate is limited by inference latency. | Wiesmayr 외, NI/OAI, AI Aerial, Ericsson |
| The closest neural-receiver work selects between neural and conventional LLRs before decoding, disables HARQ, and leaves CRC-assisted post-decoding validation to future work. | Elnashar (원문 인용 가능) |
| Existing vRAN systems decouple HARQ feedback from decoding by predicting decodability; they finish blocks predicted to succeed with the same decoder and do not re-decode failed blocks with a stronger receiver. | Nuberu, Hades |
| Existing packet recovery without retransmission combines copies from multiple receivers or changes the feedback protocol; we re-process the same samples with a different receiver at one gNB, without changing the air interface. | SOFT, MRD, PPR, Maranello, ZipTx |
| NR uplink HARQ is asynchronous and adaptive: the gNB schedules each retransmission with an uplink DCI, and the specification bounds the earliest time (K2, PUSCH preparation time) but not the latest, except through the DRX retransmission timer when DRX is configured. | TS 38.300 §5.3.5.4, TS 38.321 §5.4.2.1, TS 38.214 §6.1/§6.1.2.1/§6.4, TS 38.321 §5.7 |
| Uplink link adaptation commonly targets about 10% BLER, and commercial measurements report 7–10% uplink BLER. | Sun 외 2020, arXiv 2511.19868 |
| In 3.5 GHz TDD, 20% of slots are uplink (DDDSU: one every 2.5 ms at 30 kHz; DDDDDDDSUU/DDDSUUDDDD: two every 5 ms), and a power-class-3 UE transmits at most 23 dBm. | GSMA, Fezeu 외, TS 38.101-1 |
| For i.i.d. Rayleigh channels with four receive antennas, the normalized correlation of two users has mean 0.46, and 12.5% of random pairs lose more than 3 dB under two-user zero-forcing (analysis). | Jindal |

### 고쳐야 할 주장

| 쓰기 쉬운 문장 | 문제 | 고친 문장·조치 |
|---|---|---|
| "TS 38.214 §5.2.2.1에 따라 상향은 BLER 10%를 목표로 한다" | §5.2.2.1은 PDSCH CQI 정의이고, PUSCH에는 BLER 목표 문구가 없음 | "The CQI definition uses 0.1 for PDSCH; uplink outer-loop targets are operator-set and commonly about 10%"에 Intel 논문과 SoftBank 측정을 붙임 |
| "측정 연구가 상향 BLER 10%를 보였다(SIGCOMM)" | SIGCOMM'21은 BLER이 없고 SIGCOMM'24는 하향 위주 | SoftBank 현장 시험(arXiv 2511.19868)을 인용하고, UE 로그 기반이며 첫 전송 여부가 불명확함을 밝힘 |
| "DDDSU가 대표 패턴" | 유럽 운용사 대다수는 5 ms 패턴 | 두 패턴을 함께 쓰고 "어느 쪽이든 상향 20%"로 씀 |
| "수신 안테나 4개의 상향 MU-MIMO는 흔하다" | 공개된 상용 상향 MU-MIMO는 대규모 MIMO 사례이고, 4R 근거는 해석 결과 | 설정을 4T4R/8T8R 사이트나 소형 AI-RAN 셀로 한정하고, 해석 결과임과 스케줄러 페어링이라는 단서를 밝힘 |
| "HARQ 마감은 11.5 ms" | 규격에는 재전송 결정 시각의 상한이 없음 | "a recovery deadline we choose within the freedom of asynchronous HARQ"로 쓰고, 재전송이 실리는 상향 슬롯과 MAC 선행 시간을 함께 설명 |
| 기준선 이름 "Static (SoftBank AI-RAN)" | AITRAS Orchestrator는 트래픽 기반 동적 재배분을 공개했음(2025-07) | "fixed split as in the NVIDIA–SoftBank PoC (MIG 40/60)"처럼 출처를 좁히고, 우리 구현의 분할 수단과 PoC의 수단이 같은지 맞춰 적음 |
| "슬롯 단위 결정은 Weaver뿐", "외부 AI와 GPU를 나누는 연구 가운데 평가 규모가 가장 크다" | Eunoia(2026-10-11)가 Aerial PHY 전체, 27셀, 슬롯 단위 SM 분할을 보였음 | Eunoia를 관련 연구와 비교표에 넣고 두 문장을 고침 |
| 기준선 이름 "Slice (Weaver)" | Eunoia도 SM 분할(TPC mask와 예측기)을 씀 | Eunoia 병기를 검토하고, TPC mask 방식에 대한 질문에 답을 준비함 |
| "SMEC: MPS 낮은 우선순위" | 원문은 MPS 위 마감 긴급도별 CUDA stream 우선순위이고, RAN은 다른 서버에 있음 | 원문대로 고침 |
| "Interplay·CAORA의 MIG 동적 분할" | 조각을 다시 배정하는 시뮬레이션이고, MIG 재분할에는 수 초가 걸림(YinYangRAN) | "reassigning MIG slices (simulation)"으로 고침 |
| "AI를 작게 나눠 실행하고 멈추는 것이 새롭다" | REEF, GPreempt, XSched, LithOS에 이미 있음 | 노벨티를 판단 기준(실패 TB의 복구 마감)과 대상(L1+NRx+AI)으로 옮김 |
| "CRC 실패 시 더 강한 복호를 하는 첫 연구" | adaptive SCL, BP→OSD/GRAND, US 7254769가 있음 | "receiver-level re-processing of CRC-failed uplink TBs before the retransmission grant on a GPU shared with AI"로 좁히고 "to our knowledge"를 붙임 |
| "작은 NRx가 L1을 대신한다" | NRx는 L1 안의 추정·등화·demapping만 맡고, LDPC·CRC는 L1이 함 | "NRx inside L1 on every block"으로 고침 |
| "큰 NRx 6.3 ms는 NVIDIA 수치" | 3.07 ms(132 PRB)는 본문 수치로 계산한 값이고, 273 PRB 값은 선형 가정에서 나옴 | 우리 측정값으로 인용 |
| 게재처 | 표기 오류나 미갱신 | CAORA는 IEEE OJCOMS 6 (2025), arXiv 2602.23722는 INFOCOM 2026 Workshops, SOFT는 MobiCom 2007(2005·SIGCOMM 아님), CloudIQ는 MobiCom 2012, NVIDIA 실시간 NRx는 IEEE ICMLCN 2025; Kundu 외는 IEEE Xplore에서 확인 필요 |
| "Aerial early-HARQ"를 관련 기능으로 언급 | UCI HARQ-ACK 조기 보고 기능임 | 언급하지 않거나 무관함을 밝힘 |

### 확인되지 않은 것

가장 큰 공백은 특허와 장비 업체다. "저장한 샘플을 고급 또는 신경망 수신기로 다시 처리한 뒤 HARQ NACK을 정한다"는 청구항이 있는지 Google Patents·USPTO 전문 검색(CPC H04L 1/18과 G06N 3/08 등)으로 체계적으로 찾지 않았다. Ericsson·Nokia·Huawei·Samsung 기지국 소프트웨어가 내부적으로 CRC 실패 시 고급 수신기를 다시 돌리는지도 공개 자료로는 알 수 없다. 이 확인 전에는 "복구 경로 자체가 새롭다"를 주장의 중심에 두지 않는 편이 안전하다. 경쟁 연구 쪽에서 확인하지 못한 것은 다음과 같다. Eunoia의 게재처(원고에 ACM 템플릿 자리표시만 있음)와 후속 버전은 모른다. Weaver는 원문 뒷부분의 평가 세부를 읽지 못했다. Hades는 게재처와 정량 결과를, Nuberu는 오예측 처리 방식을 확인하지 못했다. AITRAS Orchestrator의 분할 수단과 주기도 공개되지 않았다. Aerial L1이 MPS 아래에서 채널별 SM 수(`mps_sm_*`)를 받는다는 점은 검색 요약 수준이다. Next-Gen AI-on-RAN의 GPU 분할 수단과 arXiv 2602.23722가 L1을 GPU에서 돌렸는지도 확인하지 못했다. ARCHES는 한 조사에서 이름으로 찾지 못했고 다른 조사에서 초록만 확인했으며, MobiHoc'26 채택은 확인하지 못했다. AI-RAN Alliance 백서 PDF 원문, SynaXG의 MWC 2026 시연 출처, 이름만 전해진 "Atlas" vRAN 논문도 찾지 못했다.

동기 사실 쪽에서는 다음이 남는다. 상향 BLER 측정이 첫 전송 기준인지, 벤더 OLLA 기본 목표가 얼마인지 확인하지 못했다. 상용망에서 상향 MU-MIMO가 쓰이는 비율, 페어링된 TB의 BLER, 스케줄러가 고른 쌍의 잔여 상관, O-RAN 7.2x에서 DU가 받는 상향 stream 수에 대한 공개 자료도 없다. 그래서 "실패가 MU-MIMO에 몰린다"는 주장은 이 논문의 실험으로 보여야 한다. 규격 조항은 Rel-17 판 기준이므로 Rel-18/19에서 절 번호가 바뀌었는지 확인해야 한다. "RV 0과 3은 self-decodable"은 규격 문구가 아니라 해석이다. 11.5 ms와 RLC/PDCP 타이머의 관계, 구현마다 다른 MAC 선행 시간(UL DCI를 몇 슬롯 앞에 확정하는가)은 이번 노트의 범위 밖이다. 다만 프로젝트 내부 기록에는 Aerial 설정의 slot_advance 3이 있다. 사용자·layer 수에 따른 NRx 지연 곡선은 공개 자료에 없다. 3GPP RAN1 6G 회의에서 AI receiver가 use case로 채택됐는지도 1차 회의 자료로 확인하지 못했다. Ericsson 상향링크 기사는 제목과 URL이 서로 달라 인용 전에 다시 확인해야 한다. BigStation의 게재처(SIGCOMM 2013으로 알려짐), MRD의 26%·3배 수치(저자 요약 페이지), 그리고 Concordia·vrAIn·REEF·Orion·XSched·GPreempt의 정량 결과는 검색 요약 수준이다.

## 결론

빈자리는 기법이 아니라 시간 창에 있다. NVIDIA는 실시간 NRx의 성능이 추론 지연에 묶인다고 썼다. Ericsson은 PHY와 떼어 낸 추론의 여유를 수십–수백 µs로 봤다. Elnashar는 CRC 경로가 모든 슬롯의 복호 비용을 두 배로 만든다며 미뤘다. 셋 모두 NRx가 모든 블록의 임계 경로에 있어야 한다고 전제했다. NR의 비동기 HARQ는 실패한 약 10%의 블록에만 수 ms의 창을 준다. 그 창에서만 큰 모델을 돌리면 추가 비용은 실패율에 비례한다. 서론의 논리 순서는 이 사실들을 그대로 따르면 된다. 큰 NRx는 L1 마감에 들지 않는다. 그러나 실패한 블록에는 더 늦은 마감이 있다. 그 GPU는 AI와 공유되는데, 기존 공유 방식은 L1만 지킨다.

경쟁선은 사흘 전에 움직였다. Eunoia가 Aerial PHY 전체와 LLM을 슬롯 단위로 27셀까지 나누면서, "PHY와 LLM의 슬롯 단위 GPU 공유"는 더 이상 이 논문만의 영역이 아니다. 남는 차별점은 세 가지다. 실패 블록에만 생기는 늦고 큰 작업, 그 작업의 고유 마감, 그리고 그 마감을 기준으로 한 AI 실행 결정이다. 투고 전에 할 일은 세 가지다. Eunoia를 비교표와 기준선 설명에 넣는다. 특허를 전문 검색한다. 동기 문장의 출처를 규격·관행·측정으로 갈라 다시 단다.

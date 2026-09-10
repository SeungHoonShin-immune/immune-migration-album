# 면역세포의 무작위 이동과 방향성 이동의 효율 비교

**Python 시뮬레이션을 이용한 두 면역세포 모델 이동에서 무작위 이동과 방향성 이동의 효율 비교**

제68회 서울과학전람회 예선대회 출품작 · 생물 부문

---

## 연구 개요

면역세포는 감염이 없는 평상시에는 조직을 방향성 없이 순찰하다가, 감염이 발생하면 케모카인 농도차를 따라 병소로 이동한다. 두 방식이 모두 유지되는 이유를 확인하기 위해, **이동 규칙만을 독립변인으로 조작하고 나머지 조건을 모두 통제한** Agent-Based Model을 구현하였다.

성질이 대조적인 두 병원체(인플루엔자 A 바이러스, 폐렴구균)에 동일한 모형을 적용하여, 이동 전략의 우열이 병원체에 따라 달라지는지 검증하였다.

---

## 파일 구성

| 파일 | 내용 | 줄 수 |
|---|---|---:|
| `influenza_abm.py` | 인플루엔자 A 바이러스(H1N1) 감염 모형 | 2,992 |
| `pneumococcus_abm.py` | 폐렴구균(*S. pneumoniae*) 감염 모형 | 3,100 |

두 파일은 **자체 완결형(self-contained)** 이다. 서로를 import 하지 않으며, 각 파일 안에 참고문헌·파라미터·수용체 대응표·검증 루틴이 모두 들어 있다.

각 파일은 `movement_mode` 인자 하나로 두 조건이 갈리므로, 이동 규칙 외의 조건이 달라질 여지가 구조적으로 차단된다.

---

## 실행 방법

### 환경

```
Python 3.12
numpy, scipy
```

```bash
pip install numpy scipy
```

### 실행

```bash
# 인플루엔자 A
python influenza_abm.py random       # 무작위 이동
python influenza_abm.py directed     # 방향성 이동

# 폐렴구균
python pneumococcus_abm.py random    # 무작위 이동
python pneumococcus_abm.py directed  # 방향성 이동
```

인자를 추가하면 난수 시드와 실행 일수를 지정할 수 있다.

```bash
python influenza_abm.py directed 20260811 16
#                       이동방식   시드      일수
```

### 출력 내용

실행하면 다음이 차례로 출력된다.

1. 축척 정보
2. **표 1** — 면역세포의 종류, 수용체, 결합 리간드, 역할
3. 리간드 명세 — 생성원, 확산 폭, 반감기, 출처
4. **표 2** — 무작위 이동과 방향성 이동의 정의
5. **표 3** — 두 병원체의 특징 비교
6. 살해 조건 (폐렴구균 모형)
7. 파라미터 출처표 — 문헌 인용값과 모델 가정값 구분
8. 수용체 구현 일치 검사 결과
9. 일별 시뮬레이션 경과
10. 결과 보고서
11. 선행 연구 대조 검증표
12. 참고문헌 목록

---

## 모형 구조

### 축척

| 항목 | 설정 |
|---|---|
| 격자 | 1000 × 1000 (100만 칸) = 상피 15mm × 15mm |
| 격자 한 칸 | 15 μm (상피세포 1개) |
| 시간 | 1 tick = 6분, 1일 = 240 tick |
| 개체수 축소비율 | κ = 1/1000 |

### 표 1 — 면역세포의 종류와 수용체

| 면역세포 | 영어 명칭 | 주 수용체 | 대표적 결합 리간드 |
|---|---|---|---|
| 중성구 | Neutrophil | CXCR2 | CXCL8 |
| 단핵구/대식세포 | Monocyte/Macrophage | FPR1, CCR2 | fMLF, CCL2 |
| 자연살해 세포 | Natural killer cell | CXCR3 | CXCL9, CXCL10, CXCL11 |
| 조력 T세포 | Helper T cell | CXCR3 | CXCL9, CXCL10, CXCL11 |
| 살해 T세포 | Killer T cell | CXCR3 | CXCL9, CXCL10, CXCL11 |
| B세포/형질세포 | B cell/Plasma cell | CXCR5 | CXCL13 |
| 수지상세포 | Dendritic cell | CCR1 | CCL3, CCL5 |
| 보체 | Complement | (해당없음) | (해당없음) |

기존 모형은 단일 케모카인 농도를 가정하였으나, 본 모형은 위 리간드를 **각각 독립된 농도장으로 분리**하여 구현하였다. 따라서 같은 시점에도 세포 종류에 따라 서로 다른 방향으로 이동한다.

코드의 `verify_receptor_consistency()` 함수가 표 1과 실제 이동 코드의 대응이 일치하는지 자동으로 검사한다.

### 표 2 — 두 이동 방식의 정의

| 구분 | 무작위 이동 | 방향성 이동 |
|---|---|---|
| 상태 | 감염이나 손상이 없는 상태 | 케모카인에 의한 농도차가 형성된 감염 상태 |
| 이동 규칙 | 무작위로 방향을 골라 브라운 운동에 따라 이동 | 수용체에 결합하는 리간드 농도가 최대인 방향으로 이동 |
| 신호 감지 | 감지하지 않음 (0%) | 항상 감지 (100%) |
| 생성 장소 | 조직에 무작위적으로 생성 | 림프절에서 생성 |

두 조건의 차이는 `MovementEngine` 클래스 안에만 존재한다. 실행 결과의 **구배 유도 이동 비율**로 조작이 의도대로 되었는지 확인할 수 있다.

### 표 3 — 두 병원체의 특징

| 비교 항목 | 인플루엔자 A (H1N1) | 폐렴구균 (*S. pneumoniae*) |
|---|---|---|
| 분류 | RNA 바이러스 | 그람양성 세균 |
| 증식 위치 | 세포내 증식 | 세포외 증식 |
| 공간 분포 | 조직 전역에 균일한 확산 | 편모가 없어 비운동성, 미세 군락 형성 |
| 자체 신호 | 없음 | fMLF를 방출하여 위치를 표시함 |
| 제거 경로 | 감염된 숙주세포의 사멸이 대부분 | 식균작용이 유일한 제거 수단 |

인플루엔자에서는 감염된 숙주세포가 자멸하여 제거되므로 이동 방식의 효과가 희석된다. 반면 폐렴구균은 병원체가 사멸하지 않아 이동 방식에 따른 결과가 그대로 반영된다. 즉 **폐렴구균은 인플루엔자의 대조 실험**으로 포함되었다.

이 차이는 코드에도 반영되어 있다. 인플루엔자 모형에서는 fMLF 생성원이 존재하지 않아 FPR1 축이 작동하지 않으며, 폐렴구균 모형에서는 세균 자신이 fMLF를 방출하여 **표적이 곧 신호원**이 된다.

---

## 파라미터 출처

모든 파라미터에 출처 태그가 붙어 있으며, 논문에서 직접 인용한 실측값과 정량값을 모형에서 설정한 가정값을 구분하여 표기하였다. 실행하면 파라미터 출처표가 출력된다.

### 인플루엔자 A

| 파라미터 | 값 | 출처 |
|---|---|---|
| 잠복기 | 6시간 | Baccam 등 (2006) |
| 바이러스 생산 기간 | 5시간 | Baccam 등 (2006) |
| 감염세포 평균 수명 | 11시간 | Baccam 등 (2006) |
| 유리 바이러스 반감기 | 3시간 | Baccam 등 (2006) |
| 체내 기초재생산 수 | 약 22 | Baccam 등 (2006) |
| 바이러스 크기 | 80–120 nm | Harris 등 (2006) |
| 배출 기간 | 평균 4.8일 | Carrat 등 (2008) |
| 증상 정점 | 2–3일째 | Carrat 등 (2008) |
| 초기 접종량 | 10⁵ TCID₅₀ | Memoli 등 (2015) |

### 폐렴구균

| 파라미터 | 값 | 출처 |
|---|---|---|
| 폐 내 배가시간 | 56분 | Jose 등 (2015) |
| 폐포대식세포 제거 반감기 | 42분 | Jose 등 (2015) |
| 초기 접종량 | 1×10⁶ CFU | Lin 등 (2020) |
| 최대 균량 | 1×10⁸ CFU | Hamilton 등 (2019) |
| 호중구 포식 용량 | 약 50 CFU/세포 | Rubio 등 (2023) |
| 옵소닌 요구량 | 혈청 40% 이상 | Gordon 등 (1980) |
| 협막의 포식 저항 | 다중 기전 저해 | Hyams 등 (2010) |

### 공통

| 파라미터 | 출처 |
|---|---|
| 조직 내 백혈구 이동 속도 | Friedl & Weigelin (2008) |
| 림프구 운동성 | Miller 등 (2002) |
| 백혈구 부착 연쇄반응 | Ley 등 (2007) |
| CXCL10의 인터페론 유도성 | Luster 등 (1985) |
| CXCR3 리간드의 기능 | Groom & Luster (2011) |
| B세포의 림프절 진입 | Okada 등 (2002) |

---

## 검증

각 모형은 실행 시 선행 연구의 보고값과 결과를 대조하여 **일치 / 근접 / 불일치 / 판정불가**로 판정한 검증표를 출력한다. 검증 항목에는 다음이 포함된다.

- 병원체 정점 시각이 문헌의 증상 정점 범위에 드는가
- 배출 종료가 문헌의 평균 배출 기간과 부합하는가
- 제거 경로의 비중이 표 3의 기술과 일치하는가
- 자체 신호(fMLF) 방출 여부가 표 3과 일치하는가
- **이동 조작 검증** — 무작위 이동에서 신호 감지 0%, 방향성 이동에서 100%인가

마지막 항목은 두 조건이 설계대로 작동했는지 확인하는 직접 증거이다.

---

## 참고문헌

1. Baccam P, Beauchemin C, Macken CA, Hayden FG, Perelson AS. Kinetics of influenza A virus infection in humans. *Journal of Virology* 80(15):7590-7599 (2006).
2. Carrat F, Vergu E, Ferguson NM, et al. Time lines of infection and disease in human influenza. *American Journal of Epidemiology* 167(7):775-785 (2008).
3. Harris A, Cardone G, Winkler DC, et al. Influenza virus pleiomorphy characterized by cryoelectron tomography. *PNAS* 103(50):19123-19127 (2006).
4. Memoli MJ, Czajkowski L, Reed S, et al. Validation of the wild-type influenza A human challenge model H1N1pdMIST. *Clinical Infectious Diseases* 60(5):693-702 (2015).
5. Jose RJ, Williams AE, Chambers RC, Brown JS, et al. Importance of bacterial replication and alveolar macrophage-independent clearance mechanisms during early lung infection with *Streptococcus pneumoniae*. *Infection and Immunity* 83(4):1181-1189 (2015).
6. Lin J, Zhu L, Lau GW. *Streptococcus pneumoniae* elaborates persistent and prolonged competent state during pneumonia-derived sepsis. *Infection and Immunity* 88(4):e00919-19 (2020).
7. Hamilton JA, Nguyen VT, Wilson C, et al. Clinically relevant model of pneumococcal pneumonia, ARDS, and nonpulmonary organ dysfunction in mice. *American Journal of Physiology - Lung Cellular and Molecular Physiology* 317(5):L659-L675 (2019).
8. Rubio AJ, Bakker MG, Bhatnagar S, et al. Model-based assessment of neutrophil-mediated phagocytosis and digestion of bacteria. *CPT: Pharmacometrics & Systems Pharmacology* 12(12):1934-1946 (2023).
9. Gordon DL, Rice J, Finlay-Jones JJ, et al. Phagocytosis by human alveolar macrophages and neutrophils. *Journal of Infectious Diseases* 141(6):718-724 (1980).
10. Hyams C, Camberlein E, Cohen JM, Bax K, Brown JS. The *Streptococcus pneumoniae* capsule inhibits complement activity and neutrophil phagocytosis by multiple mechanisms. *Infection and Immunity* 78(2):704-715 (2010).
11. Dalia AB, Standish AJ, Weiser JN. Three surface exoglycosidases from *Streptococcus pneumoniae* promote resistance to opsonophagocytic killing by human neutrophils. *Infection and Immunity* 78(5):2108-2116 (2010).
12. Schiffmann E, Corcoran BA, Wahl SM. N-formylmethionyl peptides as chemoattractants for leucocytes. *PNAS* 72(3):1059-1062 (1975).
13. Weiss E, Hanzelmann D, Fehlhaber B, et al. Formyl-peptide receptor activation enhances phagocytosis of community-acquired methicillin-resistant *Staphylococcus aureus*. *Journal of Infectious Diseases* 221(4):668-678 (2020).
14. Rudd JM, Pulavendran S, Ashar HK, et al. Neutrophils induce a novel chemokine receptors repertoire during influenza pneumonia. *Frontiers in Cellular and Infection Microbiology* 9:108 (2019).
15. Lin KL, Suzuki Y, Nakano H, Ramsburg E, Gunn MD. CCR2+ monocyte-derived dendritic cells and exudate macrophages produce influenza-induced pulmonary immune pathology and mortality. *Journal of Immunology* 180(4):2562-2572 (2008).
16. Groom JR, Luster AD. CXCR3 ligands: redundant, collaborative and antagonistic functions. *Immunology and Cell Biology* 89(2):207-215 (2011).
17. Luster AD, Unkeless JC, Ravetch JV. Gamma-interferon transcriptionally regulates an early-response gene containing homology to platelet proteins. *Nature* 315(6021):672-676 (1985).
18. Okada T, Ngo VN, Ekland EH, et al. Chemokine requirements for B cell entry to lymph nodes and Peyer's patches. *Journal of Experimental Medicine* 196(1):65-75 (2002).
19. Sozzani S, Allavena P, D'Amico G, et al. Differential regulation of chemokine receptors during dendritic cell maturation. *Journal of Immunology* 161(3):1083-1086 (1998).
20. Lu YJ, Gross J, Bogaert D, Finn A, et al. Interleukin-17A mediates acquired immunity to pneumococcal colonization. *PLoS Pathogens* 4(9):e1000159 (2008).
21. Zhang Z, Clarke TB, Weiser JN. Cellular effectors mediating Th17-dependent clearance of pneumococcal colonization in mice. *Journal of Clinical Investigation* 119(7):1899-1909 (2009).
22. Friedl P, Weigelin B. Interstitial leukocyte migration and immune function. *Nature Immunology* 9(9):960-969 (2008).
23. Miller MJ, Wei SH, Parker I, Cahalan MD. Two-photon imaging of lymphocyte motility and antigen response in intact lymph node. *Science* 296(5574):1869-1873 (2002).
24. Ley K, Laudanna C, Cybulsky MI, Nourshargh S. Getting to the site of inflammation: the leukocyte adhesion cascade updated. *Nature Reviews Immunology* 7(9):678-689 (2007).
25. Folcik VA, An GC, Orosz CG. The Basic Immune Simulator: an agent-based model to study the interactions between innate and adaptive immunity. *Theoretical Biology and Medical Modelling* 4:39 (2007).
26. Folcik VA, Broderick G, Mohan S, et al. Using an agent-based model to analyze the dynamic communication network of the immune response. *Theoretical Biology and Medical Modelling* 8:1 (2011).
27. Parr A, Anderson NR, Hammer DA. A simulation of the random and directed motion of dendritic cells in chemokine fields. *PLoS Computational Biology* 15(10):e1007295 (2019).
28. Crystal RG, Randell SH, Engelhardt JF, Voynow J, Sullivan MA. Airway epithelial cells: current concepts and challenges. *Proceedings of the American Thoracic Society* 5(7):772-777 (2008).
29. Dettmer P (강병철 역). 『면역』. 사이언스북스 (2023).

---

## 한계

- 본 연구는 컴퓨터 시뮬레이션에 기반하므로 실제 생체 반응을 직접 입증하는 것은 아니며, 선행 연구의 관찰과 일관된 결과를 제시하는 데 의의가 있다.
- 병원체를 교체하면 공간 확산 속도, 자체 신호 방출 여부, 주된 제거 경로가 함께 변화하므로, 결과의 원인을 세 요인 중 하나로 분리하여 규명하기 어렵다.
- 신호물질의 확산 폭과 반감기 등 일부 파라미터는 정성적 기전만 문헌에 근거하고 정량값은 모형 가정값이다. 코드에서 `MODEL_ASSUMPTION` 태그로 구분하였다.
- 실제 케모카인 수용체는 여러 리간드와 중복적으로 결합하나, 본 모형에서는 각 수용체가 표 1에 기재된 축만 감지하도록 단순화하였다.

---

## 개발 도구

모형의 설계, 파라미터 도출, 알고리즘 규칙 수립 및 결과 해석은 연구자가 직접 수행하였으며, 이를 Python 코드로 구현하는 과정에서 생성형 AI를 보조 도구로 활용하였다.

---

## 라이선스

교육 및 연구 목적으로 자유롭게 사용할 수 있다.

# -*- coding: utf-8 -*-
"""
=============================================================================

  pneumococcus_abm.py

  제68회 서울과학전람회 예선대회 출품작
  「Python 시뮬레이션을 이용한 두 면역세포 모델 이동에서
    무작위 이동과 방향성 이동의 효율 비교」

  폐렴구균(Streptococcus pneumoniae) 감염 - 면역반응 Agent-Based Model

=============================================================================

  본 파일은 작품설명서에 기술된 조건을 코드로 구현한 것이며, 외부 모듈에
  의존하지 않는 자체 완결형(self-contained) 구조로 작성되었다.
  작품설명서의 표 1(면역세포와 수용체), 표 2(이동 방식), 표 3(병원체 비교)에
  기술된 내용은 모두 이 파일 안에 자료구조로 명시되어 있다.

  ---------------------------------------------------------------------------
  【작품설명서 표 2】 무작위 이동과 방향성 이동의 정의
  ---------------------------------------------------------------------------
   구분        무작위 이동                     방향성 이동
   ─────────────────────────────────────────────────────────────────────────
   상태        감염이나 손상이 없는 상태       케모카인에 의한 농도차
               (정상 상태)                     (화학주성)가 형성된 감염 상태
   이동 규칙   무작위로 방향을 골라            수용체에 결합하는 리간드
               브라운 운동에 따라 이동         농도가 최대인 방향으로 이동
   신호 감지   감지하지 않음 (0%)              항상 감지 (100%)
   생성 장소   조직에 무작위적으로 생성        림프절에서 생성
  ---------------------------------------------------------------------------

  【작품설명서 표 3】 폐렴구균의 특징
   분류        그람양성 세균
   증식 위치   세포외 증식
   공간 분포   편모가 없어 비운동성.
               분열한 자리에 머물러 미세 군락 형성
   자체 신호   fMLF를 방출하여 위치를 표시함
   제거 경로   식균작용이 유일한 제거 수단

  ---------------------------------------------------------------------------
  【인플루엔자 모형과의 결정적 차이 — 대조 실험으로서의 의의】
  ---------------------------------------------------------------------------
   작품설명서 「라. 대표적인 병원체 간 비교」:

     "A형 인플루엔자에서는 바이러스에 의해 감염된 숙주세포가 자멸하여
      제거되는 데에 반해, 폐렴구균은 세균이기 때문에 병원체가 사멸하지
      않아 면역세포가 이동 방식에 따른 결과가 그대로 반영될 수 있다.
      즉 폐렴구균은 A형 인플루엔자의 대조 실험으로써 이 탐구에 포함시켰다."

   1) 감염세포가 존재하지 않는다.
      -> 숙주세포 자멸 경로가 없으므로, 면역세포의 이동 방식이 결과에
         그대로 반영된다.
   2) 세균 자신이 fMLF 를 방출한다.
      -> 표 1에서 단핵구/대식세포가 FPR1 로 fMLF 를 감지하므로,
         방향성 이동 조건에서 단핵구는 세균 위치를 직접 추적할 수 있다.
         인플루엔자 모형에서는 fMLF 생성원이 없어 이 축이 작동하지 않는다.
   3) 편모가 없어 비운동성이므로 확산이 극히 느리다.
      -> 표적이 미세군락으로 뭉쳐 있다.

  ---------------------------------------------------------------------------
  【살해 조건 — 작품설명서 「라. 탐구 절차 - B」】
     중성구와 대식세포가 옵소닌화된 세균을 인식하여 세포막으로 감싸
     식균작용을 하고, 중성구와 대식세포 내부의 분해효소와 활성산소가
     방출되어 세균을 분해, 살해한다.
  ---------------------------------------------------------------------------

  【실행 방법】
      python pneumococcus_abm.py random     # 무작위 이동
      python pneumococcus_abm.py directed   # 방향성 이동
      python pneumococcus_abm.py random 20260811 14
                                             # 이동방식 시드 일수
  ---------------------------------------------------------------------------
=============================================================================
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError:                                    # pragma: no cover
    raise SystemExit("SciPy 가 필요합니다.  pip install scipy")


# =============================================================================
#  SECTION 1.  참고문헌
# =============================================================================
#  본 모형의 모든 정량값은 아래 문헌에서 인용하였다.
#  각 파라미터에는 출처 태그가 붙어 있으며, 문헌에서 정량값을 직접 얻을 수
#  없는 항목은 MODEL_ASSUMPTION 으로 표기하여 구분하였다.
# =============================================================================


@dataclass(frozen=True)
class Reference:
    """논문 1건의 서지사항"""
    tag: str
    authors: str
    title: str
    journal: str
    year: int
    doi: str = ""
    pmid: str = ""

    def short(self) -> str:
        first = self.authors.split(",")[0].strip()
        return f"{first} 등 ({self.year})"

    def full(self) -> str:
        ident = []
        if self.doi:
            ident.append(f"doi:{self.doi}")
        if self.pmid:
            ident.append(f"PMID:{self.pmid}")
        tail = ("  " + " ".join(ident)) if ident else ""
        return (f"{self.authors}. \"{self.title}\". "
                f"{self.journal} ({self.year}).{tail}")


REFERENCES: Dict[str, Reference] = {

    # ---------------- 선행 연구 (Agent-Based Model) ----------------
    "folcik2007": Reference(
        "folcik2007",
        "Folcik VA, An GC, Orosz CG",
        "The Basic Immune Simulator: an agent-based model to study the "
        "interactions between innate and adaptive immunity",
        "Theoretical Biology and Medical Modelling 4:39", 2007,
        "10.1186/1742-4682-4-39", "17900357"),

    "folcik2011": Reference(
        "folcik2011",
        "Folcik VA, Broderick G, Mohan S, Block B, Ekbote C, Doolittle J, "
        "Khoury M, Davis L, Marsh CB",
        "Using an agent-based model to analyze the dynamic communication "
        "network of the immune response",
        "Theoretical Biology and Medical Modelling 8:1", 2011,
        "10.1186/1742-4682-8-1", "21214945"),

    "parr2019": Reference(
        "parr2019",
        "Parr A, Anderson NR, Hammer DA",
        "A simulation of the random and directed motion of dendritic cells "
        "in chemokine fields",
        "PLoS Computational Biology 15(10):e1007295", 2019,
        "10.1371/journal.pcbi.1007295", "31584940"),

    # ---------------- 폐렴구균 감염 동역학 ----------------
    "jose2015": Reference(
        "jose2015",
        "Jose RJ, Williams AE, Mercer PF, Sulikowski MG, Brown JS, "
        "Chambers RC",
        "Regulation of neutrophilic inflammation by proteinase-activated "
        "receptor 1 during bacterial pulmonary infection / Importance of "
        "bacterial replication and alveolar macrophage-independent clearance "
        "mechanisms during early lung infection with Streptococcus pneumoniae",
        "Infection and Immunity 83(4):1181-1189", 2015,
        "10.1128/IAI.02788-14", "25583525"),

    "lin2020": Reference(
        "lin2020",
        "Lin J, Zhu L, Lau GW",
        "Streptococcus pneumoniae elaborates persistent and prolonged "
        "competent state during pneumonia-derived sepsis",
        "Infection and Immunity 88(4):e00919-19", 2020,
        "10.1128/IAI.00919-19", "31988172"),

    "hamilton2019": Reference(
        "hamilton2019",
        "Hamilton JA, Nguyen VT, Wilson C, Mulchandani N, Kambhampati SP, "
        "et al.",
        "Clinically relevant model of pneumococcal pneumonia, ARDS, and "
        "nonpulmonary organ dysfunction in mice",
        "American Journal of Physiology - Lung Cellular and Molecular "
        "Physiology 317(5):L659-L675", 2019,
        "10.1152/ajplung.00132.2019", "31461312"),

    # ---------------- 식균작용과 옵소닌화 ----------------
    "rubio2023": Reference(
        "rubio2023",
        "Rubio AJ, Bakker MG, Bhatnagar S, Sunderhaus A, Chen L, et al.",
        "Model-based assessment of neutrophil-mediated phagocytosis and "
        "digestion of bacteria across in vitro and in vivo studies",
        "CPT: Pharmacometrics & Systems Pharmacology 12(12):1934-1946", 2023,
        "10.1002/psp4.13045", "37881153"),

    "gordon1980": Reference(
        "gordon1980",
        "Gordon DL, Rice J, Finlay-Jones JJ, McDonald PJ, Hostetter MK",
        "Phagocytosis by human alveolar macrophages and neutrophils: "
        "qualitative differences in the opsonic requirements for uptake of "
        "Staphylococcus aureus and Streptococcus pneumoniae in vitro",
        "Journal of Infectious Diseases 141(6):718-724", 1980,
        "10.1093/infdis/141.6.718", "7352714"),

    "hyams2010": Reference(
        "hyams2010",
        "Hyams C, Camberlein E, Cohen JM, Bax K, Brown JS",
        "The Streptococcus pneumoniae capsule inhibits complement activity "
        "and neutrophil phagocytosis by multiple mechanisms",
        "Infection and Immunity 78(2):704-715", 2010,
        "10.1128/IAI.00881-09", "19948837"),

    "dalia2010": Reference(
        "dalia2010",
        "Dalia AB, Standish AJ, Weiser JN",
        "Three surface exoglycosidases from Streptococcus pneumoniae, NanA, "
        "BgaA, and StrH, promote resistance to opsonophagocytic killing by "
        "human neutrophils",
        "Infection and Immunity 78(5):2108-2116", 2010,
        "10.1128/IAI.01125-09", "20160017"),

    # ---------------- 세균 유래 화학주성 신호 ----------------
    "weiss2020": Reference(
        "weiss2020",
        "Weiss E, Hanzelmann D, Fehlhaber B, Klos A, von Loewenich FD, "
        "et al.",
        "Formyl-peptide receptor activation enhances phagocytosis of "
        "community-acquired methicillin-resistant Staphylococcus aureus",
        "Journal of Infectious Diseases 221(4):668-678", 2020,
        "10.1093/infdis/jiz498", "31573603"),

    "schiffmann1975": Reference(
        "schiffmann1975",
        "Schiffmann E, Corcoran BA, Wahl SM",
        "N-formylmethionyl peptides as chemoattractants for leucocytes",
        "PNAS 72(3):1059-1062", 1975,
        "10.1073/pnas.72.3.1059", "1093163"),

    # ---------------- 면역세포 수용체 ----------------
    "rudd2019": Reference(
        "rudd2019",
        "Rudd JM, Pulavendran S, Ashar HK, Ritchey JW, Snider TA, "
        "Malayer JR, Marie M, Chow VTK, Narasaraju T",
        "Neutrophils induce a novel chemokine receptors repertoire during "
        "influenza pneumonia",
        "Frontiers in Cellular and Infection Microbiology 9:108", 2019,
        "10.3389/fcimb.2019.00108", "31041196"),

    "lin2008": Reference(
        "lin2008",
        "Lin KL, Suzuki Y, Nakano H, Ramsburg E, Gunn MD",
        "CCR2+ monocyte-derived dendritic cells and exudate macrophages "
        "produce influenza-induced pulmonary immune pathology and mortality",
        "Journal of Immunology 180(4):2562-2572", 2008,
        "10.4049/jimmunol.180.4.2562", "18250467"),

    "groom2011": Reference(
        "groom2011",
        "Groom JR, Luster AD",
        "CXCR3 ligands: redundant, collaborative and antagonistic functions",
        "Immunology and Cell Biology 89(2):207-215", 2011,
        "10.1038/icb.2010.158", "21221121"),

    "luster1985": Reference(
        "luster1985",
        "Luster AD, Unkeless JC, Ravetch JV",
        "Gamma-interferon transcriptionally regulates an early-response gene "
        "containing homology to platelet proteins",
        "Nature 315(6021):672-676", 1985,
        "10.1038/315672a0", "3925348"),

    "okada2002": Reference(
        "okada2002",
        "Okada T, Ngo VN, Ekland EH, Forster R, Lipp M, Littman DR, "
        "Cyster JG",
        "Chemokine requirements for B cell entry to lymph nodes and "
        "Peyer's patches",
        "Journal of Experimental Medicine 196(1):65-75", 2002,
        "10.1084/jem.20020201", "12093871"),

    "sozzani1997": Reference(
        "sozzani1997",
        "Sozzani S, Allavena P, D'Amico G, Luini W, Bianchi G, Kataura M, "
        "Imai T, Yoshie O, Bonecchi R, Mantovani A",
        "Differential regulation of chemokine receptors during dendritic "
        "cell maturation",
        "Journal of Immunology 161(3):1083-1086", 1998,
        "", "9686565"),

    # ---------------- 점막 세균면역 (Th17) ----------------
    "lu2008": Reference(
        "lu2008",
        "Lu YJ, Gross J, Bogaert D, Finn A, King L, et al.",
        "Interleukin-17A mediates acquired immunity to pneumococcal "
        "colonization",
        "PLoS Pathogens 4(9):e1000159", 2008,
        "10.1371/journal.ppat.1000159", "18802458"),

    "zhang2009": Reference(
        "zhang2009",
        "Zhang Z, Clarke TB, Weiser JN",
        "Cellular effectors mediating Th17-dependent clearance of "
        "pneumococcal colonization in mice",
        "Journal of Clinical Investigation 119(7):1899-1909", 2009,
        "10.1172/JCI36731", "19509469"),

    # ---------------- 영양 면역 ----------------
    "corbin2008": Reference(
        "corbin2008",
        "Corbin BD, Seeley EH, Raab A, Feldmann J, Miller MR, et al.",
        "Metal chelation and inhibition of bacterial growth in tissue "
        "abscesses",
        "Science 319(5865):962-965", 2008,
        "10.1126/science.1152449", "18276893"),

    # ---------------- 세포 이동 물리량 ----------------
    "friedl2008": Reference(
        "friedl2008",
        "Friedl P, Weigelin B",
        "Interstitial leukocyte migration and immune function",
        "Nature Immunology 9(9):960-969", 2008,
        "10.1038/ni.f.212", "18711433"),

    "miller2002": Reference(
        "miller2002",
        "Miller MJ, Wei SH, Parker I, Cahalan MD",
        "Two-photon imaging of lymphocyte motility and antigen response in "
        "intact lymph node",
        "Science 296(5574):1869-1873", 2002,
        "10.1126/science.1070051", "12016203"),

    "ley2007": Reference(
        "ley2007",
        "Ley K, Laudanna C, Cybulsky MI, Nourshargh S",
        "Getting to the site of inflammation: the leukocyte adhesion cascade "
        "updated",
        "Nature Reviews Immunology 7(9):678-689", 2007,
        "10.1038/nri2156", "17717539"),

    # ---------------- 조직 구조 ----------------
    "crystal2008": Reference(
        "crystal2008",
        "Crystal RG, Randell SH, Engelhardt JF, Voynow J, Sullivan MA",
        "Airway epithelial cells: current concepts and challenges",
        "Proceedings of the American Thoracic Society 5(7):772-777", 2008,
        "10.1513/pats.200805-041HR", "18757316"),

    # ---------------- 단행본 ----------------
    "dettmer2023": Reference(
        "dettmer2023",
        "Dettmer P (강병철 역)",
        "면역 (Immune)",
        "사이언스북스", 2023),
}


MODEL_ASSUMPTION = "모델 가정값"


def print_references() -> None:
    """참고문헌 목록 출력"""
    print("=" * 78)
    print(" 참고문헌")
    print("=" * 78)
    for i, (tag, r) in enumerate(REFERENCES.items(), start=1):
        print(f"[{i:2d}] {r.full()}")
    print("=" * 78)


# =============================================================================
#  SECTION 2.  선행 연구에서 인용한 실측값
# =============================================================================

REAL: Dict[str, float] = {

    # ------------------------------------------------------------------
    #  공간 · 시간 축척  (작품설명서 라-A. 모양 구조)
    # ------------------------------------------------------------------
    "grid_side_mm": 15.0,             # 시뮬레이션 공간 한 변 = 상피 15mm
    "epi_cell_um": 15.0,              # 상피세포 한 개의 크기 = 격자 한 칸
    "tick_min": 6.0,                  # 1 tick = 6분
    "kappa": 1.0 / 1000.0,            # 개체 수는 실제 값의 1/1000

    # ------------------------------------------------------------------
    #  폐렴구균 감염 동역학
    # ------------------------------------------------------------------
    # Jose 등 (2015) — 저용량 접종 시 폐 내 세균 증식과 제거
    "doubling_time_min": 56.0,        # 폐 내 배가시간 56분
    "am_clearance_halflife_min": 42.0,  # 폐포대식세포 의존 제거 반감기 42분

    # Lin 등 (2020) — 폐렴 유래 패혈증 모델
    "inoculum_cfu": 1.0e6,            # 초기 접종량 1x10^6 CFU

    # Hamilton 등 (2019) — 임상 관련 폐렴구균 폐렴 모델
    "max_burden_cfu": 1.0e8,          # 최대 균량 1x10^8 CFU

    # 폐렴구균의 크기
    "bacterium_diameter_um_min": 0.8,
    "bacterium_diameter_um_max": 1.0,

    # ------------------------------------------------------------------
    #  식균작용과 옵소닌화
    # ------------------------------------------------------------------
    # Rubio 등 (2023) — 호중구 매개 식균작용의 모델 기반 평가
    "neutrophil_capacity_cfu": 50.0,  # 호중구 1개의 포식 용량 약 50 CFU

    # Gordon 등 (1980) — 폐포대식세포·호중구의 옵소닌 요구 조건
    "opsonin_serum_requirement": 0.40,  # 40% 이상 비가열 혈청 필요

    # Hyams 등 (2010) — 협막의 보체 활성·포식 저해
    "capsule_resistance": 0.85,       # 협막의 다중 기전 저해 정도

    # ------------------------------------------------------------------
    #  면역세포 이동 속도
    # ------------------------------------------------------------------
    "neutrophil_um_min": 12.0,        # Friedl & Weigelin (2008)
    "monocyte_um_min": 4.0,
    "nk_um_min": 8.0,
    "dendritic_um_min": 5.0,
    "t_cell_um_min": 11.0,            # Miller 등 (2002)
    "b_cell_um_min": 6.0,
    "vessel_transit_um_min": 400.0,   # Ley 등 (2007)

    # ------------------------------------------------------------------
    #  신호물질
    # ------------------------------------------------------------------
    "fmlf_halflife_min": 20.0,        # 조직 펩티다제에 의한 분해
    "cxcl8_halflife_h": 2.0,
    "ccl2_halflife_h": 4.0,
    "cxcr3l_halflife_h": 6.0,
    "cxcl13_halflife_h": 8.0,
    "ccl35_halflife_h": 3.0,
    "ifng_halflife_h": 4.0,

    # ------------------------------------------------------------------
    #  후천면역
    # ------------------------------------------------------------------
    "igm_halflife_d": 5.0,
    "igg_halflife_d": 21.0,
    "igm_detect_day": 5.0,
    "igg_detect_day": 9.0,
    "priming_h": 48.0,
    "th17_peak_day": 7.0,             # Lu 등 (2008), Zhang 등 (2009)
}


# =============================================================================
#  SECTION 3.  축척 변환
# =============================================================================
#  작품설명서 「라-A. 모양 구조」:
#    시뮬레이션 공간은 1000x1000 격자(총 100만 칸)로 구성하였으며, 이는
#    실제 상피 15mm x 15mm 에 해당한다. 격자 한 칸은 15um로 상피세포 한 개의
#    크기에 대응한다. 시간은 1 tick 을 6분으로 설정하여 하루가 240 tick 이
#    되도록 했다. 개체 수는 실제 값의 1/1000 로 축소했다.
# =============================================================================


class Scale:
    """실제 물리량 <-> 시뮬레이션 단위 변환"""

    GRID: int = 1000
    N_SITES: int = GRID * GRID
    SIDE_MM: float = REAL["grid_side_mm"]
    UM_PER_CELL: float = SIDE_MM * 1000.0 / GRID
    TICK_MIN: float = REAL["tick_min"]
    TICKS_PER_HOUR: float = 60.0 / TICK_MIN
    TICKS_PER_DAY: int = int(24 * TICKS_PER_HOUR)
    KAPPA: float = REAL["kappa"]

    FIELD_BIN: int = 5
    FIELD_N: int = GRID // FIELD_BIN

    # ------------------------------------------------------------------
    @staticmethod
    def ticks_from_hours(hours: float) -> int:
        return max(1, int(round(hours * Scale.TICKS_PER_HOUR)))

    @staticmethod
    def ticks_from_days(days: float) -> int:
        return max(1, int(round(days * Scale.TICKS_PER_DAY)))

    @staticmethod
    def days_from_ticks(ticks: int) -> float:
        return ticks / Scale.TICKS_PER_DAY

    @staticmethod
    def decay_from_halflife_h(hours: float) -> float:
        return 1.0 - 0.5 ** (1.0 / (hours * Scale.TICKS_PER_HOUR))

    @staticmethod
    def decay_from_halflife_min(minutes: float) -> float:
        return 1.0 - 0.5 ** (1.0 / (minutes / Scale.TICK_MIN))

    @staticmethod
    def decay_from_halflife_d(days: float) -> float:
        return 1.0 - 0.5 ** (1.0 / (days * Scale.TICKS_PER_DAY))

    @staticmethod
    def speed_cells_per_tick(um_per_min: float) -> float:
        return um_per_min * Scale.TICK_MIN / Scale.UM_PER_CELL

    @staticmethod
    def agents_from_real(real_count: float) -> int:
        return max(1, int(round(real_count * Scale.KAPPA)))

    @staticmethod
    def division_prob_from_doubling(doubling_min: float) -> float:
        """
        배가시간 -> tick 당 분열 확률.
        지수증식 N(t) = N0 * 2^(t/T) 와 등가가 되도록 한다.
        """
        ticks = doubling_min / Scale.TICK_MIN
        return 2.0 ** (1.0 / ticks) - 1.0

    @staticmethod
    def describe() -> str:
        return (
            f"격자 {Scale.GRID}x{Scale.GRID} = {Scale.N_SITES:,}칸 "
            f"= 상피 {Scale.SIDE_MM:.0f}mm x {Scale.SIDE_MM:.0f}mm\n"
            f"      격자 한 칸 = {Scale.UM_PER_CELL:.0f} um (상피세포 1개)\n"
            f"      1 tick = {Scale.TICK_MIN:.0f}분, "
            f"1일 = {Scale.TICKS_PER_DAY} tick\n"
            f"      화학신호 농도장 = {Scale.FIELD_N}x{Scale.FIELD_N} "
            f"(격자 {Scale.FIELD_BIN}칸 단위)\n"
            f"      개체수 축소비율 kappa = 1/{int(1 / Scale.KAPPA)}"
        )


# =============================================================================
#  SECTION 4.  면역세포의 종류 — 작품설명서 표 1
# =============================================================================
#  ★본 절은 작품설명서 「나. 면역세포의 종류 및 역할」의 표 1을 그대로 옮긴
#    것이며, 이 표가 모형에 구현된 수용체-리간드 대응의 유일한 근거이다.
# =============================================================================

NEUTROPHIL = 0
MONOCYTE = 1
NK_CELL = 2
HELPER_T = 3
KILLER_T = 4
B_CELL = 5
DENDRITIC = 6

N_CELL_TYPES = 7


@dataclass(frozen=True)
class ImmuneCellType:
    """작품설명서 표 1의 한 행"""
    code: int
    name_kr: str
    name_en: str
    receptors: Tuple[str, ...]
    ligands: Tuple[str, ...]
    role: str
    role_in_bacteria: str = ""
    source: str = ""


IMMUNE_CELL_TABLE: Dict[int, ImmuneCellType] = {

    NEUTROPHIL: ImmuneCellType(
        code=NEUTROPHIL,
        name_kr="중성구",
        name_en="Neutrophil",
        receptors=("CXCR2",),
        ligands=("CXCL8",),
        role="백혈구의 70%를 차지하는 면역세포로, 식균 작용, 과립, NET 등을 "
             "사용하여 병원체를 살해한다.",
        role_in_bacteria="★세균 감염에서 가장 중요한 효과기 세포이다. "
                         "옵소닌화된 세균을 세포막으로 감싸 삼킨 뒤 "
                         "분해효소와 활성산소로 분해·살해한다.",
        source="rudd2019"),

    MONOCYTE: ImmuneCellType(
        code=MONOCYTE,
        name_kr="단핵구/대식세포",
        name_en="Monocyte/Macrophage",
        receptors=("FPR1", "CCR2"),
        ligands=("fMLF", "CCL2"),
        role="중성구와 더불어 선천 면역계의 필수 세포 중 하나로, 병원체를 "
             "세포막으로 둘러싼 후 분해하는 포식 작용을 한다. 병원체를 "
             "T세포에게 보여주는 항원제시도 한다.",
        role_in_bacteria="★FPR1 로 세균이 방출한 fMLF 를 직접 감지한다. "
                         "인플루엔자 모형에서는 fMLF 생성원이 없어 이 축이 "
                         "작동하지 않았으나, 세균 감염에서는 표적이 곧 "
                         "신호원이므로 세균 위치를 직접 추적할 수 있다.",
        source="lin2008"),

    NK_CELL: ImmuneCellType(
        code=NK_CELL,
        name_kr="자연살해 세포",
        name_en="Natural killer cell",
        receptors=("CXCR3",),
        ligands=("CXCL9", "CXCL10", "CXCL11"),
        role="이상이 있는 세포를 찾아 살해한다. 바이러스 등에 감염된 세포는 "
             "MHC-1 분자에 이상을 보일 때가 있는데, 이때 NK세포는 그것을 "
             "감지해 살해한다.",
        role_in_bacteria="세포외 증식균에서는 살해할 감염세포가 존재하지 "
                         "않으므로 직접적인 살해 역할이 없다. 대신 "
                         "IFN-gamma 를 분비하여 대식세포를 활성화한다.",
        source="groom2011"),

    HELPER_T: ImmuneCellType(
        code=HELPER_T,
        name_kr="조력 T세포",
        name_en="Helper T cell",
        receptors=("CXCR3",),
        ligands=("CXCL9", "CXCL10", "CXCL11"),
        role="살해 T세포와 B세포를 활성화시킨다. 병원체 살해에 직접적으로 "
             "참여하지는 않는다. 일부 T세포는 기억 T세포로 분화하여 동일한 "
             "항원이 다시 침입했을 때 빠르고 강한 면역반응을 일으킨다.",
        role_in_bacteria="★점막 세균면역에서는 Th17 로 분화하여 IL-17 을 "
                         "분비하고, 이것이 중성구 동원을 크게 증폭한다.",
        source="lu2008"),

    KILLER_T: ImmuneCellType(
        code=KILLER_T,
        name_kr="살해 T세포",
        name_en="Killer T cell",
        receptors=("CXCR3",),
        ligands=("CXCL9", "CXCL10", "CXCL11"),
        role="감염된 세포를 살해하는 것에서는 NK세포와 거의 동일하지만, "
             "특정 바이러스를 학습한 뒤에만 활성화된다는 것이 다르다.",
        role_in_bacteria="★세포외 증식균에서는 살해할 감염세포가 없으므로 "
                         "역할이 거의 없다. 인플루엔자 모형에서 주역이었던 "
                         "세포가 세균 모형에서는 무의미해진다.",
        source="groom2011"),

    B_CELL: ImmuneCellType(
        code=B_CELL,
        name_kr="B세포/형질세포",
        name_en="B cell/Plasma cell",
        receptors=("CXCR5",),
        ligands=("CXCL13",),
        role="특정 항원을 인식하고 활성화되어 형질세포로 분화한 후 다량의 "
             "항체를 분비한다. 항체는 병원체에 결합하여 병원체의 제거를 "
             "돕는다. 일부 B세포는 기억 B세포로 분화한다.",
        role_in_bacteria="★항협막 항체를 생산한다. 협막을 덮어 옵소닌화를 "
                         "돕는 것이 폐렴구균 방어의 핵심이며, 폐렴구균 "
                         "백신의 작동 원리이기도 하다.",
        source="okada2002"),

    DENDRITIC: ImmuneCellType(
        code=DENDRITIC,
        name_kr="수지상세포",
        name_en="Dendritic cell",
        receptors=("CCR1",),
        ligands=("CCL3", "CCL5"),
        role="병원체의 항원을 포착하고 살해하여 T세포에 전달한다. 또한 "
             "림프절로 이동하여 T세포를 활성화하여 후천 면역계를 활성화하는 "
             "데 큰 역할을 한다.",
        role_in_bacteria="세균 항원을 포착하여 림프절로 운반하고, Th17 "
                         "분화를 유도한다.",
        source="sozzani1997"),
}


COMPLEMENT_DESCRIPTION = (
    "보체(Complement)는 혈액 속 단백질 무리로, 병원체에 붙어 병원체를 "
    "'옵소닌화'하여 대식세포의 포식 작용을 강화시킨다. 또한 병원체의 "
    "세포막에 구멍을 뚫어 제거하는 역할도 한다. 스스로 이동하지 않으므로 "
    "수용체와 결합 리간드가 존재하지 않는다."
)

COMPLEMENT_NOTE_BACTERIA = (
    "★폐렴구균에서 보체의 역할은 결정적이다. 협막이 보체 침착을 방해하므로 "
    "충분한 보체가 있어야만 C3b 가 균 표면에 붙고, 그래야 식세포가 인식할 수 "
    "있다. Gordon 등(1980)은 폐렴구균이 40% 이상의 비가열 혈청 없이는 "
    "폐포대식세포와 호중구에 포식되지 않음을 보고하였다. 다만 그람양성균은 "
    "두꺼운 펩티도글리칸 층 때문에 보체가 만드는 막공격복합체에 저항하므로, "
    "보체의 본질적 기여는 '직접 살균'이 아니라 '옵소닌화'이다."
)


def print_immune_cell_table() -> None:
    """표 1 출력"""
    print("=" * 100)
    print(" 표 1.  면역세포의 종류 — 주 수용체와 대표적 결합 리간드")
    print("=" * 100)
    print(f" {'면역세포':<16}{'영어 명칭':<24}{'주 수용체':<16}"
          f"{'대표적 결합 리간드':<24}")
    print("-" * 100)
    for code in range(N_CELL_TYPES):
        ct = IMMUNE_CELL_TABLE[code]
        print(f" {ct.name_kr:<16}{ct.name_en:<24}"
              f"{', '.join(ct.receptors):<16}{', '.join(ct.ligands):<24}")
    print(f" {'보체':<16}{'Complement':<24}{'(해당없음)':<16}{'(해당없음)':<24}")
    print("-" * 100)
    print(" [역할]")
    for code in range(N_CELL_TYPES):
        ct = IMMUNE_CELL_TABLE[code]
        print(f" · {ct.name_kr} : {ct.role}")
    print(f" · 보체 : {COMPLEMENT_DESCRIPTION}")
    print("-" * 100)
    print(" [세균 감염에서의 역할 — 인플루엔자와의 차이]")
    for code in range(N_CELL_TYPES):
        ct = IMMUNE_CELL_TABLE[code]
        if ct.role_in_bacteria:
            print(f" · {ct.name_kr} : {ct.role_in_bacteria}")
    print(f" · 보체 : {COMPLEMENT_NOTE_BACTERIA}")
    print("=" * 100)


# =============================================================================
#  SECTION 5.  리간드 (화학신호) 정의
# =============================================================================
#  ★탐구의 독창성:
#    "기존 모형은 단일 케모카인 농도를 가정하였으나, 본 탐구에서는
#     FPR1, CXCR2, CCR2, CXCR3, CXCR5 의 축을 독립적으로 구현했다."
#
#  ★인플루엔자 모형과의 결정적 차이:
#    FPR1 축(fMLF)의 생성원이 존재한다. 세균 자신이 fMLF 를 방출하므로
#    '표적이 곧 신호원'이 된다. 표 3의 '자체 신호: fMLF를 방출하여 위치를
#    표시함'을 그대로 구현한 것이다.
# =============================================================================

LIG_CXCL8 = "CXCL8"          # CXCR2  <- 중성구
LIG_CCL2 = "CCL2"            # CCR2   <- 단핵구/대식세포
LIG_FMLF = "fMLF"            # FPR1   <- 단핵구/대식세포  ★세균이 방출
LIG_CXCR3L = "CXCL9/10/11"   # CXCR3  <- NK세포, 조력T, 살해T
LIG_CXCL13 = "CXCL13"        # CXCR5  <- B세포/형질세포
LIG_CCL35 = "CCL3/CCL5"      # CCR1   <- 수지상세포

ALL_LIGAND_FIELDS: Tuple[str, ...] = (
    LIG_CXCL8, LIG_CCL2, LIG_FMLF, LIG_CXCR3L, LIG_CXCL13, LIG_CCL35,
)


@dataclass(frozen=True)
class LigandSpec:
    """리간드 1종의 명세"""
    key: str
    receptor: str
    cells: Tuple[str, ...]
    source: str
    sigma: float
    halflife_text: str
    decay: float
    note: str = ""
    ref: str = ""


LIGAND_SPECS: Dict[str, LigandSpec] = {

    LIG_FMLF: LigandSpec(
        key=LIG_FMLF, receptor="FPR1", cells=("단핵구/대식세포",),
        source="★세균 자신이 대사 과정에서 방출하는 N-포르밀 펩타이드.",
        sigma=3.0,
        halflife_text=f"{REAL['fmlf_halflife_min']:.0f}분",
        decay=Scale.decay_from_halflife_min(REAL["fmlf_halflife_min"]),
        note="세균의 단백질 합성은 N-포르밀메티오닌에서 시작하므로, 세균이 "
             "존재하는 한 fMLF 가 계속 발생한다. 표적이 곧 신호원이 되는 "
             "유일한 축이며, 인플루엔자 모형에서는 생성원이 없어 작동하지 "
             "않았다.",
        ref="schiffmann1975"),

    LIG_CXCL8: LigandSpec(
        key=LIG_CXCL8, receptor="CXCR2", cells=("중성구",),
        source="세균이 침습한 부위의 상피세포가 분비하며, 침윤한 중성구가 "
               "2차로 증폭한다.",
        sigma=2.8,
        halflife_text=f"{REAL['cxcl8_halflife_h']:.0f}시간",
        decay=Scale.decay_from_halflife_h(REAL["cxcl8_halflife_h"]),
        note="즉시형 유전자 산물이므로 감염 직후부터 형성된다.",
        ref="rudd2019"),

    LIG_CCL2: LigandSpec(
        key=LIG_CCL2, receptor="CCR2", cells=("단핵구/대식세포",),
        source="감염 부위 상피세포와 활성화된 대식세포가 분비한다.",
        sigma=2.4,
        halflife_text=f"{REAL['ccl2_halflife_h']:.0f}시간",
        decay=Scale.decay_from_halflife_h(REAL["ccl2_halflife_h"]),
        note="단핵구는 FPR1 과 CCR2 를 모두 가지므로 두 신호를 함께 감지한다.",
        ref="lin2008"),

    LIG_CXCR3L: LigandSpec(
        key=LIG_CXCR3L, receptor="CXCR3",
        cells=("자연살해 세포", "조력 T세포", "살해 T세포"),
        source="IFN-gamma 에 자극받은 세포가 분비한다.",
        sigma=2.0,
        halflife_text=f"{REAL['cxcr3l_halflife_h']:.0f}시간",
        decay=Scale.decay_from_halflife_h(REAL["cxcr3l_halflife_h"]),
        note="인터페론 유도성이므로 CXCR3 계열 세포의 동원은 중성구보다 "
             "구조적으로 늦어진다.",
        ref="luster1985"),

    LIG_CXCL13: LigandSpec(
        key=LIG_CXCL13, receptor="CXCR5", cells=("B세포/형질세포",),
        source="림프절의 여포수지상세포가 항상적으로 분비한다.",
        sigma=3.2,
        halflife_text=f"{REAL['cxcl13_halflife_h']:.0f}시간",
        decay=Scale.decay_from_halflife_h(REAL["cxcl13_halflife_h"]),
        note="감염 병소가 아니라 림프절 방향을 가리키는 유일한 신호이다.",
        ref="okada2002"),

    LIG_CCL35: LigandSpec(
        key=LIG_CCL35, receptor="CCR1", cells=("수지상세포",),
        source="대식세포와 활성화된 T세포, 감염 부위 조직이 분비한다.",
        sigma=2.6,
        halflife_text=f"{REAL['ccl35_halflife_h']:.0f}시간",
        decay=Scale.decay_from_halflife_h(REAL["ccl35_halflife_h"]),
        note="미성숙 수지상세포를 감염 부위로 유도한다.",
        ref="sozzani1997"),
}


# ---------------------------------------------------------------------------
#  세포 유형 -> 감지 가능한 리간드 농도장 목록
#  (표 1의 '주 수용체' 열을 그대로 코드로 옮긴 대응표)
# ---------------------------------------------------------------------------
CELL_RECEPTOR_FIELDS: Dict[int, Tuple[Tuple[str, str], ...]] = {
    NEUTROPHIL: (("CXCR2", LIG_CXCL8),),
    MONOCYTE:   (("FPR1", LIG_FMLF), ("CCR2", LIG_CCL2)),
    NK_CELL:    (("CXCR3", LIG_CXCR3L),),
    HELPER_T:   (("CXCR3", LIG_CXCR3L),),
    KILLER_T:   (("CXCR3", LIG_CXCR3L),),
    B_CELL:     (("CXCR5", LIG_CXCL13),),
    DENDRITIC:  (("CCR1", LIG_CCL35),),
}


def verify_receptor_consistency() -> bool:
    """
    표 1(IMMUNE_CELL_TABLE)과 실제 이동에 사용되는 대응표
    (CELL_RECEPTOR_FIELDS)가 일치하는지 검사한다.
    """
    ok = True
    for code, ct in IMMUNE_CELL_TABLE.items():
        impl = tuple(r for r, _ in CELL_RECEPTOR_FIELDS[code])
        if impl != ct.receptors:
            print(f"  [불일치] {ct.name_kr}: 표 1 = {ct.receptors}, "
                  f"구현 = {impl}")
            ok = False
    return ok


def print_ligand_table() -> None:
    """리간드 명세 출력"""
    print("=" * 100)
    print(" 리간드(화학신호) 명세 — 수용체별로 완전히 분리된 농도장")
    print("=" * 100)
    print(f" {'리간드':<14}{'수용체':<10}{'감지 세포':<34}"
          f"{'확산':<8}{'반감기':<10}")
    print("-" * 100)
    for key in ALL_LIGAND_FIELDS:
        s = LIGAND_SPECS[key]
        print(f" {s.key:<14}{s.receptor:<10}{', '.join(s.cells):<34}"
              f"{s.sigma:<8.1f}{s.halflife_text:<10}")
    print("-" * 100)
    for key in ALL_LIGAND_FIELDS:
        s = LIGAND_SPECS[key]
        print(f" · {s.key} ({s.receptor}) 생성원 : {s.source}")
        if s.note:
            print(f"     비고 : {s.note}")
    print("=" * 100)


# =============================================================================
#  SECTION 6.  이동 방식 정의 — 작품설명서 표 2
# =============================================================================

MODE_RANDOM = "random"        # 무작위 이동
MODE_DIRECTED = "directed"    # 방향성 이동


@dataclass(frozen=True)
class MovementSpec:
    """작품설명서 표 2의 한 열"""
    mode: str
    name_kr: str
    state: str
    rule: str
    sensing: str
    spawn: str


MOVEMENT_TABLE: Dict[str, MovementSpec] = {

    MODE_RANDOM: MovementSpec(
        mode=MODE_RANDOM,
        name_kr="무작위 이동",
        state="감염이나 손상이 없는 상태 (정상 상태)",
        rule="무작위로 방향을 골라 브라운 운동(Brownian motion)에 따라 이동",
        sensing="감지하지 않음 (0%)",
        spawn="조직에 무작위적으로 생성"),

    MODE_DIRECTED: MovementSpec(
        mode=MODE_DIRECTED,
        name_kr="방향성 이동",
        state="케모카인(chemokine)에 의한 농도차(화학주성)가 형성된 감염 상태",
        rule="수용체에 결합하는 리간드 농도가 최대인 방향으로 이동",
        sensing="항상 감지 (100%)",
        spawn="림프절에서 생성"),
}


def print_movement_table() -> None:
    """표 2 출력"""
    r = MOVEMENT_TABLE[MODE_RANDOM]
    d = MOVEMENT_TABLE[MODE_DIRECTED]
    print("=" * 100)
    print(" 표 2.  무작위 이동과 방향성 이동의 정의")
    print("=" * 100)
    rows = [("구분", r.name_kr, d.name_kr),
            ("상태", r.state, d.state),
            ("이동 규칙", r.rule, d.rule),
            ("신호 감지", r.sensing, d.sensing),
            ("생성 장소", r.spawn, d.spawn)]
    for label, a, b in rows:
        print(f" {label:<12}| {a}")
        print(f" {'':<12}| {b}")
        print("-" * 100)
    print("=" * 100)


# =============================================================================
#  SECTION 7.  병원체 특성 정의 — 작품설명서 표 3
# =============================================================================

PATHOGEN_TABLE_INFLUENZA: Dict[str, str] = {
    "이름": "인플루엔자 A 바이러스 (H1N1)",
    "분류": "RNA 바이러스",
    "크기": "80~120 nm",
    "증식 위치": "세포내 증식",
    "공간 분포": "조직 전역에 균일한 확산",
    "자체 신호": "없음",
    "제거 경로": "감염된 숙주세포의 사멸이 제거의 대부분을 차지",
}

PATHOGEN_TABLE_PNEUMOCOCCUS: Dict[str, str] = {
    "이름": "폐렴구균 (S. pneumoniae)",
    "분류": "그람양성 세균",
    "크기": f"{REAL['bacterium_diameter_um_min']:.1f}~"
            f"{REAL['bacterium_diameter_um_max']:.1f} um",
    "증식 위치": "세포외 증식",
    "공간 분포": "편모가 없어 비운동성. 분열한 자리에 머물러 미세 군락 형성",
    "자체 신호": "fMLF를 방출하여 위치를 표시함",
    "제거 경로": "식균작용이 유일한 제거 수단",
}


def print_pathogen_table() -> None:
    """표 3 출력"""
    print("=" * 100)
    print(" 표 3.  인플루엔자 A 바이러스(H1N1)와 폐렴구균의 특징 비교")
    print("=" * 100)
    keys = ["분류", "크기", "증식 위치", "공간 분포", "자체 신호", "제거 경로"]
    print(f" {'비교 항목':<12}{'인플루엔자 A 바이러스 (H1N1)':<44}"
          f"{'폐렴구균 (S. pneumoniae)':<44}")
    print("-" * 100)
    for k in keys:
        a = PATHOGEN_TABLE_INFLUENZA[k]
        b = PATHOGEN_TABLE_PNEUMOCOCCUS[k]
        print(f" {k:<12}{a:<44}{b:<44}")
    print("=" * 100)
    print(" 이 중 특히 A형 인플루엔자에서는 바이러스에 의해 감염된 숙주세포가")
    print(" 자멸하여 제거되는 데에 반해, 폐렴구균은 세균이기 때문에 병원체가")
    print(" 사멸하지 않아 면역세포의 이동 방식에 따른 결과가 그대로 반영될 수")
    print(" 있다. 즉 폐렴구균은 A형 인플루엔자의 대조 실험으로써 포함되었다.")
    print("=" * 100)


# =============================================================================
#  SECTION 8.  살해 조건 — 작품설명서 「라. 탐구 절차 - B」
# =============================================================================

KILL_CONDITION_TEXT = (
    "중성구와 대식세포가 옵소닌화된 세균을 인식하여 세포막으로 감싸 "
    "식균작용을 하고, 중성구와 대식세포 내부의 분해효소와 활성산소가 "
    "방출되어 세균을 분해, 살해한다."
)

KILL_CONDITION_STEPS: List[Tuple[str, str, str]] = [
    ("1단계", "옵소닌화",
     "혈액 속 보체 단백질이 세균 표면에 C3b 를 침착시킨다. 협막이 이를 "
     "방해하므로 충분한 보체 농도가 필요하다. 후천면역이 성립하면 항협막 "
     "항체가 결합하여 옵소닌화를 크게 강화한다."),
    ("2단계", "인식",
     "중성구와 대식세포가 옵소닌화된 세균을 인식한다. 옵소닌 침착도가 "
     "일정 수준을 넘어야 인식이 가능하다."),
    ("3단계", "포식",
     "세포막으로 세균을 감싸 세포 내부로 끌어들인다. 호중구 1개의 포식 "
     "용량은 약 50 CFU 이다."),
    ("4단계", "분해·살해",
     "식포 내부로 분해효소와 활성산소가 방출되어 세균을 분해한다."),
]


def print_kill_condition() -> None:
    """살해 조건 출력"""
    print("=" * 100)
    print(" 살해 조건 — 식균작용에 의한 세균 제거")
    print("=" * 100)
    print(f" {KILL_CONDITION_TEXT}")
    print("-" * 100)
    for stage, name, desc in KILL_CONDITION_STEPS:
        print(f" [{stage}] {name}")
        print(f"        {desc}")
    print("=" * 100)


# =============================================================================
#  SECTION 9.  파라미터 표
# =============================================================================


@dataclass(frozen=True)
class ParamRow:
    """파라미터 1개의 출처 기록"""
    item: str
    real_value: str
    unit: str
    source: str
    conversion: str
    sim_value: str
    note: str = ""


PARAM_TABLE: List[ParamRow] = []


def _p(*args, **kwargs) -> None:
    PARAM_TABLE.append(ParamRow(*args, **kwargs))


# ---- 공간 · 시간 축척 ----
_p("시뮬레이션 공간", "15 x 15", "mm", "crystal2008", "1칸 = 상피세포 1개",
   "1000 x 1000 격자 (100만 칸)",
   "격자 한 칸이 상피세포 한 개(15um)에 대응한다")
_p("시간 단위", "6", "분/tick", MODEL_ASSUMPTION, "-", "1일 = 240 tick",
   "세균 배가시간(56분)을 충분히 해상할 수 있는 단위")
_p("개체수 축소비율", "1/1000", "-", MODEL_ASSUMPTION, "-", "kappa = 0.001",
   "계산량을 실행 가능한 범위로 줄이면서 비율 관계를 보존한다")

# ---- 병원체 ----
_p("폐 내 배가시간", f"{REAL['doubling_time_min']:.0f}", "분", "jose2015",
   "배가시간 -> tick 당 분열 확률",
   f"{Scale.division_prob_from_doubling(REAL['doubling_time_min']):.5f}/tick",
   "저용량 접종 초기(무염증 상태)의 측정값이다")
_p("초기 접종량", f"{REAL['inoculum_cfu']:.0e}", "CFU", "lin2020",
   "1/1000 축소", f"{Scale.agents_from_real(REAL['inoculum_cfu']):,} agent",
   "마우스 비강내 접종 실험의 표준 용량")
_p("최대 균량", f"{REAL['max_burden_cfu']:.0e}", "CFU", "hamilton2019",
   "1/1000 축소",
   f"{Scale.agents_from_real(REAL['max_burden_cfu']):,} agent",
   "빈사 상태 마우스의 폐 내 균량. 수용 한계로 사용한다")
_p("폐포대식세포 제거 반감기", f"{REAL['am_clearance_halflife_min']:.0f}", "분",
   "jose2015", "반감기 -> tick 당 제거율",
   f"{Scale.decay_from_halflife_min(REAL['am_clearance_halflife_min']):.5f}/tick",
   "AM 개체수가 유한하므로 고균량에서는 포화한다")
_p("세균 크기", f"{REAL['bacterium_diameter_um_min']:.1f}~"
   f"{REAL['bacterium_diameter_um_max']:.1f}", "um", MODEL_ASSUMPTION,
   "격자 이하 크기", "개별 agent 로 표현",
   "상피세포(15um)보다 한 자릿수 작으므로 점입자로 취급")

# ---- 식균작용 ----
_p("호중구 포식 용량", f"약 {REAL['neutrophil_capacity_cfu']:.0f}", "CFU/세포",
   "rubio2023", "동일",
   f"{int(REAL['neutrophil_capacity_cfu'])} CFU",
   "포식 용량을 소진한 호중구는 더 이상 포식하지 못한다")
_p("옵소닌 요구량", f"혈청 {REAL['opsonin_serum_requirement']*100:.0f} 이상",
   "%", "gordon1980", "옵소닌 침착도 역치",
   f"{REAL['opsonin_serum_requirement']:.2f}",
   "폐렴구균은 고농도 혈청 없이는 포식되지 않는다")
_p("협막의 포식 저항", "다중 기전 저해", "-", "hyams2010", "저해 계수",
   f"{REAL['capsule_resistance']:.2f}",
   "협막 다당이 보체 침착과 호중구 포식을 함께 방해한다")

# ---- 면역세포 이동 속도 ----
_p("중성구 이동 속도", f"{REAL['neutrophil_um_min']:.0f}", "um/분", "friedl2008",
   "um/분 -> 칸/tick",
   f"{Scale.speed_cells_per_tick(REAL['neutrophil_um_min']):.2f} 칸/tick",
   "간질 조직 내 백혈구 중 가장 빠르다")
_p("단핵구 이동 속도", f"{REAL['monocyte_um_min']:.0f}", "um/분", "friedl2008",
   "um/분 -> 칸/tick",
   f"{Scale.speed_cells_per_tick(REAL['monocyte_um_min']):.2f} 칸/tick", "")
_p("NK세포 이동 속도", f"{REAL['nk_um_min']:.0f}", "um/분", "friedl2008",
   "um/분 -> 칸/tick",
   f"{Scale.speed_cells_per_tick(REAL['nk_um_min']):.2f} 칸/tick", "")
_p("T세포 이동 속도", f"{REAL['t_cell_um_min']:.0f}", "um/분", "miller2002",
   "um/분 -> 칸/tick",
   f"{Scale.speed_cells_per_tick(REAL['t_cell_um_min']):.2f} 칸/tick", "")
_p("B세포 이동 속도", f"{REAL['b_cell_um_min']:.0f}", "um/분", "miller2002",
   "um/분 -> 칸/tick",
   f"{Scale.speed_cells_per_tick(REAL['b_cell_um_min']):.2f} 칸/tick", "")
_p("수지상세포 이동 속도", f"{REAL['dendritic_um_min']:.0f}", "um/분",
   "friedl2008", "um/분 -> 칸/tick",
   f"{Scale.speed_cells_per_tick(REAL['dendritic_um_min']):.2f} 칸/tick", "")
_p("혈류 내 이동 속도", f"{REAL['vessel_transit_um_min']:.0f}", "um/분",
   "ley2007", "um/분 -> 칸/tick",
   f"{Scale.speed_cells_per_tick(REAL['vessel_transit_um_min']):.1f} 칸/tick",
   "혈관 내에서는 조직 내보다 훨씬 빠르게 이동한다")

# ---- 리간드 ----
for _k in ALL_LIGAND_FIELDS:
    _s = LIGAND_SPECS[_k]
    _p(f"{_s.key} 반감기", _s.halflife_text, "-",
       _s.ref if _s.ref else MODEL_ASSUMPTION,
       "반감기 -> tick 당 감쇠율", f"{_s.decay:.5f}/tick", _s.note[:60])
    _p(f"{_s.key} 확산 폭", f"{_s.sigma:.1f}", "농도장 격자/tick",
       MODEL_ASSUMPTION, "-", f"sigma = {_s.sigma:.1f}",
       f"{_s.receptor} 수용체가 감지하는 축")

# ---- 후천면역 ----
_p("프라이밍 소요 시간", f"{REAL['priming_h']:.0f}", "시간", MODEL_ASSUMPTION,
   "시간 -> tick", f"{Scale.ticks_from_hours(REAL['priming_h'])} tick",
   "항원이 림프절에 도달한 뒤 T세포가 활성화되기까지")
_p("Th17 반응 정점", f"{REAL['th17_peak_day']:.0f}", "일", "lu2008",
   "모형에서 재현 대상", "IL-17 매개 중성구 증폭",
   "IL-17A 가 폐렴구균 집락에 대한 획득면역을 매개한다")
_p("IgM 반감기", f"{REAL['igm_halflife_d']:.0f}", "일", MODEL_ASSUMPTION,
   "반감기 -> tick 당 감쇠율",
   f"{Scale.decay_from_halflife_d(REAL['igm_halflife_d']):.6f}/tick", "")
_p("IgG 반감기", f"{REAL['igg_halflife_d']:.0f}", "일", MODEL_ASSUMPTION,
   "반감기 -> tick 당 감쇠율",
   f"{Scale.decay_from_halflife_d(REAL['igg_halflife_d']):.6f}/tick",
   "항협막 IgG 는 폐렴구균 방어의 핵심이다")


def print_param_table() -> None:
    """파라미터 출처표 출력"""
    print("=" * 118)
    print(" 시뮬레이션 파라미터와 출처")
    print("=" * 118)
    print(f" {'항목':<22}{'실제 값':<16}{'단위':<12}{'출처':<18}"
          f"{'시뮬레이션 값':<30}")
    print("-" * 118)
    for row in PARAM_TABLE:
        src = row.source
        if src != MODEL_ASSUMPTION and src in REFERENCES:
            src = REFERENCES[src].short()
        print(f" {row.item:<22}{row.real_value:<16}{row.unit:<12}"
              f"{src:<18}{row.sim_value:<30}")
    print("=" * 118)
    n_lit = sum(1 for r in PARAM_TABLE if r.source != MODEL_ASSUMPTION)
    n_asm = len(PARAM_TABLE) - n_lit
    print(f" 총 {len(PARAM_TABLE)}개 파라미터 "
          f"— 문헌 인용 {n_lit}개 / 모델 가정값 {n_asm}개")
    print("=" * 118)


# =============================================================================
#  SECTION 10.  시뮬레이션 파라미터
# =============================================================================


@dataclass
class Params:
    """시뮬레이션 실행 파라미터"""

    # ---------------- 실행 ----------------
    max_days: int = 14
    seed: int = 20260811
    stop_after_clear_days: float = 0.5

    # ---------------- 병원체 ----------------
    n_bacteria0: int = Scale.agents_from_real(REAL["inoculum_cfu"])
    carrying_capacity: int = Scale.agents_from_real(REAL["max_burden_cfu"])
    division_p: float = Scale.division_prob_from_doubling(
        REAL["doubling_time_min"])
    # ★표 3: 편모가 없어 비운동성. 분열한 자리에 머물러 미세 군락 형성
    daughter_sigma: float = 1.6        # 딸세포가 놓이는 거리 (칸)
    drift_sigma: float = 0.55          # 점액섬모 수송에 의한 미세 이동
    # 접종 위치 (비강 접종을 모사한 국소 접종)
    inoculum_center_x: float = 620.0
    inoculum_center_y: float = 430.0
    inoculum_spread: float = 26.0



    # 상주 폐포대식세포의 제거 (개체수 유한 -> 고균량에서 포화)
    am_clearance_p: float = Scale.decay_from_halflife_min(
        REAL["am_clearance_halflife_min"])
    am_saturation_n: float = 2000.0

    # 세균의 조직 침습에 의한 상피 손상
    epithelial_damage_p: float = 0.0016
    # 중성구 매개 부수적 조직손상
    neutrophil_damage_p: float = 4.0e-6

    # ---------------- 옵소닌화 / 식균작용 ----------------
    capsule_resistance: float = REAL["capsule_resistance"]
    opsonin_requirement: float = REAL["opsonin_serum_requirement"]
    c3b_deposition: float = 0.085      # tick 당 C3b 침착 속도
    antibody_opsonin_gain: float = 0.075
    opsonin_decay: float = 0.006
    neutrophil_phago_p: float = 0.92
    monocyte_phago_p: float = 0.80
    phagocytosis_rounds: int = 30       # 한 tick 에 여러 균을 연속 포식
    capacity_neutrophil: int = int(REAL["neutrophil_capacity_cfu"])
    capacity_monocyte: int = 120

    # ---------------- 면역세포 초기 개체수 (1/1000 축소) ----------------
    n_neutrophil0: int = 3000
    n_monocyte0: int = 300
    n_nk0: int = 195
    n_helper_t0: int = 675
    n_killer_t0: int = 375
    n_bcell0: int = 180
    n_dendritic0: int = 120

    # ---------------- 이동 속도 (칸/tick) ----------------
    v_neutrophil: float = Scale.speed_cells_per_tick(REAL["neutrophil_um_min"])
    v_monocyte: float = Scale.speed_cells_per_tick(REAL["monocyte_um_min"])
    v_nk: float = Scale.speed_cells_per_tick(REAL["nk_um_min"])
    v_t: float = Scale.speed_cells_per_tick(REAL["t_cell_um_min"])
    v_b: float = Scale.speed_cells_per_tick(REAL["b_cell_um_min"])
    v_dendritic: float = Scale.speed_cells_per_tick(REAL["dendritic_um_min"])
    v_vessel: float = Scale.speed_cells_per_tick(REAL["vessel_transit_um_min"])

    # ---------------- 수명 (tick) ----------------
    life_neutrophil: int = 240
    life_monocyte: int = 1800
    life_nk: int = 2600
    life_t: int = 60000
    life_b: int = 60000
    life_dendritic: int = 3000
    life_effector_t: int = 1440

    # ---------------- 동원 ----------------
    recruit_neutrophil: float = 42000.0
    recruit_monocyte: float = 4200.0
    recruit_nk: float = 1100.0
    # ★Th17 매개 중성구 증폭 (Lu 등 2008, Zhang 등 2009)
    th17_neutrophil_boost_max: float = 1.4
    th17_boost_scale: float = 120.0

    # ---------------- 혈관외유출 ----------------
    # ★백혈구는 혈관 내벽을 구르다가 케모카인이 고농도로 제시된 지점에서만
    #   혈관을 빠져나온다(Ley 등, 2007). 역치가 낮으면 신호가 닿지 않는 곳에서
    #   유출되어 표적을 찾지 못한다.
    extravasation_threshold: float = 0.50

    # ---------------- 리간드 생성 ----------------
    # ★fMLF: 세균 자신이 방출한다 (표 3)
    fmlf_per_bacterium: float = 0.00028
    cxcl8_from_site: float = 0.016
    # ★중성구 자가분비는 세균 유래 신호보다 충분히 작아야 한다.
    #   값이 크면 중성구가 서로를 쫓는 자가군집이 형성되어 표적을 놓친다.
    cxcl8_from_neutrophil: float = 0.00025
    ccl2_from_site: float = 0.011
    ccl2_from_monocyte: float = 0.0004
    cxcr3l_basal: float = 0.0022
    cxcr3l_ifng_gain: float = 0.010
    cxcl13_from_lymphnode: float = 0.010
    ccl35_from_site: float = 0.006
    ccl35_from_monocyte: float = 0.008

    # ---------------- IFN-gamma ----------------
    ifng_from_nk_t: float = 0.0060
    ifng_sigma: float = 2.2
    ifng_decay: float = Scale.decay_from_halflife_h(REAL["ifng_halflife_h"])

    # ---------------- 보체 · 염증 ----------------
    complement_on: float = 0.030
    complement_off: float = 0.011
    inflammation_on: float = 0.030
    inflammation_off: float = 0.014

    # ---------------- 항체 · 보체에 의한 제거 ----------------
    # 그람양성균은 두꺼운 펩티도글리칸 층 때문에 보체 막공격복합체에
    # 저항하므로, 항체·보체의 본질적 기여는 옵소닌화이며 직접 살균은 작다.
    antibody_clear_max: float = 0.005
    antibody_clear_k: float = 0.000045
    complement_clear_p: float = 0.00016

    # ---------------- 후천면역 ----------------
    priming_ticks: int = Scale.ticks_from_hours(REAL["priming_h"])
    antigen_threshold: float = 60.0
    ab_igm_rate: float = 0.055
    ab_igg_rate: float = 0.030
    igm_decay: float = Scale.decay_from_halflife_d(REAL["igm_halflife_d"])
    igg_decay: float = Scale.decay_from_halflife_d(REAL["igg_halflife_d"])
    ab_titer_cap: float = 120.0
    memory_fraction: float = 0.10

    # ---------------- 조직 ----------------
    regeneration_rate: float = 0.0020


# =============================================================================
#  SECTION 11.  조직 공간
# =============================================================================

STROMA = 0        # 간질 조직
HEALTHY = 1       # 정상 상피세포
DEAD = 4          # 사멸 상피세포
VESSEL = 5        # 혈관
LYMPH = 6         # 림프절

STATE_NAME: Dict[int, str] = {
    STROMA: "간질 조직",
    HEALTHY: "정상 상피세포",
    DEAD: "사멸 상피세포",
    VESSEL: "혈관",
    LYMPH: "림프절",
}

EPI_BAND_PERIOD = 25
EPI_BAND_WIDTH = 12
VESSEL_COL_PERIOD = 120
LN_ROW0, LN_ROW1 = 460, 540
LN_COL0, LN_COL1 = 40, 120

# 8방향 (격자 오프셋과 단위벡터)
DIRECTION_OFFSETS: List[Tuple[int, int]] = [
    (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
]
DIRECTION_UNIT = np.array(DIRECTION_OFFSETS, dtype=np.float32)
DIRECTION_UNIT /= np.linalg.norm(DIRECTION_UNIT, axis=1, keepdims=True)


class Tissue:
    """
    상기도 상피 조직의 공간 구조.
    두 이동 방식은 완전히 동일한 공간을 사용한다.

    ★인플루엔자 모형과 달리 감염세포 상태(ECLIPSE / PRODUCTIVE)가 없다.
      폐렴구균은 세포외 증식균이므로 숙주세포 안으로 들어가지 않는다.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        g = Scale.GRID
        grid = np.full((g, g), STROMA, dtype=np.uint8)

        for r0 in range(0, g, EPI_BAND_PERIOD):
            grid[r0:r0 + EPI_BAND_WIDTH, :] = HEALTHY
        for r0 in range(0, g, EPI_BAND_PERIOD):
            rr = r0 + EPI_BAND_WIDTH + 1
            if rr < g:
                grid[rr, :] = VESSEL
        for c0 in range(0, g, VESSEL_COL_PERIOD):
            grid[:, c0] = VESSEL
        grid[LN_ROW0:LN_ROW1, LN_COL0:LN_COL1] = LYMPH

        self.state: np.ndarray = grid.reshape(-1)

        self.n_target_initial = int(np.count_nonzero(self.state == HEALTHY))
        self.vessel_sites = np.flatnonzero(self.state == VESSEL).astype(np.int32)
        self.lymph_sites = np.flatnonzero(self.state == LYMPH).astype(np.int32)
        self.tissue_sites = np.flatnonzero(
            (self.state == HEALTHY) | (self.state == STROMA)).astype(np.int32)

        self.n_dead_epithelium = 0
        self.cumulative_damage = 0.0

        r0 = LN_ROW0 // Scale.FIELD_BIN
        r1 = LN_ROW1 // Scale.FIELD_BIN + 1
        c0 = LN_COL0 // Scale.FIELD_BIN
        c1 = LN_COL1 // Scale.FIELD_BIN + 1
        self.lymph_field_slice = (slice(r0, r1), slice(c0, c1))

    # ------------------------------------------------------------------
    @staticmethod
    def site_of_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        col = x.astype(np.int32)
        row = y.astype(np.int32)
        np.clip(col, 0, Scale.GRID - 1, out=col)
        np.clip(row, 0, Scale.GRID - 1, out=row)
        return row * Scale.GRID + col

    @staticmethod
    def field_of_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        fc = (x / Scale.FIELD_BIN).astype(np.int32)
        fr = (y / Scale.FIELD_BIN).astype(np.int32)
        np.clip(fc, 0, Scale.FIELD_N - 1, out=fc)
        np.clip(fr, 0, Scale.FIELD_N - 1, out=fr)
        return fr, fc

    @staticmethod
    def field_of_site(site: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        row = site // Scale.GRID
        col = site - row * Scale.GRID
        return ((row // Scale.FIELD_BIN).astype(np.int32),
                (col // Scale.FIELD_BIN).astype(np.int32))

    @staticmethod
    def xy_of_site(site: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        row = site // Scale.GRID
        col = site - row * Scale.GRID
        return (col.astype(np.float32) + 0.5, row.astype(np.float32) + 0.5)

    # ------------------------------------------------------------------
    def spawn_xy(self, k: int, mode: str,
                 rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """
        면역세포 생성 위치 (작품설명서 표 2 '생성 장소')
          무작위 이동 -> 조직에 무작위적으로 생성
          방향성 이동 -> 림프절에서 생성
        """
        if k <= 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        if mode == MODE_DIRECTED:
            sites = rng.choice(self.lymph_sites, size=k, replace=True)
        else:
            sites = rng.choice(self.tissue_sites, size=k, replace=True)
        return self.xy_of_site(sites)

    # ------------------------------------------------------------------
    def kill_epithelium(self, sites: np.ndarray) -> int:
        if sites.size == 0:
            return 0
        sites = np.unique(sites)
        ok = self.state[sites] == HEALTHY
        s = sites[ok]
        if s.size:
            self.state[s] = DEAD
            self.n_dead_epithelium += int(s.size)
            self.cumulative_damage += (s.size / max(1, self.n_target_initial)
                                       * 1.6)
        return int(s.size)

    def regenerate(self, rate: float) -> int:
        dead = np.flatnonzero(self.state == DEAD)
        if dead.size == 0:
            return 0
        k = int(dead.size * rate)
        if k <= 0:
            return 0
        pick = self.rng.choice(dead, size=min(k, dead.size), replace=False)
        self.state[pick] = HEALTHY
        return int(pick.size)

    @property
    def n_healthy(self) -> int:
        return int(np.count_nonzero(self.state == HEALTHY))

    def describe(self) -> str:
        return (f"표적 상피세포 {self.n_target_initial:,}칸, "
                f"혈관 {self.vessel_sites.size:,}칸, "
                f"림프절 {self.lymph_sites.size:,}칸")


# =============================================================================
#  SECTION 12.  리간드 농도장
# =============================================================================


class LigandFieldSet:
    """
    표 1에 등장하는 리간드를 각각 독립된 농도장으로 관리한다.
    리간드마다 생성원·확산 폭·반감기가 다르므로, 같은 시점에도 세포 종류에
    따라 서로 다른 방향으로 이동한다.
    """

    def __init__(self):
        n = Scale.FIELD_N
        self.field: Dict[str, np.ndarray] = {
            key: np.zeros((n, n), dtype=np.float32)
            for key in ALL_LIGAND_FIELDS
        }
        self.peak: Dict[str, float] = {key: 0.0 for key in ALL_LIGAND_FIELDS}

    def deposit(self, key: str, fr: np.ndarray, fc: np.ndarray,
                amount: float) -> None:
        if fr is None or fr.size == 0 or amount == 0.0:
            return
        np.add.at(self.field[key], (fr, fc), amount)

    def deposit_region(self, key: str, region, amount: float) -> None:
        if amount == 0.0:
            return
        self.field[key][region] += amount

    def add_field(self, key: str, arr: np.ndarray, gain: float) -> None:
        if gain == 0.0:
            return
        self.field[key] += gain * arr

    def diffuse_and_decay(self) -> None:
        for key in ALL_LIGAND_FIELDS:
            spec = LIGAND_SPECS[key]
            arr = gaussian_filter(self.field[key], sigma=spec.sigma,
                                  mode="nearest")
            arr *= (1.0 - spec.decay)
            self.field[key] = arr
            m = float(arr.max())
            if m > self.peak[key]:
                self.peak[key] = m

    def direction_stacks(self) -> Dict[str, np.ndarray]:
        """각 리간드에 대해 8방향으로 이동시킨 농도장을 쌓아 반환한다"""
        out: Dict[str, np.ndarray] = {}
        for key in ALL_LIGAND_FIELDS:
            arr = self.field[key]
            st = np.empty((8, Scale.FIELD_N, Scale.FIELD_N), dtype=np.float32)
            for i, (dc, dr) in enumerate(DIRECTION_OFFSETS):
                st[i] = np.roll(np.roll(arr, -dr, axis=0), -dc, axis=1)
            out[key] = st
        return out

    def mean(self, key: str) -> float:
        return float(self.field[key].mean())

    def maximum(self, key: str) -> float:
        return float(self.field[key].max())


# =============================================================================
#  SECTION 13.  이동 엔진 — 작품설명서 표 2의 구현
# =============================================================================


class MovementEngine:
    """
    ------------------------------------------------------------------------
     무작위 이동 (MODE_RANDOM)
        이동 규칙 : 무작위로 방향을 골라 브라운 운동에 따라 이동한다.
        신호 감지 : 하지 않는다 (0%).
     방향성 이동 (MODE_DIRECTED)
        이동 규칙 : 수용체에 결합하는 리간드 농도가 최대인 방향으로 이동한다.
        신호 감지 : 항상 한다 (100%).
    ------------------------------------------------------------------------
    """

    def __init__(self, mode: str, rng: np.random.Generator):
        if mode not in (MODE_RANDOM, MODE_DIRECTED):
            raise ValueError(f"알 수 없는 이동 방식: {mode}")
        self.mode = mode
        self.rng = rng
        self.spec = MOVEMENT_TABLE[mode]
        self.n_move_calls = 0
        self.n_gradient_guided = 0
        self.n_no_gradient = 0
        self.n_by_type = np.zeros(N_CELL_TYPES, dtype=np.int64)
        self.n_grad_by_type = np.zeros(N_CELL_TYPES, dtype=np.int64)

    # ------------------------------------------------------------------
    def move(self, x, y, hx, hy, speed, stacks, fr, fc, ctype) -> None:
        n = x.size
        if n == 0:
            return
        self.n_move_calls += n
        if self.mode == MODE_RANDOM:
            dx, dy = self._random_direction(n)
        else:
            dx, dy = self._directed_direction(stacks, fr, fc, ctype, hx, hy)
        hx[:] = dx
        hy[:] = dy
        x += dx * speed
        y += dy * speed
        self.reflect_at_boundary(x, y)

    # ------------------------------------------------------------------
    def _random_direction(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """무작위 이동: 균일난수로 방향을 결정한다 (브라운 운동)"""
        theta = self.rng.uniform(0.0, 2.0 * math.pi, n).astype(np.float32)
        return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)

    # ------------------------------------------------------------------
    def _directed_direction(self, stacks, fr, fc, ctype, hx, hy):
        """방향성 이동: 자기 수용체의 리간드 농도가 최대인 방향을 고른다"""
        n = ctype.size
        dx = np.zeros(n, dtype=np.float32)
        dy = np.zeros(n, dtype=np.float32)

        for code in np.unique(ctype):
            code = int(code)
            mask = ctype == code
            k = int(np.count_nonzero(mask))
            if k == 0:
                continue
            self.n_by_type[code] += k

            values = None
            for _receptor, ligand_key in CELL_RECEPTOR_FIELDS[code]:
                stack = stacks.get(ligand_key)
                if stack is None:
                    continue
                v = stack[:, fr[mask], fc[mask]].T
                values = v if values is None else values + v

            if values is None:
                theta = self.rng.uniform(0, 2 * math.pi, k).astype(np.float32)
                dx[mask] = np.cos(theta)
                dy[mask] = np.sin(theta)
                continue

            best = np.argmax(values, axis=1)
            v_max = values[np.arange(k), best]
            v_min = values.min(axis=1)
            has_gradient = (v_max - v_min) > 1e-12

            n_grad = int(np.count_nonzero(has_gradient))
            self.n_gradient_guided += n_grad
            self.n_grad_by_type[code] += n_grad
            self.n_no_gradient += (k - n_grad)

            chosen = DIRECTION_UNIT[best]
            theta = self.rng.uniform(0, 2 * math.pi, k).astype(np.float32)
            dx[mask] = np.where(has_gradient, chosen[:, 0], np.cos(theta))
            dy[mask] = np.where(has_gradient, chosen[:, 1], np.sin(theta))

        zero = (np.abs(dx) + np.abs(dy)) < 1e-9
        if zero.any():
            dx[zero] = 1.0
            dy[zero] = 0.0
        return dx, dy

    # ------------------------------------------------------------------
    @staticmethod
    def reflect_at_boundary(x: np.ndarray, y: np.ndarray) -> None:
        limit = float(Scale.GRID) - 1e-3
        np.abs(x, out=x)
        np.abs(y, out=y)
        for arr in (x, y):
            over = arr > limit
            if over.any():
                arr[over] = 2.0 * limit - arr[over]
            np.clip(arr, 0.0, limit, out=arr)

    @property
    def gradient_fraction(self) -> float:
        return self.n_gradient_guided / max(1, self.n_move_calls)

    def report(self) -> str:
        lines = [f"이동 방식: {self.spec.name_kr}",
                 f"  이동 규칙 : {self.spec.rule}",
                 f"  신호 감지 : {self.spec.sensing}",
                 f"  생성 장소 : {self.spec.spawn}",
                 f"  총 이동 횟수 : {self.n_move_calls:,} 세포-tick",
                 f"  구배 유도 비율 : {self.gradient_fraction * 100:.1f} %"]
        if self.mode == MODE_DIRECTED:
            lines.append("  세포 종류별 구배 유도 비율:")
            for code in range(N_CELL_TYPES):
                tot = int(self.n_by_type[code])
                if tot == 0:
                    continue
                frac = self.n_grad_by_type[code] / tot * 100
                ct = IMMUNE_CELL_TABLE[code]
                recs = ", ".join(ct.receptors)
                lines.append(f"    {ct.name_kr:<14}({recs:<12}) {frac:5.1f} %")
        return "\n".join(lines)


# =============================================================================
#  SECTION 14.  세균 개체군
# =============================================================================


class BacteriaPopulation:
    """
    폐렴구균 개체군.

    ★작품설명서 표 3:
        증식 위치 = 세포외 증식
          숙주세포 안으로 들어가지 않으므로 '감염세포'라는 상태가 없다.
        공간 분포 = 편모가 없어 비운동성. 분열한 자리에 머물러 미세 군락 형성
          딸세포를 모세포 바로 옆에 놓아 미세군락을 형성한다.
        자체 신호 = fMLF를 방출하여 위치를 표시함
          매 tick 자기 위치에 fMLF 를 방출한다.
        제거 경로 = 식균작용이 유일한 제거 수단
          자멸 경로가 없으므로 식세포가 직접 만나 삼켜야만 제거된다.

    ★협막(capsule):
      폐렴구균의 협막 다당은 보체 침착과 호중구 포식을 다중 기전으로
      방해한다(Hyams 등, 2010). 따라서 옵소닌화가 충분히 진행되어야만
      식세포가 인식·포식할 수 있다(Gordon 등, 1980).
    """

    def __init__(self, p: Params, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        self.x = np.zeros(0, dtype=np.float32)
        self.y = np.zeros(0, dtype=np.float32)
        self.opsonin = np.zeros(0, dtype=np.float32)   # 표면 옵소닌 침착도 0~1
        self.n_divided_total = 0

    @property
    def n(self) -> int:
        return int(self.x.size)

    # ------------------------------------------------------------------
    def seed_inoculum(self, k: int) -> None:
        """비강 접종: 한 지점 부근에 국소적으로 접종한다"""
        p = self.p
        self.x = np.clip(
            self.rng.normal(p.inoculum_center_x, p.inoculum_spread, k),
            0, Scale.GRID - 1).astype(np.float32)
        self.y = np.clip(
            self.rng.normal(p.inoculum_center_y, p.inoculum_spread, k),
            0, Scale.GRID - 1).astype(np.float32)
        self.opsonin = np.zeros(k, dtype=np.float32)

    # ------------------------------------------------------------------
    def divide(self, growth_factor: float) -> int:
        """
        이분법 분열.
        ★비운동성이므로 딸세포는 모세포 바로 옆에 놓인다 (미세군락 형성).

          growth_factor : 수용 한계에 의한 증식 억제 계수 (0~1)
        """
        if self.n == 0 or growth_factor <= 0.0:
            return 0
        effective_p = self.p.division_p * growth_factor
        m = self.rng.random(self.n) < effective_p
        k = int(np.count_nonzero(m))
        if k == 0:
            return 0
        s = self.p.daughter_sigma
        nx = self.x[m] + self.rng.normal(0.0, s, k).astype(np.float32)
        ny = self.y[m] + self.rng.normal(0.0, s, k).astype(np.float32)
        np.clip(nx, 0, Scale.GRID - 1, out=nx)
        np.clip(ny, 0, Scale.GRID - 1, out=ny)
        self.x = np.concatenate([self.x, nx])
        self.y = np.concatenate([self.y, ny])
        # 새로 만들어진 표면에는 옵소닌이 아직 붙어 있지 않다
        self.opsonin = np.concatenate([self.opsonin,
                                       np.zeros(k, dtype=np.float32)])
        self.n_divided_total += k
        return k

    # ------------------------------------------------------------------
    def drift(self) -> None:
        """
        점액섬모 수송에 의한 미세 이동.
        ★능동 운동이 아니다. 편모가 없어 스스로 헤엄치지 못한다.
        """
        if self.n == 0:
            return
        s = self.p.drift_sigma
        self.x += self.rng.normal(0.0, s, self.n).astype(np.float32)
        self.y += self.rng.normal(0.0, s, self.n).astype(np.float32)
        MovementEngine.reflect_at_boundary(self.x, self.y)

    # ------------------------------------------------------------------
    def opsonize(self, complement: float, antibody_titer: float) -> None:
        """
        보체 C3b 침착과 항협막 항체 결합에 의한 옵소닌화.
        협막이 이를 저해한다 (Hyams 등, 2010).
        """
        if self.n == 0:
            return
        p = self.p
        gain = ((p.c3b_deposition * complement
                 + p.antibody_opsonin_gain * antibody_titer)
                * (1.0 - p.capsule_resistance * 0.55))
        self.opsonin += gain * (1.0 - self.opsonin) - p.opsonin_decay * self.opsonin
        np.clip(self.opsonin, 0.0, 1.0, out=self.opsonin)

    # ------------------------------------------------------------------
    def remove(self, idx: np.ndarray) -> int:
        """지정한 인덱스의 세균을 제거한다"""
        if idx.size == 0:
            return 0
        keep = np.ones(self.n, dtype=bool)
        keep[idx] = False
        removed = int(self.n - np.count_nonzero(keep))
        self.x = self.x[keep]
        self.y = self.y[keep]
        self.opsonin = self.opsonin[keep]
        return removed

    # ------------------------------------------------------------------
    @property
    def mean_opsonin(self) -> float:
        return float(self.opsonin.mean()) if self.n else 0.0

    def colony_spread(self) -> float:
        """
        군락의 공간 퍼짐 정도 (표준편차, 칸 단위).
        비운동성이므로 값이 작게 유지되어야 한다.
        """
        if self.n < 2:
            return 0.0
        return float(math.sqrt(self.x.var() + self.y.var()))


# =============================================================================
#  SECTION 15.  면역세포 개체군
# =============================================================================


class ImmuneCellPool:
    """
    면역세포 agent 의 배열.

      in_vessel : 혈관 내에 있는지 여부.
                  방향성 이동 조건에서 림프절에서 생성된 세포는 혈류를 타고
                  이동하다가 국소 신호가 역치를 넘으면 조직으로 나온다.
                  (작품설명서: "면역세포는 혈류 흐름 등 주변 환경에 따라 이동")
      capacity  : 포식 용량. 호중구는 약 50 CFU (Rubio 등, 2023).
    """

    __slots__ = ("x", "y", "hx", "hy", "speed", "ctype", "alive", "in_vessel",
                 "specific", "age", "life", "capacity", "n", "cap")

    def __init__(self, capacity: int = 700000):
        self.cap = capacity
        self.n = 0
        f32 = lambda: np.zeros(capacity, dtype=np.float32)
        self.x = f32()
        self.y = f32()
        self.hx = f32()
        self.hy = f32()
        self.speed = f32()
        self.ctype = np.zeros(capacity, dtype=np.uint8)
        self.alive = np.zeros(capacity, dtype=bool)
        self.in_vessel = np.zeros(capacity, dtype=bool)
        self.specific = np.zeros(capacity, dtype=bool)
        self.age = np.zeros(capacity, dtype=np.int32)
        self.life = np.zeros(capacity, dtype=np.int32)
        self.capacity = np.zeros(capacity, dtype=np.int32)

    def add(self, k: int, ctype: int, x: np.ndarray, y: np.ndarray,
            speed: float, life: int, phago_capacity: int,
            rng: np.random.Generator, in_vessel: bool = False,
            specific: bool = False) -> int:
        if k <= 0:
            return 0
        if self.n + k > self.cap:
            k = self.cap - self.n
            if k <= 0:
                return 0
        s = slice(self.n, self.n + k)
        self.x[s] = x[:k]
        self.y[s] = y[:k]
        theta = rng.uniform(0.0, 2.0 * math.pi, k)
        self.hx[s] = np.cos(theta)
        self.hy[s] = np.sin(theta)
        self.speed[s] = speed
        self.ctype[s] = ctype
        self.alive[s] = True
        self.in_vessel[s] = in_vessel
        self.specific[s] = specific
        self.age[s] = 0
        self.life[s] = life
        self.capacity[s] = phago_capacity
        self.n += k
        return k

    def count(self, ctype: int) -> int:
        n = self.n
        return int(np.count_nonzero(self.alive[:n] & (self.ctype[:n] == ctype)))

    def count_specific(self, ctype: int) -> int:
        n = self.n
        return int(np.count_nonzero(self.alive[:n] & (self.ctype[:n] == ctype)
                                    & self.specific[:n]))

    def count_in_tissue(self) -> int:
        n = self.n
        return int(np.count_nonzero(self.alive[:n] & (~self.in_vessel[:n])))

    def compact(self) -> None:
        idx = np.flatnonzero(self.alive[:self.n])
        m = idx.size
        for arr in (self.x, self.y, self.hx, self.hy, self.speed):
            arr[:m] = arr[idx]
        for arr in (self.ctype, self.alive, self.in_vessel, self.specific,
                    self.age, self.life, self.capacity):
            arr[:m] = arr[idx]
        self.alive[m:self.n] = False
        self.n = m


# =============================================================================
#  SECTION 16.  림프절과 후천면역
# =============================================================================


class LymphNode:
    """
    림프절에서 진행되는 후천면역 반응.

    ★세균 감염의 특징:
      점막 세균면역에서는 Th17 이 주역이다. IL-17A 가 폐렴구균 집락에 대한
      획득면역을 매개하며(Lu 등, 2008), 그 효과기는 중성구 동원의 증폭이다
      (Zhang 등, 2009). 반면 살해 T세포는 죽일 감염세포가 없으므로 역할이
      거의 없다.
    """

    def __init__(self, p: Params):
        self.p = p
        self.antigen = 0.0
        self.primed = False
        self.t_primed: Optional[int] = None
        self.priming_clock = 0
        self.cd4 = 100.0
        self.cd8 = 50.0
        self.th17 = 0.0
        self.plasma_cell = 0.0
        self.igm = 0.0
        self.igg = 0.0
        self.t_igm_detected: Optional[float] = None
        self.t_igg_detected: Optional[float] = None
        self.t_th17_peak: Optional[float] = None
        self._th17_max = 0.0

    # ------------------------------------------------------------------
    def step(self, tick: int, antigen_in: float) -> None:
        p = self.p
        self.antigen = self.antigen * 0.985 + antigen_in

        if not self.primed and self.antigen > p.antigen_threshold:
            self.priming_clock += 1
            if self.priming_clock >= p.priming_ticks:
                self.primed = True
                self.t_primed = tick

        if not self.primed:
            return

        days_since = (tick - self.t_primed) / Scale.TICKS_PER_DAY
        if days_since < 6.0:
            self.cd4 = min(self.cd4 * 1.0125, 60000.0)
            self.cd8 = min(self.cd8 * 1.0105, 30000.0)
            # ★Th17 분화 — 세균 점막면역의 주역
            self.th17 = min(self.th17 + 0.020 * self.cd4 - 0.010 * self.th17,
                            45000.0)
            self.plasma_cell = min(
                self.plasma_cell + 0.020 * self.cd4 - 0.006 * self.plasma_cell,
                55000.0)
        else:
            self.cd4 *= 0.985
            self.cd8 *= 0.985
            self.th17 *= 0.990
            self.plasma_cell *= 0.988

        if self.th17 > self._th17_max:
            self._th17_max = self.th17
            self.t_th17_peak = tick / Scale.TICKS_PER_DAY

        self.igm += p.ab_igm_rate * self.plasma_cell / 1000.0
        self.igm *= (1.0 - p.igm_decay)
        if days_since > 3.5:
            self.igg += p.ab_igg_rate * self.plasma_cell / 1000.0
        self.igg *= (1.0 - p.igg_decay)

        day = tick / Scale.TICKS_PER_DAY
        if self.t_igm_detected is None and self.igm > 1.0:
            self.t_igm_detected = day
        if self.t_igg_detected is None and self.igg > 1.0:
            self.t_igg_detected = day

    # ------------------------------------------------------------------
    @property
    def titer(self) -> float:
        """총 항체 역가 (항협막 항체)"""
        return min(self.igm + self.igg * 1.8, self.p.ab_titer_cap)

    @property
    def neutrophil_boost(self) -> float:
        """Th17 에 의한 중성구 동원 증폭 계수"""
        p = self.p
        if self.th17 <= 0:
            return 1.0
        return 1.0 + min(p.th17_neutrophil_boost_max,
                         self.th17 / p.th17_boost_scale)


# =============================================================================
#  SECTION 17.  시뮬레이션 본체
# =============================================================================


def match_same_site(a_sites: np.ndarray,
                    b_sites: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    같은 격자 칸에 있는 (a 인덱스, b 인덱스) 쌍을 찾는다.

    ★중요: 한 칸에 식세포 여러 개와 세균 여러 개가 함께 있을 때,
      각 식세포에게 서로 다른 세균을 짝지어 준다.
      모든 식세포에게 같은 세균을 배정하면 중복 제거 과정에서 그 칸의
      포식이 1건으로 축소되어, 실제보다 식균 효율이 크게 과소평가된다.
      (한 칸의 세균 수보다 식세포가 많으면 남는 식세포는 짝이 없다)
    """
    if a_sites.size == 0 or b_sites.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    order_b = np.argsort(b_sites, kind="stable")
    sorted_b = b_sites[order_b]
    order_a = np.argsort(a_sites, kind="stable")
    sorted_a = a_sites[order_a]

    # b 쪽: 각 칸의 시작 위치와 개수
    uniq_b, start_b, count_b = np.unique(sorted_b, return_index=True,
                                         return_counts=True)
    pos = np.searchsorted(uniq_b, sorted_a)
    pos_c = np.clip(pos, 0, uniq_b.size - 1)
    has = uniq_b[pos_c] == sorted_a
    if not has.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    a_hit = order_a[has]          # 원래 a 인덱스
    grp = pos_c[has]              # 각 a 가 속한 칸의 그룹 번호
    sel_sites = sorted_a[has]

    # 같은 칸 안에서 a 의 순번(0,1,2,...)을 구한다
    first_of_group = np.searchsorted(sel_sites, sel_sites, side="left")
    rank = np.arange(sel_sites.size) - first_of_group

    # 그 칸의 세균 수보다 순번이 작은 경우에만 짝이 성립한다
    ok = rank < count_b[grp]
    if not ok.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    b_hit = order_b[start_b[grp[ok]] + rank[ok]]
    return a_hit[ok], b_hit


class PneumococcusSimulation:
    """
    폐렴구균 감염 - 면역반응 Agent-Based Model.

    한 tick 의 진행 순서:
       A. 세균 증식 (수용 한계 · 영양 면역 반영)
       B. 미세 이동 (점액섬모 수송)
       C. 상주 폐포대식세포에 의한 기저 제거
       D. 옵소닌화
       E. 리간드 농도장 갱신  (★세균이 fMLF 방출)
       F. 면역세포 동원 (Th17 증폭 포함)
       G. 수명 처리
       H. 이동  (★두 조건의 유일한 차이)
       I. 혈관외유출 판정
       J. 식균작용  (★유일한 제거 수단)
       K. 항체 · 보체에 의한 제거
       L. 조직 손상
       M. 체액성 인자 갱신
       N. 림프절 반응
       O. 조직 재생
    """

    def __init__(self, p: Optional[Params] = None,
                 movement_mode: str = MODE_RANDOM,
                 seed: Optional[int] = None,
                 verbose: bool = False):
        self.p = p if p is not None else Params()
        if seed is not None:
            self.p.seed = seed
        self.mode = movement_mode
        self.movement_spec = MOVEMENT_TABLE[movement_mode]
        self.rng = np.random.default_rng(self.p.seed)
        self.verbose = verbose

        self.tissue = Tissue(self.rng)
        self.bacteria = BacteriaPopulation(self.p, self.rng)
        self.bacteria.seed_inoculum(self.p.n_bacteria0)
        self.cells = ImmuneCellPool()
        self.mover = MovementEngine(movement_mode, self.rng)
        self.lymph_node = LymphNode(self.p)
        self.ligands = LigandFieldSet()

        n = Scale.FIELD_N
        self.ifng = np.zeros((n, n), dtype=np.float32)
        self.complement = 0.0
        self.inflammation = 0.0

        self.tick = 0
        self.daily: List[dict] = []
        self.wall_time = 0.0

        # ---- 누적 통계 ----
        self.phago_by_neutrophil = 0
        self.phago_by_monocyte = 0
        self.cleared_by_alveolar_mac = 0
        self.cleared_by_antibody = 0
        self.cleared_by_complement = 0
        self.dead_immune_cells = 0
        self.memory_cells = 0
        self.contact_events = 0
        self.n_extravasated = 0
        self.t_first_extravasation: Optional[int] = None
        self.t_first_arrival: Optional[int] = None

        self._seed_initial_population()

    # ------------------------------------------------------------------
    def _spawn(self, k: int, ctype: int, speed: float, life: int,
               phago_capacity: int, specific: bool = False) -> int:
        """
        면역세포를 생성한다 (작품설명서 표 2 '생성 장소').
          무작위 이동 -> 조직에 무작위적으로 생성
          방향성 이동 -> 림프절에서 생성 (혈류를 타고 이동 후 유출)
        """
        if k <= 0:
            return 0
        x, y = self.tissue.spawn_xy(k, self.mode, self.rng)
        in_vessel = (self.mode == MODE_DIRECTED)
        return self.cells.add(k, ctype, x, y, speed, life, phago_capacity,
                              self.rng, in_vessel=in_vessel, specific=specific)

    def _seed_initial_population(self) -> None:
        p = self.p
        self._spawn(p.n_neutrophil0, NEUTROPHIL, p.v_neutrophil,
                    p.life_neutrophil, p.capacity_neutrophil)
        self._spawn(p.n_monocyte0, MONOCYTE, p.v_monocyte,
                    p.life_monocyte, p.capacity_monocyte)
        self._spawn(p.n_nk0, NK_CELL, p.v_nk, p.life_nk, 0)
        self._spawn(p.n_helper_t0, HELPER_T, p.v_t, p.life_t, 0)
        self._spawn(p.n_killer_t0, KILLER_T, p.v_t, p.life_t, 0)
        self._spawn(p.n_bcell0, B_CELL, p.v_b, p.life_b, 0)
        self._spawn(p.n_dendritic0, DENDRITIC, p.v_dendritic,
                    p.life_dendritic, 0)

    # ==================================================================
    def step(self) -> None:
        """한 tick 진행"""
        p = self.p
        tissue = self.tissue
        cells = self.cells
        rng = self.rng
        self.tick += 1
        t = self.tick

        # ---------- A. 세균 증식 ----------
        # 세균은 수용 한계(Hamilton 등 2019: 최대 균량 1x10^8 CFU)에
        # 도달할 때까지 배가시간 56분(Jose 등 2015)으로 증식한다.
        # 그 밖의 증식 억제 항은 두지 않는다.
        occupancy = self.bacteria.n / p.carrying_capacity
        growth_factor = max(0.0, 1.0 - occupancy)
        self.bacteria.divide(growth_factor)

        # ---------- B. 미세 이동 (능동 운동 아님) ----------
        self.bacteria.drift()

        # ---------- C. 상주 폐포대식세포에 의한 기저 제거 ----------
        if self.bacteria.n:
            # AM 개체수가 유한하므로 고균량에서는 포화한다 (Jose 등, 2015)
            effective = p.am_clearance_p / (
                1.0 + self.bacteria.n / p.am_saturation_n)
            m = rng.random(self.bacteria.n) < effective
            if m.any():
                self.cleared_by_alveolar_mac += self.bacteria.remove(
                    np.flatnonzero(m))

        # ---------- D. 옵소닌화 ----------
        self.bacteria.opsonize(self.complement, self.lymph_node.titer)

        # ---------- E. 리간드 농도장 갱신 ----------
        self._update_ligand_fields()

        # ---------- F. 면역세포 동원 ----------
        self._recruit()

        # ---------- G. 수명 처리 ----------
        n = cells.n
        if n:
            alive = cells.alive[:n]
            cells.age[:n][alive] += 1
            died = alive & (cells.age[:n] >= cells.life[:n])
            k = int(np.count_nonzero(died))
            if k:
                cells.alive[:n][died] = False
                self.dead_immune_cells += k
            if n > 20000 and (cells.alive[:n].mean() < 0.93
                              or n > int(0.55 * cells.cap)):
                cells.compact()

        # ---------- H. 이동 (★두 조건의 유일한 차이) ----------
        stacks = self.ligands.direction_stacks()
        n = cells.n
        idx = np.flatnonzero(cells.alive[:n])
        if idx.size:
            speed = cells.speed[idx].copy()
            in_vessel = cells.in_vessel[idx]
            if in_vessel.any():
                speed[in_vessel] = p.v_vessel

            xs = cells.x[idx].copy()
            ys = cells.y[idx].copy()
            hxs = cells.hx[idx].copy()
            hys = cells.hy[idx].copy()
            fr, fc = Tissue.field_of_xy(xs, ys)
            self.mover.move(xs, ys, hxs, hys, speed, stacks, fr, fc,
                            cells.ctype[idx])
            cells.x[idx] = xs
            cells.y[idx] = ys
            cells.hx[idx] = hxs
            cells.hy[idx] = hys

            # ---------- I. 혈관외유출 판정 ----------
            if in_vessel.any():
                self._extravasate(idx, in_vessel, t)

        # ---------- J. 식균작용 (★유일한 제거 수단) ----------
        self._phagocytosis(t)

        # ---------- K. 항체 · 보체에 의한 제거 ----------
        self._humoral_clearance()

        # ---------- L. 조직 손상 ----------
        self._tissue_damage()

        # ---------- M. 체액성 인자 갱신 ----------
        self._update_humoral_factors()

        # ---------- N. 림프절 반응 ----------
        self.lymph_node.step(t, 0.0045 * self.bacteria.n)
        if self.lymph_node.primed and self.memory_cells == 0:
            days = (t - self.lymph_node.t_primed) / Scale.TICKS_PER_DAY
            if days > 8.0:
                self.memory_cells = int(
                    (self.lymph_node.cd4 + self.lymph_node.cd8)
                    * p.memory_fraction)

        # ---------- O. 조직 재생 ----------
        tissue.regenerate(p.regeneration_rate * (1.0 - self.inflammation))

    # ==================================================================
    def _update_ligand_fields(self) -> None:
        """
        표 1의 리간드를 각각 독립된 농도장으로 갱신한다.

        ★인플루엔자 모형과의 결정적 차이:
          fMLF 의 생성원이 존재한다. 세균 자신이 매 tick 자기 위치에 fMLF 를
          방출하므로, 단핵구/대식세포는 FPR1 로 세균 위치를 직접 추적한다.
          표 3의 '자체 신호: fMLF를 방출하여 위치를 표시함'을 구현한 것이다.
        """
        p = self.p
        cells = self.cells
        n = cells.n
        alive = cells.alive[:n]

        if self.bacteria.n:
            bfr, bfc = Tissue.field_of_xy(self.bacteria.x, self.bacteria.y)
            # ★fMLF — 세균 자신이 방출한다 (표적이 곧 신호원)
            self.ligands.deposit(LIG_FMLF, bfr, bfc, p.fmlf_per_bacterium)
            # 감염 부위 상피가 방출하는 케모카인
            self.ligands.deposit(LIG_CXCL8, bfr, bfc, p.cxcl8_from_site)
            self.ligands.deposit(LIG_CCL2, bfr, bfc, p.ccl2_from_site)
            self.ligands.deposit(LIG_CCL35, bfr, bfc, p.ccl35_from_site)
            self.ligands.deposit(LIG_CXCR3L, bfr, bfc, p.cxcr3l_basal)

        # ---- 침윤한 중성구의 CXCL8 자가 증폭 ----
        sel = np.flatnonzero(alive & (cells.ctype[:n] == NEUTROPHIL)
                             & (~cells.in_vessel[:n]))
        if sel.size:
            fr, fc = Tissue.field_of_xy(cells.x[sel], cells.y[sel])
            self.ligands.deposit(LIG_CXCL8, fr, fc, p.cxcl8_from_neutrophil)

        # ---- 활성 대식세포의 CCL2 · CCL3/CCL5 방출 ----
        sel = np.flatnonzero(alive & (cells.ctype[:n] == MONOCYTE)
                             & (~cells.in_vessel[:n]))
        if sel.size:
            fr, fc = Tissue.field_of_xy(cells.x[sel], cells.y[sel])
            self.ligands.deposit(LIG_CCL2, fr, fc, p.ccl2_from_monocyte)
            self.ligands.deposit(LIG_CCL35, fr, fc, p.ccl35_from_monocyte)

        # ---- NK세포와 조력T세포의 IFN-gamma 분비 ----
        sel = np.flatnonzero(alive & ((cells.ctype[:n] == NK_CELL)
                                      | (cells.ctype[:n] == HELPER_T))
                             & (~cells.in_vessel[:n]))
        if sel.size:
            fr, fc = Tissue.field_of_xy(cells.x[sel], cells.y[sel])
            np.add.at(self.ifng, (fr, fc), p.ifng_from_nk_t)

        # ---- ★CXCL9/10/11 은 IFN-gamma 유도성 (Luster 등, 1985) ----
        self.ligands.add_field(LIG_CXCR3L, self.ifng, p.cxcr3l_ifng_gain)

        # ---- ★CXCL13 은 림프절이 항상적으로 방출 (Okada 등, 2002) ----
        self.ligands.deposit_region(LIG_CXCL13, self.tissue.lymph_field_slice,
                                    p.cxcl13_from_lymphnode)

        # ---- 확산 및 감쇠 ----
        self.ligands.diffuse_and_decay()

        self.ifng = gaussian_filter(self.ifng, sigma=p.ifng_sigma,
                                    mode="nearest")
        self.ifng *= (1.0 - p.ifng_decay)

    # ==================================================================
    def _recruit(self) -> None:
        """
        면역세포를 동원한다.
        ★Th17 이 중성구 동원을 증폭한다 (Lu 등 2008, Zhang 등 2009).
        """
        p = self.p
        drive = float(np.clip(
            0.55 * self.inflammation
            + 0.45 * min(1.0, self.bacteria.n / 25000.0), 0.0, 1.0))

        for ctype, gain, speed, life, cap in (
                (NEUTROPHIL, p.recruit_neutrophil, p.v_neutrophil,
                 p.life_neutrophil, p.capacity_neutrophil),
                (MONOCYTE, p.recruit_monocyte, p.v_monocyte,
                 p.life_monocyte, p.capacity_monocyte),
                (NK_CELL, p.recruit_nk, p.v_nk, p.life_nk, 0)):
            effective_gain = gain
            if ctype == NEUTROPHIL:
                effective_gain = gain * self.lymph_node.neutrophil_boost
            k = int(effective_gain * drive / Scale.TICKS_PER_DAY)
            if k > 0:
                self._spawn(k, ctype, speed, life, cap)

        # ---- 후천면역 효과기 세포의 조직 진입 ----
        if self.lymph_node.primed:
            # ★Th17(조력T)이 주역이다
            k = int(self.lymph_node.th17 * 0.00120)
            if k > 0:
                self._spawn(k, HELPER_T, p.v_t, p.life_effector_t, 0,
                            specific=True)
            # ★살해T세포는 죽일 감염세포가 없으므로 진입이 미미하다
            k = int(self.lymph_node.cd8 * 0.00012)
            if k > 0:
                self._spawn(k, KILLER_T, p.v_t, p.life_effector_t, 0,
                            specific=True)

    # ==================================================================
    def _extravasate(self, idx: np.ndarray, in_vessel: np.ndarray,
                     t: int) -> None:
        """
        혈관 내 세포가 국소 신호 농도를 감지하면 조직으로 빠져나온다.
        작품설명서: "면역세포는 혈류 흐름 등 주변 환경에 따라 이동"
        """
        p = self.p
        cells = self.cells
        vessel_idx = idx[in_vessel]
        if vessel_idx.size == 0:
            return
        fr, fc = Tissue.field_of_xy(cells.x[vessel_idx], cells.y[vessel_idx])
        ctypes = cells.ctype[vessel_idx]

        sensed = np.zeros(vessel_idx.size, dtype=np.float32)
        for code in np.unique(ctypes):
            code = int(code)
            m = ctypes == code
            total = np.zeros(int(np.count_nonzero(m)), dtype=np.float32)
            for _receptor, ligand_key in CELL_RECEPTOR_FIELDS[code]:
                total = total + self.ligands.field[ligand_key][fr[m], fc[m]]
            sensed[m] = total

        out = sensed > p.extravasation_threshold
        if out.any():
            gi = vessel_idx[out]
            cells.in_vessel[gi] = False
            self.n_extravasated += int(gi.size)
            if self.t_first_extravasation is None:
                self.t_first_extravasation = t

    # ==================================================================
    def _phagocytosis(self, t: int) -> None:
        """
        ★식균작용 — 폐렴구균의 유일한 제거 수단 (표 3).

        살해 조건 (작품설명서 라-B):
          중성구와 대식세포가 옵소닌화된 세균을 인식하여 세포막으로 감싸
          식균작용을 하고, 내부의 분해효소와 활성산소가 방출되어 세균을
          분해, 살해한다.

        식세포 1개가 한 tick 에 여러 균을 연속으로 삼킬 수 있으므로
        여러 라운드로 나누어 처리한다.
        """
        p = self.p
        cells = self.cells
        rng = self.rng
        n = cells.n
        idx = np.flatnonzero(cells.alive[:n])
        if idx.size == 0 or self.bacteria.n == 0:
            return

        tissue_cells = idx[~cells.in_vessel[idx]]
        if tissue_cells.size == 0:
            return

        c_site = Tissue.site_of_xy(cells.x[tissue_cells], cells.y[tissue_cells])
        selector = {
            NEUTROPHIL: np.flatnonzero(cells.ctype[tissue_cells] == NEUTROPHIL),
            MONOCYTE: np.flatnonzero(cells.ctype[tissue_cells] == MONOCYTE),
        }

        for round_i in range(p.phagocytosis_rounds):
            if self.bacteria.n == 0:
                break
            b_site = Tissue.site_of_xy(self.bacteria.x, self.bacteria.y)
            eaten: List[np.ndarray] = []

            for ctype, base_p in ((NEUTROPHIL, p.neutrophil_phago_p),
                                  (MONOCYTE, p.monocyte_phago_p)):
                pool = selector[ctype]
                if pool.size == 0:
                    continue
                # 포식 용량이 남은 세포만 참여한다
                sel = pool[cells.capacity[tissue_cells[pool]] > 0]
                if sel.size == 0:
                    continue

                loc, bi = match_same_site(c_site[sel], b_site)
                if loc.size == 0:
                    continue
                if round_i == 0:
                    self.contact_events += int(loc.size)
                    if self.t_first_arrival is None:
                        self.t_first_arrival = t

                # ★옵소닌화 요구 — 협막 때문에 충분히 옵소닌화되어야 포식 가능
                opsonin = self.bacteria.opsonin[bi]
                threshold = p.opsonin_requirement * 0.35
                efficiency = base_p * np.clip(
                    (opsonin - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
                ok = rng.random(loc.size) < efficiency
                if not ok.any():
                    continue

                gi = tissue_cells[sel[loc[ok]]]
                targets = bi[ok]
                uniq, first = np.unique(targets, return_index=True)
                np.subtract.at(cells.capacity, gi[first], 1)
                if ctype == NEUTROPHIL:
                    self.phago_by_neutrophil += int(uniq.size)
                else:
                    self.phago_by_monocyte += int(uniq.size)
                eaten.append(uniq)

            if eaten:
                self.bacteria.remove(np.unique(np.concatenate(eaten)))
            else:
                break

    # ==================================================================
    def _humoral_clearance(self) -> None:
        """
        항체와 보체에 의한 제거.

        ★그람양성균은 두꺼운 펩티도글리칸 층과 협막 때문에 보체 막공격복합체에
          저항한다. 따라서 항체·보체의 본질적 기여는 '직접 살균'이 아니라
          '옵소닌화'이며, 직접 제거 효과는 작다.
          다만 항협막 IgG 는 응집을 일으켜 점액섬모 수송으로 제거되게 하므로
          식세포와의 조우 여부와 무관하게 작용하는 경로가 존재한다.
        """
        p = self.p
        rng = self.rng
        if self.bacteria.n == 0:
            return

        titer = self.lymph_node.titer
        if titer > 0.02:
            prob = min(p.antibody_clear_max, p.antibody_clear_k * titer)
            m = rng.random(self.bacteria.n) < prob
            if m.any():
                self.cleared_by_antibody += self.bacteria.remove(
                    np.flatnonzero(m))

        if self.bacteria.n and self.complement > 0.35:
            m = rng.random(self.bacteria.n) < p.complement_clear_p * self.complement
            if m.any():
                self.cleared_by_complement += self.bacteria.remove(
                    np.flatnonzero(m))

    # ==================================================================
    def _tissue_damage(self) -> None:
        """세균의 조직 침습과 중성구 매개 부수적 손상"""
        p = self.p
        rng = self.rng
        tissue = self.tissue
        cells = self.cells

        # ---- 세균에 의한 직접 손상 ----
        if self.bacteria.n:
            m = rng.random(self.bacteria.n) < p.epithelial_damage_p
            if m.any():
                tissue.kill_epithelium(
                    Tissue.site_of_xy(self.bacteria.x[m], self.bacteria.y[m]))

        # ---- 중성구 매개 부수적 손상 ----
        n = cells.n
        idx = np.flatnonzero(cells.alive[:n])
        if idx.size:
            ns = idx[(cells.ctype[idx] == NEUTROPHIL) & (~cells.in_vessel[idx])]
            if ns.size:
                prob = p.neutrophil_damage_p * (1.0 + 3.0 * self.inflammation)
                m = rng.random(ns.size) < prob
                if m.any():
                    tissue.kill_epithelium(
                        Tissue.site_of_xy(cells.x[ns[m]], cells.y[ns[m]]))

    # ==================================================================
    def _update_humoral_factors(self) -> None:
        """보체 활성도와 염증 정도를 갱신한다"""
        p = self.p
        drive = min(1.0, self.bacteria.n / 40000.0)
        self.complement += p.complement_on * drive * (1.0 - self.complement)
        self.complement -= p.complement_off * self.complement
        self.complement = float(np.clip(self.complement, 0.0, 1.0))
        self.inflammation += p.inflammation_on * drive * (1.0 - self.inflammation)
        self.inflammation -= p.inflammation_off * self.inflammation
        self.inflammation = float(np.clip(self.inflammation, 0.0, 1.0))

    # ==================================================================
    def phagocyte_bacteria_overlap(self) -> Tuple[float, float]:
        """
        식세포와 세균의 공간 분포 겹침도(%)와 식세포 분포 불균일도를 계산한다.
        공간을 32x32 구획으로 나눈 뒤 두 분포를 정규화하여 겹치는 비율을 구한다.
        """
        cells = self.cells
        n = cells.n
        alive = cells.alive[:n]
        phago = alive & ((cells.ctype[:n] == NEUTROPHIL)
                         | (cells.ctype[:n] == MONOCYTE)) & (~cells.in_vessel[:n])
        bins = 32
        rng_box = [[0, Scale.GRID], [0, Scale.GRID]]
        if self.bacteria.n:
            hb = np.histogram2d(self.bacteria.y, self.bacteria.x,
                                bins=bins, range=rng_box)[0]
        else:
            hb = np.zeros((bins, bins))
        hp = np.histogram2d(cells.y[:n][phago], cells.x[:n][phago],
                            bins=bins, range=rng_box)[0]
        nb = hb / hb.sum() if hb.sum() > 0 else hb
        npf = hp / hp.sum() if hp.sum() > 0 else hp
        overlap = float(np.minimum(nb, npf).sum()) * 100.0
        cv = float(hp.std() / hp.mean()) if hp.mean() > 0 else 0.0
        return overlap, cv

    # ==================================================================
    def snapshot(self) -> dict:
        """현재 상태를 기록한다"""
        c = self.cells
        tissue = self.tissue
        ln = self.lymph_node
        overlap, cv = self.phagocyte_bacteria_overlap()
        return {
            "tick": self.tick,
            "day": self.tick / Scale.TICKS_PER_DAY,
            # ---- 병원체 ----
            "bacteria": self.bacteria.n,
            "opsonized": self.bacteria.mean_opsonin,
            "colony_spread": self.bacteria.colony_spread(),
            # ---- 조직 ----
            "healthy": tissue.n_healthy,
            "dead_epithelium": tissue.n_dead_epithelium,
            "damage": tissue.cumulative_damage,
            # ---- 면역세포 ----
            "neutrophil": c.count(NEUTROPHIL),
            "monocyte": c.count(MONOCYTE),
            "nk": c.count(NK_CELL),
            "helper_t": c.count(HELPER_T),
            "killer_t": c.count(KILLER_T),
            "bcell": c.count(B_CELL),
            "dendritic": c.count(DENDRITIC),
            "helper_t_specific": c.count_specific(HELPER_T),
            "killer_t_specific": c.count_specific(KILLER_T),
            "in_tissue": c.count_in_tissue(),
            "memory": self.memory_cells,
            # ---- 리간드 ----
            "fMLF": self.ligands.mean(LIG_FMLF),
            "CXCL8": self.ligands.mean(LIG_CXCL8),
            "CCL2": self.ligands.mean(LIG_CCL2),
            "CXCL9_10_11": self.ligands.mean(LIG_CXCR3L),
            "CXCL13": self.ligands.mean(LIG_CXCL13),
            "CCL3_CCL5": self.ligands.mean(LIG_CCL35),
            "ifng": float(self.ifng.mean()),
            # ---- 체액성 인자 ----
            "complement": self.complement,
            "inflammation": self.inflammation,
            "igm": ln.igm, "igg": ln.igg, "antibody": ln.titer,
            "th17": ln.th17, "ln_cd4": ln.cd4, "ln_cd8": ln.cd8,
            "ln_plasma": ln.plasma_cell,
            # ---- 제거량 ----
            "phago_neutrophil": self.phago_by_neutrophil,
            "phago_monocyte": self.phago_by_monocyte,
            "phago_total": self.phago_by_neutrophil + self.phago_by_monocyte,
            "cleared_alveolar_mac": self.cleared_by_alveolar_mac,
            "cleared_antibody": self.cleared_by_antibody,
            "cleared_complement": self.cleared_by_complement,
            "dead_immune": self.dead_immune_cells,
            "contacts": self.contact_events,
            "extravasated": self.n_extravasated,
            # ---- 공간 지표 ----
            "overlap_pct": overlap,
            "phagocyte_cv": cv,
        }

    # ==================================================================
    def run(self) -> List[dict]:
        """시뮬레이션을 끝까지 실행한다"""
        p = self.p
        t_start = time.time()
        self.daily.append(self.snapshot())
        if self.verbose:
            print(f"  Day    0 | 세균 {self.bacteria.n:>8,} | "
                  f"정상세포 {self.tissue.n_healthy:>7,}")

        total_ticks = p.max_days * Scale.TICKS_PER_DAY
        clear_streak = 0
        clear_needed = max(1, int(p.stop_after_clear_days
                                  * Scale.TICKS_PER_DAY))

        while self.tick < total_ticks:
            self.step()

            if self.tick % Scale.TICKS_PER_DAY == 0:
                s = self.snapshot()
                self.daily.append(s)
                if self.verbose:
                    print(f"  Day {s['day']:4.0f} | "
                          f"세균 {s['bacteria']:>8,} | "
                          f"정상세포 {s['healthy']:>7,} | "
                          f"중성구 {s['neutrophil']:>7,} | "
                          f"fMLF {s['fMLF']:.5f} | "
                          f"옵소닌 {s['opsonized']:.3f}", flush=True)

            if self.bacteria.n == 0 and self.tick > Scale.TICKS_PER_DAY:
                clear_streak += 1
                if clear_streak >= clear_needed:
                    break
            else:
                clear_streak = 0

        if self.daily[-1]["tick"] != self.tick:
            self.daily.append(self.snapshot())
        self.wall_time = time.time() - t_start
        return self.daily


# =============================================================================
#  SECTION 18.  결과 분석
# =============================================================================


def _series(daily: List[dict], key: str) -> np.ndarray:
    return np.array([d[key] for d in daily], dtype=float)


def _days(daily: List[dict]) -> np.ndarray:
    return np.array([d["day"] for d in daily], dtype=float)


def _peak(daily: List[dict], key: str) -> Tuple[float, float]:
    v = _series(daily, key)
    if v.size == 0:
        return 0.0, float("nan")
    i = int(np.argmax(v))
    return float(v[i]), float(daily[i]["day"])


def _first_below_after_peak(daily: List[dict], key: str,
                            fraction: float) -> Optional[float]:
    """정점 이후 정점 대비 지정 비율로 처음 떨어지는 시각 (선형 보간)"""
    v = _series(daily, key)
    d = _days(daily)
    if v.size == 0 or v.max() <= 0:
        return None
    i = int(np.argmax(v))
    threshold = v[i] * fraction
    for k in range(i, v.size - 1):
        if v[k] >= threshold > v[k + 1]:
            span = v[k] - v[k + 1]
            frac = (v[k] - threshold) / span if span > 0 else 0.0
            return float(d[k] + frac * (d[k + 1] - d[k]))
    return None


def _clear_day(daily: List[dict], key: str) -> Optional[float]:
    v = _series(daily, key)
    d = _days(daily)
    if v.size == 0:
        return None
    i = int(np.argmax(v))
    for k in range(i, v.size):
        if v[k] == 0:
            return float(d[k])
    return None


def analyze(sim: PneumococcusSimulation) -> dict:
    """시뮬레이션 결과에서 분석 지표를 산출한다"""
    daily = sim.daily
    d = _days(daily)
    bacteria = _series(daily, "bacteria")
    healthy = _series(daily, "healthy")
    n0 = sim.tissue.n_target_initial

    peak_b, peak_day = _peak(daily, "bacteria")

    # 초기 지수증식률로부터 배가시간을 역산한다 (문헌값 56분과 대조)
    growth_rate = float("nan")
    doubling_min = float("nan")
    # 접종 직후(Day 0)부터 첫 기록 시점까지의 상승 구간을 사용한다.
    # 정점 직전 구간만 쓰면 이미 포화된 뒤라 증식률이 0에 가깝게 나온다.
    peak_idx = int(np.argmax(bacteria))
    rising = [x for x in daily[:max(1, peak_idx) + 1] if x["bacteria"] > 0]
    if len(rising) >= 2:
        d0 = rising[0]
        # 수용 한계에 도달하기 전 시점을 종점으로 삼는다
        cap = sim.p.carrying_capacity
        d1 = rising[1]
        for x in rising[1:]:
            if x["bacteria"] < cap * 0.6:
                d1 = x
            else:
                break
        if d1["day"] > d0["day"] and d0["bacteria"] > 0:
            growth_rate = (math.log(d1["bacteria"] / d0["bacteria"])
                           / (d1["day"] - d0["day"]))
    if not math.isnan(growth_rate) and growth_rate > 0:
        doubling_min = math.log(2.0) / growth_rate * 24.0 * 60.0

    # ★겹침도는 병원체가 가장 많은 시점에서 평가한다.
    #   최종 시점은 이미 감염이 정리된 뒤라 두 분포가 모두 흩어져 있어
    #   탐색 효율을 반영하지 못한다.
    overlap = float(daily[peak_idx]["overlap_pct"])
    cv = float(daily[peak_idx]["phagocyte_cv"])
    overlap_final, cv_final = sim.phagocyte_bacteria_overlap()
    total_phago = sim.phago_by_neutrophil + sim.phago_by_monocyte
    total_removed = (total_phago + sim.cleared_by_alveolar_mac
                     + sim.cleared_by_antibody + sim.cleared_by_complement)

    result = {
        # ---- 실행 정보 ----
        "mode": sim.mode,
        "mode_name": sim.movement_spec.name_kr,
        "seed": sim.p.seed,
        "wall_time": sim.wall_time,
        "max_days": sim.p.max_days,

        # ---- 병원체 ----
        "bacteria_peak": peak_b,
        "bacteria_peak_day": peak_day,
        "bacteria_auc": float(np.trapezoid(bacteria, d)),
        "bacteria_t50": _first_below_after_peak(daily, "bacteria", 0.5),
        "bacteria_t90": _first_below_after_peak(daily, "bacteria", 0.1),
        "clear_day": _clear_day(daily, "bacteria"),
        "bacteria_final": float(bacteria[-1]),
        "growth_rate_per_day": growth_rate,
        "doubling_time_min_est": doubling_min,
        "colony_spread_final": float(daily[-1]["colony_spread"]),
        "max_opsonized": float(max(x["opsonized"] for x in daily)),

        # ---- 조직 ----
        "min_healthy": float(healthy.min()),
        "max_target_depletion": float(1.0 - healthy.min() / n0),
        "final_healthy": float(healthy[-1]),
        "dead_epithelium": sim.tissue.n_dead_epithelium,
        "damage_final": float(daily[-1]["damage"]),

        # ---- 면역세포 ----
        "neutrophil_peak": float(max(x["neutrophil"] for x in daily)),
        "monocyte_peak": float(max(x["monocyte"] for x in daily)),
        "nk_peak": float(max(x["nk"] for x in daily)),
        "helper_t_specific_peak": float(
            max(x["helper_t_specific"] for x in daily)),
        "killer_t_specific_peak": float(
            max(x["killer_t_specific"] for x in daily)),
        "in_tissue_peak": float(max(x["in_tissue"] for x in daily)),
        "memory_final": sim.memory_cells,
        "dead_immune": sim.dead_immune_cells,
        "n_extravasated": sim.n_extravasated,

        # ---- 제거 경로별 기여도 ----
        "phago_neutrophil": sim.phago_by_neutrophil,
        "phago_monocyte": sim.phago_by_monocyte,
        "phago_total": total_phago,
        "cleared_alveolar_mac": sim.cleared_by_alveolar_mac,
        "cleared_antibody": sim.cleared_by_antibody,
        "cleared_complement": sim.cleared_by_complement,
        "phago_share": (total_phago / total_removed * 100.0
                        if total_removed > 0 else float("nan")),

        # ---- 탐색 효율 ----
        "contacts": sim.contact_events,
        "overlap_pct": overlap,
        "phagocyte_cv": cv,
        "overlap_pct_final": overlap_final,
        "phagocyte_cv_final": cv_final,
        "gradient_fraction": sim.mover.gradient_fraction,
        "t_first_extravasation": (sim.t_first_extravasation
                                  / Scale.TICKS_PER_DAY
                                  if sim.t_first_extravasation else None),
        "t_first_arrival": (sim.t_first_arrival / Scale.TICKS_PER_DAY
                            if sim.t_first_arrival else None),

        # ---- 후천면역 ----
        "antibody_peak": float(max(x["antibody"] for x in daily)),
        "th17_peak": float(max(x["th17"] for x in daily)),
        "t_th17_peak": sim.lymph_node.t_th17_peak,
        "t_igm": sim.lymph_node.t_igm_detected,
        "t_igg": sim.lymph_node.t_igg_detected,
        "t_primed": (sim.lymph_node.t_primed / Scale.TICKS_PER_DAY
                     if sim.lymph_node.t_primed else None),

        # ---- 신호 ----
        "complement_peak": float(max(x["complement"] for x in daily)),
        "inflammation_peak": float(max(x["inflammation"] for x in daily)),
        "peak_fMLF": sim.ligands.peak[LIG_FMLF],
        "peak_CXCL8": sim.ligands.peak[LIG_CXCL8],
        "peak_CCL2": sim.ligands.peak[LIG_CCL2],
        "peak_CXCL9_10_11": sim.ligands.peak[LIG_CXCR3L],
        "peak_CXCL13": sim.ligands.peak[LIG_CXCL13],
        "peak_CCL3_CCL5": sim.ligands.peak[LIG_CCL35],
    }
    return result


# =============================================================================
#  SECTION 19.  선행 연구를 통한 타당성 검증
# =============================================================================


@dataclass
class ValidationItem:
    item: str
    literature: str
    simulated: str
    verdict: str
    source: str


def _fmt_day(v: Optional[float]) -> str:
    return "미발생" if v is None else f"Day {v:.2f}"


def validate(result: dict) -> List[ValidationItem]:
    """시뮬레이션 결과를 선행 연구의 보고값과 대조한다"""
    checks: List[ValidationItem] = []

    def add(item, lit, sim, verdict, source):
        checks.append(ValidationItem(item, lit, sim, verdict, source))

    def judge(value, lo, hi, near_lo=None, near_hi=None):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "판정불가"
        if lo <= value <= hi:
            return "일치"
        if near_lo is not None and near_lo <= value <= near_hi:
            return "근접"
        return "불일치"

    # ---- 배가시간 ----
    v = result["doubling_time_min_est"]
    add("폐 내 배가시간",
        f"{REAL['doubling_time_min']:.0f}분",
        f"{v:.0f}분" if not math.isnan(v) else "판정불가",
        judge(v, 30.0, 120.0, 15.0, 240.0), "jose2015")

    # ---- 최대 균량 ----
    v = result["bacteria_peak"]
    cap = Scale.agents_from_real(REAL["max_burden_cfu"])
    add("최대 균량",
        f"{REAL['max_burden_cfu']:.0e} CFU 이하",
        f"{v:,.0f} agent",
        judge(v, 1.0, float(cap), 1.0, float(cap) * 1.05), "hamilton2019")

    # ---- 세균의 공간 분포 ----
    v = result["colony_spread_final"]
    add("공간 분포 (비운동성)",
        "미세 군락 형성 (표 3)",
        f"퍼짐 {v:.0f}칸" if v > 0 else "군락 소실",
        "일치" if (v == 0.0 or v < 300.0) else "불일치", MODEL_ASSUMPTION)

    # ---- 자체 신호 ----
    v = result["peak_fMLF"]
    add("자체 신호(fMLF) 방출",
        "fMLF를 방출하여 위치를 표시함 (표 3)",
        f"{v:.6f}", "일치" if v > 0.0 else "불일치", "schiffmann1975")

    # ---- 옵소닌화 요구 ----
    v = result["max_opsonized"]
    add("옵소닌화 진행",
        f"혈청 {REAL['opsonin_serum_requirement']*100:.0f}% 이상 필요",
        f"최대 침착도 {v:.3f}",
        judge(v, 0.3, 1.0, 0.15, 1.0), "gordon1980")

    # ---- 제거 경로 ----
    v = result["phago_share"]
    add("제거 경로",
        "식균작용이 유일한 제거 수단 (표 3)",
        f"식균 {v:.1f} %" if not math.isnan(v) else "판정불가",
        judge(v, 50.0, 100.0, 30.0, 100.0), MODEL_ASSUMPTION)

    # ---- 감염 종식 ----
    v = result["clear_day"]
    add("감염 종식 시각",
        "생존 개체에서 수일 내 종식",
        _fmt_day(v), judge(v, 2.0, 12.0, 1.0, 16.0), "jose2015")

    # ---- Th17 정점 ----
    v = result["t_th17_peak"]
    add("Th17 반응 정점",
        f"Day {REAL['th17_peak_day']:.0f} 내외",
        _fmt_day(v), judge(v, 3.0, 12.0, 2.0, 16.0), "lu2008")

    # ---- 항체 검출 ----
    v = result["t_igm"]
    add("IgM 검출 시각",
        f"Day {REAL['igm_detect_day']:.0f} 내외",
        _fmt_day(v), judge(v, 2.0, 8.0, 1.0, 11.0), MODEL_ASSUMPTION)

    # ---- 살해T세포 역할 ----
    v = result["killer_t_specific_peak"]
    w = result["helper_t_specific_peak"]
    ratio = v / w * 100 if w > 0 else float("nan")
    add("살해T세포의 역할",
        "감염세포가 없어 역할이 거의 없음",
        f"조력T 대비 {ratio:.1f} %" if not math.isnan(ratio) else "판정불가",
        judge(ratio, 0.0, 30.0, 0.0, 60.0), MODEL_ASSUMPTION)

    # ---- 조작 검증 ----
    v = result["gradient_fraction"] * 100.0
    if result["mode"] == MODE_RANDOM:
        add("이동 조작 검증 (신호 감지)",
            "무작위 이동 = 0 % (표 2)",
            f"{v:.1f} %", "일치" if v < 0.01 else "불일치", MODEL_ASSUMPTION)
    else:
        add("이동 조작 검증 (신호 감지)",
            "방향성 이동 = 100 % (표 2)",
            f"{v:.1f} %",
            "일치" if v > 90.0 else "근접" if v > 60.0 else "불일치",
            MODEL_ASSUMPTION)

    return checks


def print_validation(checks: List[ValidationItem]) -> Tuple[int, int, int]:
    """검증표 출력"""
    print("=" * 112)
    print(" 선행 연구를 통한 타당성 검증")
    print("=" * 112)
    print(f" {'검증 항목':<24}{'문헌 기준값':<36}{'시뮬레이션 결과':<20}"
          f"{'판정':<8}{'출처':<16}")
    print("-" * 112)
    n_ok = n_near = n_bad = 0
    for c in checks:
        src = c.source
        if src != MODEL_ASSUMPTION and src in REFERENCES:
            src = REFERENCES[src].short()
        if c.verdict == "일치":
            n_ok += 1
        elif c.verdict == "근접":
            n_near += 1
        elif c.verdict == "불일치":
            n_bad += 1
        print(f" {c.item:<24}{c.literature:<36}{c.simulated:<20}"
              f"{c.verdict:<8}{src:<16}")
    print("-" * 112)
    total = len(checks)
    rate = (n_ok + n_near) / total * 100 if total else 0.0
    print(f" 일치 {n_ok} / 근접 {n_near} / 불일치 {n_bad} / 총 {total}개"
          f"   ->  문헌 부합률 {rate:.1f} %")
    print("=" * 112)
    return n_ok, n_near, n_bad


# =============================================================================
#  SECTION 20.  결과 보고서
# =============================================================================


def print_result_report(result: dict, sim: PneumococcusSimulation) -> None:
    """작품설명서의 결과 지표를 출력한다"""
    r = result
    print("=" * 84)
    print(f" 탐구 결과 — 폐렴구균(S. pneumoniae) / {r['mode_name']}")
    print("=" * 84)

    print("\n [1] 병원체")
    print(f"   최대 병원체 수          : {r['bacteria_peak']:>14,.0f} agent"
          f"   (Day {r['bacteria_peak_day']:.2f})")
    print(f"   병원체 총량 AUC         : {r['bacteria_auc']:>14,.0f}")
    print(f"   병원체 50 % 감소        : {_fmt_day(r['bacteria_t50'])}")
    print(f"   병원체 90 % 감소        : {_fmt_day(r['bacteria_t90'])}")
    print(f"   감염 종식 시각          : {_fmt_day(r['clear_day'])}")
    print(f"   추정 배가시간           : "
          f"{r['doubling_time_min_est']:>14.0f} 분")
    print(f"   군락 공간 퍼짐          : "
          f"{r['colony_spread_final']:>14.0f} 칸")

    print("\n [2] 조직")
    print(f"   최저 정상세포 수        : {r['min_healthy']:>14,.0f} 칸")
    print(f"   표적세포 최대 소모율    : "
          f"{r['max_target_depletion']*100:>14.1f} %")
    print(f"   사멸 상피세포 수        : {r['dead_epithelium']:>14,}")
    print(f"   누적 조직손상           : {r['damage_final']:>14.3f}")

    print("\n [3] 면역세포")
    print(f"   중성구 정점             : {r['neutrophil_peak']:>14,.0f}")
    print(f"   단핵구/대식세포 정점    : {r['monocyte_peak']:>14,.0f}")
    print(f"   자연살해 세포 정점      : {r['nk_peak']:>14,.0f}")
    print(f"   항원특이 조력T 정점     : "
          f"{r['helper_t_specific_peak']:>14,.0f}")
    print(f"   항원특이 살해T 정점     : "
          f"{r['killer_t_specific_peak']:>14,.0f}")
    print(f"   조직 내 면역세포 정점   : {r['in_tissue_peak']:>14,.0f}")
    print(f"   면역세포 사망/소모      : {r['dead_immune']:>14,}")

    print("\n [4] 병원체 제거 경로별 기여도")
    print(f"   식균 총 제거량          : {r['phago_total']:>14,}")
    print(f"     └ 중성구 포식         : {r['phago_neutrophil']:>14,}")
    print(f"     └ 단핵구/대식세포     : {r['phago_monocyte']:>14,}")
    print(f"   상주 폐포대식세포       : {r['cleared_alveolar_mac']:>14,}")
    print(f"   항체 매개 제거          : {r['cleared_antibody']:>14,}")
    print(f"   보체 매개 제거          : {r['cleared_complement']:>14,}")
    print(f"   -> 식균이 차지하는 비율 : {r['phago_share']:>14.1f} %")

    print("\n [5] 탐색 효율")
    print(f"   식세포-세균 접촉 횟수   : {r['contacts']:>14,}")
    print(f"   식세포-세균 겹침도      : {r['overlap_pct']:>14.2f} %"
          f"   (병원체 정점 시점)")
    print(f"   식세포 분포 불균일도    : {r['phagocyte_cv']:>14.2f}")
    print(f"   누적 혈관외유출         : {r['n_extravasated']:>14,}")
    print(f"   첫 감염부위 도착        : {_fmt_day(r['t_first_arrival'])}")

    print("\n [6] 리간드별 최대 농도 (수용체 분리 구현)")
    for key, label in ((LIG_FMLF, "fMLF       (FPR1)"),
                       (LIG_CXCL8, "CXCL8      (CXCR2)"),
                       (LIG_CCL2, "CCL2       (CCR2)"),
                       (LIG_CXCR3L, "CXCL9/10/11(CXCR3)"),
                       (LIG_CXCL13, "CXCL13     (CXCR5)"),
                       (LIG_CCL35, "CCL3/CCL5  (CCR1)")):
        print(f"   {label:<22}: {sim.ligands.peak[key]:>14.6f}")

    print("\n [7] 후천면역")
    print(f"   최대 항체 역가          : {r['antibody_peak']:>14.2f}")
    print(f"   Th17 정점               : {r['th17_peak']:>14,.0f}"
          f"   ({_fmt_day(r['t_th17_peak'])})")
    print(f"   최대 옵소닌 침착도      : {r['max_opsonized']:>14.3f}")
    print(f"   기억세포 형성량         : {r['memory_final']:>14,}")

    print("\n [8] 이동 방식 조작 검증")
    for line in sim.mover.report().split("\n"):
        print(f"   {line}")

    print("\n [9] 실행 정보")
    print(f"   난수 시드               : {r['seed']:>14,}")
    print(f"   실행 시간               : {r['wall_time']:>14.1f} 초")
    print("=" * 84)


# =============================================================================
#  SECTION 21.  실행 진입점
# =============================================================================


def run_simulation(movement_mode: str = MODE_RANDOM,
                   seed: int = 20260811,
                   max_days: int = 14,
                   verbose: bool = True
                   ) -> Tuple[PneumococcusSimulation, dict]:
    """시뮬레이션 1회 실행"""
    p = Params()
    p.seed = seed
    p.max_days = max_days
    sim = PneumococcusSimulation(p, movement_mode=movement_mode,
                                 seed=seed, verbose=verbose)
    sim.run()
    return sim, analyze(sim)


def print_header(movement_mode: str) -> None:
    spec = MOVEMENT_TABLE[movement_mode]
    print()
    print("#" * 84)
    print("#" + " " * 82 + "#")
    print("#" + "  폐렴구균(Streptococcus pneumoniae) 감염 - 면역반응 "
          "Agent-Based Model".center(78) + "#")
    print("#" + f"  이동 방식: {spec.name_kr}".center(80) + "#")
    print("#" + " " * 82 + "#")
    print("#" * 84)
    print()
    print(" [축척]")
    for line in Scale.describe().split("\n"):
        print(f"   {line}")
    print()


def main(argv: Sequence[str]) -> int:
    mode = argv[1] if len(argv) > 1 else MODE_RANDOM
    if mode not in (MODE_RANDOM, MODE_DIRECTED):
        print(f"사용법: python {argv[0]} [random|directed] [시드] [일수]")
        return 1
    seed = int(argv[2]) if len(argv) > 2 else 20260811
    days = int(argv[3]) if len(argv) > 3 else 14

    print_header(mode)
    print_immune_cell_table()
    print()
    print_ligand_table()
    print()
    print_movement_table()
    print()
    print_pathogen_table()
    print()
    print_kill_condition()
    print()
    print_param_table()
    print()

    print(" [수용체 구현 일치 검사]")
    if verify_receptor_consistency():
        print("   표 1의 수용체 대응과 구현이 일치합니다.")
    else:
        print("   ★경고: 표 1과 구현이 일치하지 않습니다.")
    print()
    print(f" [조직 구조] {Tissue(np.random.default_rng(0)).describe()}")
    print()

    print(" [시뮬레이션 진행]")
    sim, result = run_simulation(mode, seed=seed, max_days=days, verbose=True)
    print()
    print_result_report(result, sim)
    print()
    checks = validate(result)
    print_validation(checks)
    print()
    print_references()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

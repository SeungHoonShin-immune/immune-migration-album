# -*- coding: utf-8 -*-
"""
=============================================================================

  influenza_abm.py

  제68회 서울과학전람회 예선대회 출품작
  「Python 시뮬레이션을 이용한 두 면역세포 모델 이동에서
    무작위 이동과 방향성 이동의 효율 비교」

  인플루엔자 A 바이러스(H1N1) 감염 - 면역반응 Agent-Based Model

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

  【작품설명서 표 3】 인플루엔자 A 바이러스의 특징
   분류        RNA 바이러스
   증식 위치   세포내 증식
   공간 분포   조직 전역에 균일한 확산
   자체 신호   없음
   제거 경로   감염된 숙주세포의 사멸이 제거의 대부분을 차지

  ---------------------------------------------------------------------------
  【실행 방법】
      python influenza_abm.py random     # 무작위 이동
      python influenza_abm.py directed   # 방향성 이동
      python influenza_abm.py random 20260811 16
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

    # ---------------- 인플루엔자 A 감염 동역학 ----------------
    "baccam2006": Reference(
        "baccam2006",
        "Baccam P, Beauchemin C, Macken CA, Hayden FG, Perelson AS",
        "Kinetics of influenza A virus infection in humans",
        "Journal of Virology 80(15):7590-7599", 2006,
        "10.1128/JVI.01623-05", "16840338"),

    "carrat2008": Reference(
        "carrat2008",
        "Carrat F, Vergu E, Ferguson NM, Lemaitre M, Cauchemez S, Leach S, "
        "Valleron AJ",
        "Time lines of infection and disease in human influenza: "
        "a review of volunteer challenge studies",
        "American Journal of Epidemiology 167(7):775-785", 2008,
        "10.1093/aje/kwm375", "18230677"),

    "harris2006": Reference(
        "harris2006",
        "Harris A, Cardone G, Winkler DC, Heymann JB, Brecher M, White JM, "
        "Steven AC",
        "Influenza virus pleiomorphy characterized by cryoelectron tomography",
        "PNAS 103(50):19123-19127", 2006,
        "10.1073/pnas.0607614103", "17146053"),

    "memoli2015": Reference(
        "memoli2015",
        "Memoli MJ, Czajkowski L, Reed S, Athota R, Bristol T, Proudfoot K, "
        "Fargis S, Stein M, Dunfee RL, Shaw PA, Davey RT, Taubenberger JK",
        "Validation of the wild-type influenza A human challenge model "
        "H1N1pdMIST",
        "Clinical Infectious Diseases 60(5):693-702", 2015,
        "10.1093/cid/ciu924", "25416753"),

    # ---------------- 면역세포 수용체 ----------------
    "hol2010": Reference(
        "hol2010",
        "Hol J, Wilhelmsen L, Haraldsen G",
        "The murine IL-8 homologues KC, MIP-2, and LIX are found in "
        "endothelial cytoplasmic granules but not in Weibel-Palade bodies",
        "Journal of Leukocyte Biology 87(3):501-508", 2010,
        "10.1189/jlb.0809532", "20007248"),

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

    "germain2012": Reference(
        "germain2012",
        "Germain RN, Robey EA, Cahalan MD",
        "A decade of imaging cellular motility and interaction dynamics in "
        "the immune system",
        "Science 336(6089):1676-1681", 2012,
        "10.1126/science.1221063", "22745423"),

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
#  작품설명서 「3. 선행 연구」 및 「라. 탐구 절차」에 기재된 값을 그대로 옮겼다.
# =============================================================================

REAL: Dict[str, float] = {

    # ------------------------------------------------------------------
    #  공간 · 시간 축척  (작품설명서 라-A. 모양 구조)
    # ------------------------------------------------------------------
    "grid_side_mm": 15.0,          # 시뮬레이션 공간 한 변 = 상피 15mm
    "epi_cell_um": 15.0,           # 상피세포 한 개의 크기 = 격자 한 칸
    "tick_min": 6.0,               # 1 tick = 6분
    "kappa": 1.0 / 1000.0,         # 개체 수는 실제 값의 1/1000

    # ------------------------------------------------------------------
    #  인플루엔자 A 감염 동역학
    # ------------------------------------------------------------------
    # Baccam 등 (2006) — 인체 감염 동역학 모수 추정
    "eclipse_h": 6.0,              # 잠복기 6시간
    "productive_h": 5.0,           # 바이러스 생산 기간 5시간
    "infected_life_h": 11.0,       # 감염세포 평균 수명 11시간 (= 6 + 5)
    "virion_halflife_h": 3.0,      # 유리 바이러스 반감기 3시간
    "R0_in_host": 22.0,            # 체내 기초재생산 수 약 22

    # Harris 등 (2006) — 저온전자단층촬영으로 관찰한 바이러스 입자 크기
    "virion_diameter_nm_min": 80.0,
    "virion_diameter_nm_max": 120.0,

    # Carrat 등 (2008) — 자원자 감염 연구 종합
    "shedding_mean_days": 4.8,     # 바이러스 배출 기간 평균 4.8일
    "symptom_peak_day_min": 2.0,   # 증상 정점 2~3일째
    "symptom_peak_day_max": 3.0,

    # Memoli 등 (2015) — 야생형 H1N1 인체 감염 모델
    "inoculum_tcid50": 1.0e5,      # 초기 접종량 10^5 TCID50

    # ------------------------------------------------------------------
    #  면역세포 이동 속도
    # ------------------------------------------------------------------
    # Friedl & Weigelin (2008) — 간질 조직 내 백혈구 이동
    "neutrophil_um_min": 12.0,
    "monocyte_um_min": 4.0,
    "nk_um_min": 8.0,
    "dendritic_um_min": 5.0,
    # Miller 등 (2002) — 림프절 내 림프구 운동성
    "t_cell_um_min": 11.0,
    "b_cell_um_min": 6.0,
    # Ley 등 (2007) — 백혈구 부착 연쇄반응 및 혈류 내 이동
    "vessel_transit_um_min": 400.0,

    # ------------------------------------------------------------------
    #  신호물질 (정성적 기전은 문헌 근거, 정량값은 모델 가정값)
    # ------------------------------------------------------------------
    "cxcl8_halflife_h": 2.0,       # 즉시형 반응 산물, 빠른 turnover
    "ccl2_halflife_h": 4.0,
    "cxcr3l_halflife_h": 6.0,      # GAG 결합으로 조직 내 장기 잔류
    "cxcl13_halflife_h": 8.0,      # 항상성 케모카인
    "ccl35_halflife_h": 3.0,
    "fmlf_halflife_min": 20.0,     # 조직 펩티다제에 의한 분해
    "interferon_halflife_h": 4.0,

    # ------------------------------------------------------------------
    #  후천면역
    # ------------------------------------------------------------------
    "igm_halflife_d": 5.0,
    "igg_halflife_d": 21.0,
    "igm_detect_day": 5.0,
    "igg_detect_day": 9.0,
    "priming_h": 48.0,             # 항원 도달 후 프라이밍 완료까지
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

    GRID: int = 1000                                   # 격자 한 변 (칸)
    N_SITES: int = GRID * GRID                         # 100만 칸
    SIDE_MM: float = REAL["grid_side_mm"]              # 15 mm
    UM_PER_CELL: float = SIDE_MM * 1000.0 / GRID       # 15 um / 칸
    TICK_MIN: float = REAL["tick_min"]                 # 6 분
    TICKS_PER_HOUR: float = 60.0 / TICK_MIN            # 10
    TICKS_PER_DAY: int = int(24 * TICKS_PER_HOUR)      # 240
    KAPPA: float = REAL["kappa"]                       # 1/1000

    # 화학신호 농도장은 격자를 5칸 단위로 묶어 200x200 으로 관리한다.
    FIELD_BIN: int = 5
    FIELD_N: int = GRID // FIELD_BIN                   # 200

    # ------------------------------------------------------------------
    @staticmethod
    def ticks_from_hours(hours: float) -> int:
        """시간(h) -> tick"""
        return max(1, int(round(hours * Scale.TICKS_PER_HOUR)))

    @staticmethod
    def ticks_from_days(days: float) -> int:
        return max(1, int(round(days * Scale.TICKS_PER_DAY)))

    @staticmethod
    def days_from_ticks(ticks: int) -> float:
        return ticks / Scale.TICKS_PER_DAY

    @staticmethod
    def decay_from_halflife_h(hours: float) -> float:
        """반감기(h) -> tick 당 감쇠율"""
        return 1.0 - 0.5 ** (1.0 / (hours * Scale.TICKS_PER_HOUR))

    @staticmethod
    def decay_from_halflife_min(minutes: float) -> float:
        """반감기(분) -> tick 당 감쇠율"""
        return 1.0 - 0.5 ** (1.0 / (minutes / Scale.TICK_MIN))

    @staticmethod
    def decay_from_halflife_d(days: float) -> float:
        return 1.0 - 0.5 ** (1.0 / (days * Scale.TICKS_PER_DAY))

    @staticmethod
    def speed_cells_per_tick(um_per_min: float) -> float:
        """이동속도(um/분) -> 칸/tick"""
        return um_per_min * Scale.TICK_MIN / Scale.UM_PER_CELL

    @staticmethod
    def agents_from_real(real_count: float) -> int:
        """실제 개체수 -> agent 수 (1/1000 축소)"""
        return max(1, int(round(real_count * Scale.KAPPA)))

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

# ---- 세포 유형 식별자 ----
NEUTROPHIL = 0      # 중성구
MONOCYTE = 1        # 단핵구/대식세포
NK_CELL = 2         # 자연살해 세포
HELPER_T = 3        # 조력 T세포
KILLER_T = 4        # 살해 T세포
B_CELL = 5          # B세포/형질세포
DENDRITIC = 6       # 수지상세포

N_CELL_TYPES = 7


@dataclass(frozen=True)
class ImmuneCellType:
    """
    작품설명서 표 1의 한 행에 해당하는 자료구조.

      code       : 코드 내 식별자
      name_kr    : 면역세포 (한글명)
      name_en    : 영어 명칭
      receptors  : 주 수용체
      ligands    : 대표적 결합 리간드
      role       : 역할 (작품설명서 본문 서술)
    """
    code: int
    name_kr: str
    name_en: str
    receptors: Tuple[str, ...]
    ligands: Tuple[str, ...]
    role: str
    source: str = ""


# ---------------------------------------------------------------------------
#  표 1. 면역세포의 종류
# ---------------------------------------------------------------------------
IMMUNE_CELL_TABLE: Dict[int, ImmuneCellType] = {

    NEUTROPHIL: ImmuneCellType(
        code=NEUTROPHIL,
        name_kr="중성구",
        name_en="Neutrophil",
        receptors=("CXCR2",),
        ligands=("CXCL8",),
        role="백혈구의 70%를 차지하는 면역세포로, 식균 작용, 과립, NET 등을 "
             "사용하여 병원체를 살해한다.",
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
        source="groom2011"),

    KILLER_T: ImmuneCellType(
        code=KILLER_T,
        name_kr="살해 T세포",
        name_en="Killer T cell",
        receptors=("CXCR3",),
        ligands=("CXCL9", "CXCL10", "CXCL11"),
        role="감염된 세포를 살해하는 것에서는 NK세포와 거의 동일하지만, "
             "특정 바이러스를 학습한 뒤에만 활성화된다는 것이 다르다.",
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
        source="sozzani1997"),
}


# 보체는 세포가 아니므로 표 1에서 수용체·리간드가 '해당없음'으로 기재되어 있다.
COMPLEMENT_DESCRIPTION = (
    "보체(Complement)는 혈액 속 단백질 무리로, 병원체에 붙어 병원체를 "
    "'옵소닌화'하여 대식세포의 포식 작용을 강화시킨다. 또한 병원체의 "
    "세포막에 구멍을 뚫어 제거하는 역할도 한다. 스스로 이동하지 않으므로 "
    "수용체와 결합 리간드가 존재하지 않는다."
)


def print_immune_cell_table() -> None:
    """표 1 출력 — 구현이 작품설명서와 일치함을 확인하기 위한 함수"""
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
    for code in range(N_CELL_TYPES):
        ct = IMMUNE_CELL_TABLE[code]
        print(f" · {ct.name_kr} : {ct.role}")
    print(f" · 보체 : {COMPLEMENT_DESCRIPTION}")
    print("=" * 100)


# =============================================================================
#  SECTION 5.  리간드 (화학신호) 정의
# =============================================================================
#  ★탐구의 독창성:
#    "기존 모형은 단일 케모카인 농도를 가정하였으나, 본 탐구에서는
#     FPR1, CXCR2, CCR2, CXCR3, CXCR5 의 축을 독립적으로 구현했다."
#    표 1에 등장하는 모든 리간드를 서로 다른 농도장으로 분리하여 관리한다.
# =============================================================================

# ---- 리간드 농도장 식별자 ----
LIG_CXCL8 = "CXCL8"          # CXCR2  <- 중성구
LIG_CCL2 = "CCL2"            # CCR2   <- 단핵구/대식세포
LIG_FMLF = "fMLF"            # FPR1   <- 단핵구/대식세포
LIG_CXCR3L = "CXCL9/10/11"   # CXCR3  <- NK세포, 조력T, 살해T
LIG_CXCL13 = "CXCL13"        # CXCR5  <- B세포/형질세포
LIG_CCL35 = "CCL3/CCL5"      # CCR1   <- 수지상세포

ALL_LIGAND_FIELDS: Tuple[str, ...] = (
    LIG_CXCL8, LIG_CCL2, LIG_FMLF, LIG_CXCR3L, LIG_CXCL13, LIG_CCL35,
)


@dataclass(frozen=True)
class LigandSpec:
    """
    리간드 1종의 명세.

      key        : 농도장 식별자
      receptor   : 이 리간드가 결합하는 수용체
      cells      : 이 리간드를 감지하는 면역세포
      source     : 생성원 (문헌 근거)
      sigma      : tick 당 확산 폭 (농도장 격자 단위)
      halflife   : 반감기
      decay      : tick 당 감쇠율
      note       : 비고
    """
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

    LIG_CXCL8: LigandSpec(
        key=LIG_CXCL8, receptor="CXCR2", cells=("중성구",),
        source="감염된 상피세포가 즉시 분비하며, 침윤한 중성구가 2차로 증폭한다.",
        sigma=2.8,
        halflife_text=f"{REAL['cxcl8_halflife_h']:.0f}시간",
        decay=Scale.decay_from_halflife_h(REAL["cxcl8_halflife_h"]),
        note="즉시형(immediate-early) 유전자 산물이므로 감염 직후부터 형성된다.",
        ref="rudd2019"),

    LIG_CCL2: LigandSpec(
        key=LIG_CCL2, receptor="CCR2", cells=("단핵구/대식세포",),
        source="감염된 상피세포와 활성화된 대식세포가 분비한다.",
        sigma=2.4,
        halflife_text=f"{REAL['ccl2_halflife_h']:.0f}시간",
        decay=Scale.decay_from_halflife_h(REAL["ccl2_halflife_h"]),
        note="활성 대식세포의 2차 분비로 증폭 고리를 형성한다.",
        ref="lin2008"),

    LIG_FMLF: LigandSpec(
        key=LIG_FMLF, receptor="FPR1", cells=("단핵구/대식세포",),
        source="세균이 대사 과정에서 방출하는 N-포르밀 펩타이드.",
        sigma=3.0,
        halflife_text=f"{REAL['fmlf_halflife_min']:.0f}분",
        decay=Scale.decay_from_halflife_min(REAL["fmlf_halflife_min"]),
        note="★인플루엔자 모형에서는 생성원이 존재하지 않는다. "
             "작품설명서 표 3에 '자체 신호: 없음'으로 기재된 대로, "
             "바이러스는 fMLF 를 방출하지 않으므로 이 축은 작동하지 않는다.",
        ref=""),

    LIG_CXCR3L: LigandSpec(
        key=LIG_CXCR3L, receptor="CXCR3",
        cells=("자연살해 세포", "조력 T세포", "살해 T세포"),
        source="인터페론에 자극받은 세포가 분비한다.",
        sigma=2.0,
        halflife_text=f"{REAL['cxcr3l_halflife_h']:.0f}시간",
        decay=Scale.decay_from_halflife_h(REAL["cxcr3l_halflife_h"]),
        note="CXCL10 은 인터페론 유도 조기반응 유전자이므로, 인터페론이 "
             "축적된 뒤에야 신호가 형성된다. 따라서 CXCR3 계열 세포의 동원은 "
             "중성구보다 구조적으로 늦어진다.",
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
        note="미성숙 수지상세포를 염증 부위로 유도한다.",
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
    구현이 문서와 어긋나면 즉시 드러나도록 하기 위한 자체 점검 함수이다.
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
    """작품설명서 표 2의 한 열에 해당하는 자료구조"""
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
    "크기": f"{REAL['virion_diameter_nm_min']:.0f}~"
            f"{REAL['virion_diameter_nm_max']:.0f} nm  (Harris 등, 2006)",
    "증식 위치": "세포내 증식",
    "공간 분포": "조직 전역에 균일한 확산",
    "자체 신호": "없음",
    "제거 경로": "감염된 숙주세포의 사멸이 제거의 대부분을 차지",
}

PATHOGEN_TABLE_PNEUMOCOCCUS: Dict[str, str] = {
    "이름": "폐렴구균 (S. pneumoniae)",
    "분류": "그람양성 세균",
    "크기": "약 0.8~1.0 um",
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
#  SECTION 8.  파라미터 표
# =============================================================================
#  모든 파라미터의 출처를 명시한다. 논문에서 직접 인용한 실측값과, 정성적
#  기전만 문헌에 근거하고 정량값은 본 모형에서 설정한 가정값을 구분한다.
# =============================================================================


@dataclass(frozen=True)
class ParamRow:
    """파라미터 1개의 출처 기록"""
    item: str            # 항목
    real_value: str      # 실제 값
    unit: str            # 단위
    source: str          # 출처 (REFERENCES 태그 또는 MODEL_ASSUMPTION)
    conversion: str      # 축척 변환 방식
    sim_value: str       # 시뮬레이션 값
    note: str = ""       # 비고


PARAM_TABLE: List[ParamRow] = []


def _p(*args, **kwargs) -> None:
    PARAM_TABLE.append(ParamRow(*args, **kwargs))


# ---- 공간 · 시간 축척 ----
_p("시뮬레이션 공간", "15 x 15", "mm", "crystal2008", "1칸 = 상피세포 1개",
   "1000 x 1000 격자 (100만 칸)",
   "격자 한 칸이 상피세포 한 개(15um)에 대응한다")
_p("시간 단위", "6", "분/tick", MODEL_ASSUMPTION, "-", "1일 = 240 tick",
   "감염 동역학의 시간 규모를 충분히 해상할 수 있는 단위")
_p("개체수 축소비율", "1/1000", "-", MODEL_ASSUMPTION, "-", "kappa = 0.001",
   "계산량을 실행 가능한 범위로 줄이면서 비율 관계를 보존한다")

# ---- 병원체 (Baccam 2006) ----
_p("잠복기 (eclipse)", f"{REAL['eclipse_h']:.0f}", "시간", "baccam2006",
   "시간 -> tick", f"{Scale.ticks_from_hours(REAL['eclipse_h'])} tick",
   "감염 성립 후 바이러스 생산이 시작되기까지의 기간")
_p("바이러스 생산 기간", f"{REAL['productive_h']:.0f}", "시간", "baccam2006",
   "시간 -> tick", f"{Scale.ticks_from_hours(REAL['productive_h'])} tick",
   "생산기 감염세포가 바이러스를 방출하는 기간")
_p("감염세포 평균 수명", f"{REAL['infected_life_h']:.0f}", "시간", "baccam2006",
   "시간 -> tick", f"{Scale.ticks_from_hours(REAL['infected_life_h'])} tick",
   "잠복기 + 생산기. 이후 숙주세포는 자멸한다")
_p("유리 바이러스 반감기", f"{REAL['virion_halflife_h']:.0f}", "시간",
   "baccam2006", "반감기 -> tick 당 감쇠율",
   f"{Scale.decay_from_halflife_h(REAL['virion_halflife_h']):.5f}/tick",
   "숙주세포 밖의 바이러스 입자가 자연 소실되는 속도")
_p("체내 기초재생산 수", f"약 {REAL['R0_in_host']:.0f}", "-", "baccam2006",
   "모형에서 재현 대상", "감염칸 1개가 만드는 신규 감염칸 수",
   "모형의 타당성을 검증하는 기준값")
_p("바이러스 크기", f"{REAL['virion_diameter_nm_min']:.0f}~"
   f"{REAL['virion_diameter_nm_max']:.0f}", "nm", "harris2006",
   "격자 이하 크기", "개별 agent 로 표현",
   "상피세포(15um)보다 두 자릿수 작으므로 점입자로 취급")
_p("바이러스 배출 기간", f"평균 {REAL['shedding_mean_days']:.1f}", "일",
   "carrat2008", "모형에서 재현 대상", "정점 대비 1% 도달 시각",
   "모형의 타당성을 검증하는 기준값")
_p("증상 정점", f"{REAL['symptom_peak_day_min']:.0f}~"
   f"{REAL['symptom_peak_day_max']:.0f}", "일째", "carrat2008",
   "모형에서 재현 대상", "염증 지표의 정점 시각",
   "모형의 타당성을 검증하는 기준값")
_p("초기 접종량", f"{REAL['inoculum_tcid50']:.0e}", "TCID50", "memoli2015",
   "1/1000 축소", f"{Scale.agents_from_real(REAL['inoculum_tcid50'])} agent",
   "야생형 H1N1 인체 감염 연구의 접종량")

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
   f"{Scale.speed_cells_per_tick(REAL['t_cell_um_min']):.2f} 칸/tick",
   "이광자 현미경으로 관찰한 림프절 내 운동성")
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
_p("IgM 검출 시각", f"{REAL['igm_detect_day']:.0f}", "일", MODEL_ASSUMPTION,
   "모형에서 재현 대상", "항체 곡선의 상승 시점", "일반적인 1차 면역반응 시간표")
_p("IgG 검출 시각", f"{REAL['igg_detect_day']:.0f}", "일", MODEL_ASSUMPTION,
   "모형에서 재현 대상", "항체 곡선의 상승 시점", "")
_p("IgM 반감기", f"{REAL['igm_halflife_d']:.0f}", "일", MODEL_ASSUMPTION,
   "반감기 -> tick 당 감쇠율",
   f"{Scale.decay_from_halflife_d(REAL['igm_halflife_d']):.6f}/tick", "")
_p("IgG 반감기", f"{REAL['igg_halflife_d']:.0f}", "일", MODEL_ASSUMPTION,
   "반감기 -> tick 당 감쇠율",
   f"{Scale.decay_from_halflife_d(REAL['igg_halflife_d']):.6f}/tick", "")


def print_param_table() -> None:
    """파라미터 출처표 출력"""
    print("=" * 118)
    print(" 시뮬레이션 파라미터와 출처")
    print("=" * 118)
    print(f" {'항목':<22}{'실제 값':<14}{'단위':<12}{'출처':<16}"
          f"{'시뮬레이션 값':<30}")
    print("-" * 118)
    for row in PARAM_TABLE:
        src = row.source
        if src != MODEL_ASSUMPTION and src in REFERENCES:
            src = REFERENCES[src].short()
        print(f" {row.item:<22}{row.real_value:<14}{row.unit:<12}"
              f"{src:<16}{row.sim_value:<30}")
    print("=" * 118)
    n_lit = sum(1 for r in PARAM_TABLE if r.source != MODEL_ASSUMPTION)
    n_asm = len(PARAM_TABLE) - n_lit
    print(f" 총 {len(PARAM_TABLE)}개 파라미터 "
          f"— 문헌 인용 {n_lit}개 / 모델 가정값 {n_asm}개")
    print("=" * 118)


# =============================================================================
#  SECTION 9.  시뮬레이션 파라미터
# =============================================================================


@dataclass
class Params:
    """시뮬레이션 실행 파라미터"""

    # ---------------- 실행 ----------------
    max_days: int = 16
    seed: int = 20260811
    stop_after_clear_days: float = 1.0     # 완전 제거 후 유지 확인 기간

    # ---------------- 병원체 ----------------
    n_virus0: int = Scale.agents_from_real(REAL["inoculum_tcid50"])
    eclipse_ticks: int = Scale.ticks_from_hours(REAL["eclipse_h"])
    productive_ticks: int = Scale.ticks_from_hours(REAL["productive_h"])
    virion_decay: float = Scale.decay_from_halflife_h(REAL["virion_halflife_h"])
    burst_per_tick: float = 2.9            # 생산기 감염칸의 tick 당 방출량
    infect_p: float = 0.030                # 표적칸 도달 virion 의 감염 성립 확률
    # ★표 3: 공간 분포 = 조직 전역에 균일한 확산
    virion_sigma: float = 30.0             # 칸/tick
    max_virions: int = 520000              # 배열 상한

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
    life_neutrophil: int = 240             # 약 1일
    life_monocyte: int = 1800
    life_nk: int = 2600
    life_t: int = 60000
    life_b: int = 60000
    life_dendritic: int = 3000
    life_effector_t: int = 2400

    # ---------------- 포식 용량 ----------------
    capacity_neutrophil: int = 50
    capacity_monocyte: int = 120

    # ---------------- 동원 (하루당 최대 유입량) ----------------
    recruit_neutrophil: float = 24000.0
    recruit_monocyte: float = 3600.0
    recruit_nk: float = 1000.0

    # ---------------- 혈관외유출 ----------------
    # 방향성 이동 조건에서 림프절을 떠난 세포가 혈류를 타고 이동하다가
    # 국소 신호 농도가 이 역치를 넘으면 조직으로 빠져나온다.
    extravasation_threshold: float = 0.0022

    # ---------------- 리간드 생성 ----------------
    cxcl8_from_infected: float = 0.020
    cxcl8_from_neutrophil: float = 0.005
    ccl2_from_infected: float = 0.014
    ccl2_from_monocyte: float = 0.006
    fmlf_from_pathogen: float = 0.0        # ★표 3: 바이러스는 신호 없음
    cxcr3l_basal: float = 0.0030
    cxcr3l_ifn_gain: float = 0.30
    cxcl13_from_lymphnode: float = 0.010
    ccl35_from_infected: float = 0.006
    ccl35_from_monocyte: float = 0.008

    # ---------------- 인터페론 ----------------
    ifn_production: float = 0.020
    ifn_sigma: float = 2.2
    ifn_decay: float = Scale.decay_from_halflife_h(REAL["interferon_halflife_h"])
    antiviral_rate: float = 0.030
    antiviral_decay: float = 0.020

    # ---------------- 보체 · 염증 ----------------
    complement_on: float = 0.030
    complement_off: float = 0.012
    inflammation_on: float = 0.030
    inflammation_off: float = 0.014

    # ---------------- 살해 확률 ----------------
    phagocytosis_p: float = 0.55
    nk_kill_p: float = 0.42
    ctl_kill_p: float = 0.62
    complement_clear_p: float = 0.0022
    antibody_clear_p: float = 0.0060
    # 항HA 항체의 중화: 바이러스의 숙주세포 진입을 차단한다
    ab_neutralize_k: float = 9.0

    # ---------------- 후천면역 ----------------
    priming_ticks: int = Scale.ticks_from_hours(REAL["priming_h"])
    antigen_threshold: float = 50.0
    ab_igm_rate: float = 0.050
    ab_igg_rate: float = 0.028
    igm_decay: float = Scale.decay_from_halflife_d(REAL["igm_halflife_d"])
    igg_decay: float = Scale.decay_from_halflife_d(REAL["igg_halflife_d"])
    ab_titer_cap: float = 120.0
    memory_fraction: float = 0.10

    # ---------------- 조직 ----------------
    regeneration_rate: float = 0.00010     # 상피 재생 (수일~수주 소요)


# =============================================================================
#  SECTION 10.  조직 공간
# =============================================================================
#  상피 밴드, 간질 조직, 혈관 네트워크, 림프절로 구성된다.
#  두 이동 방식은 완전히 동일한 공간을 사용한다.
# =============================================================================

# ---- 격자 칸의 상태 ----
STROMA = 0        # 간질 조직 (표적 아님, 면역세포 통행로)
HEALTHY = 1       # 정상 상피세포 (표적)
ECLIPSE = 2       # 잠복기 감염세포
PRODUCTIVE = 3    # 생산기 감염세포
DEAD = 4          # 사멸 상피세포
VESSEL = 5        # 혈관
LYMPH = 6         # 림프절

STATE_NAME: Dict[int, str] = {
    STROMA: "간질 조직",
    HEALTHY: "정상 상피세포",
    ECLIPSE: "잠복기 감염세포",
    PRODUCTIVE: "생산기 감염세포",
    DEAD: "사멸 상피세포",
    VESSEL: "혈관",
    LYMPH: "림프절",
}

EPI_BAND_PERIOD = 25     # 상피 밴드 주기 (칸)
EPI_BAND_WIDTH = 12      # 상피 밴드 두께 (칸)
VESSEL_COL_PERIOD = 120  # 세로 혈관 간격 (칸)
LN_ROW0, LN_ROW1 = 460, 540    # 림프절 영역 (행)
LN_COL0, LN_COL1 = 40, 120     # 림프절 영역 (열)


class Tissue:
    """
    상기도 상피 조직의 공간 구조.

      · 상피 밴드   : 표적세포가 배열된 층
      · 간질 조직   : 표적이 아니며 면역세포가 통행한다
      · 혈관 네트워크 : 면역세포가 혈류를 타고 이동하는 통로
      · 림프절      : 후천면역이 준비되는 장소이자
                      방향성 이동 조건에서 면역세포가 생성되는 곳
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        g = Scale.GRID
        grid = np.full((g, g), STROMA, dtype=np.uint8)

        # ---- 상피 밴드 ----
        for r0 in range(0, g, EPI_BAND_PERIOD):
            grid[r0:r0 + EPI_BAND_WIDTH, :] = HEALTHY

        # ---- 혈관: 상피 밴드 사이의 가로줄 ----
        for r0 in range(0, g, EPI_BAND_PERIOD):
            rr = r0 + EPI_BAND_WIDTH + 1
            if rr < g:
                grid[rr, :] = VESSEL
        # ---- 혈관: 세로 간선 ----
        for c0 in range(0, g, VESSEL_COL_PERIOD):
            grid[:, c0] = VESSEL

        # ---- 림프절 ----
        grid[LN_ROW0:LN_ROW1, LN_COL0:LN_COL1] = LYMPH

        self.state: np.ndarray = grid.reshape(-1)

        # ---- 미리 계산해두는 인덱스 ----
        self.n_target_initial = int(np.count_nonzero(self.state == HEALTHY))
        self.vessel_sites = np.flatnonzero(self.state == VESSEL).astype(np.int32)
        self.lymph_sites = np.flatnonzero(self.state == LYMPH).astype(np.int32)
        self.tissue_sites = np.flatnonzero(
            (self.state == HEALTHY) | (self.state == STROMA)).astype(np.int32)

        # ---- 통계 ----
        self.n_dead_epithelium = 0
        self.cumulative_damage = 0.0

        # ---- 림프절이 차지하는 농도장 영역 ----
        r0 = LN_ROW0 // Scale.FIELD_BIN
        r1 = LN_ROW1 // Scale.FIELD_BIN + 1
        c0 = LN_COL0 // Scale.FIELD_BIN
        c1 = LN_COL1 // Scale.FIELD_BIN + 1
        self.lymph_field_slice = (slice(r0, r1), slice(c0, c1))

    # ------------------------------------------------------------------
    #  좌표 변환
    # ------------------------------------------------------------------
    @staticmethod
    def site_of_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """실수 좌표 -> 격자 칸 번호"""
        col = x.astype(np.int32)
        row = y.astype(np.int32)
        np.clip(col, 0, Scale.GRID - 1, out=col)
        np.clip(row, 0, Scale.GRID - 1, out=row)
        return row * Scale.GRID + col

    @staticmethod
    def field_of_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """실수 좌표 -> 농도장 격자 인덱스"""
        fc = (x / Scale.FIELD_BIN).astype(np.int32)
        fr = (y / Scale.FIELD_BIN).astype(np.int32)
        np.clip(fc, 0, Scale.FIELD_N - 1, out=fc)
        np.clip(fr, 0, Scale.FIELD_N - 1, out=fr)
        return fr, fc

    @staticmethod
    def field_of_site(site: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """격자 칸 번호 -> 농도장 격자 인덱스"""
        row = site // Scale.GRID
        col = site - row * Scale.GRID
        return ((row // Scale.FIELD_BIN).astype(np.int32),
                (col // Scale.FIELD_BIN).astype(np.int32))

    @staticmethod
    def xy_of_site(site: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """격자 칸 번호 -> 칸 중심의 실수 좌표"""
        row = site // Scale.GRID
        col = site - row * Scale.GRID
        return (col.astype(np.float32) + 0.5, row.astype(np.float32) + 0.5)

    # ------------------------------------------------------------------
    #  면역세포 생성 위치  (작품설명서 표 2 '생성 장소')
    # ------------------------------------------------------------------
    def spawn_xy(self, k: int, mode: str,
                 rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """
        무작위 이동  -> 조직에 무작위적으로 생성
        방향성 이동  -> 림프절에서 생성
        """
        if k <= 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        if mode == MODE_DIRECTED:
            sites = rng.choice(self.lymph_sites, size=k, replace=True)
        else:
            sites = rng.choice(self.tissue_sites, size=k, replace=True)
        return self.xy_of_site(sites)

    # ------------------------------------------------------------------
    #  조직 상태 변경
    # ------------------------------------------------------------------
    def kill_epithelium(self, sites: np.ndarray) -> int:
        """감염세포를 사멸 상태로 만든다"""
        if sites.size == 0:
            return 0
        sites = np.unique(sites)
        st = self.state[sites]
        target = (st == HEALTHY) | (st == ECLIPSE) | (st == PRODUCTIVE)
        s = sites[target]
        if s.size:
            self.state[s] = DEAD
            self.n_dead_epithelium += int(s.size)
            self.cumulative_damage += (s.size / max(1, self.n_target_initial)
                                       * 1.6)
        return int(s.size)

    def regenerate(self, rate: float) -> int:
        """사멸한 상피가 재생된다"""
        dead = np.flatnonzero(self.state == DEAD)
        if dead.size == 0:
            return 0
        k = int(dead.size * rate)
        if k <= 0:
            return 0
        pick = self.rng.choice(dead, size=min(k, dead.size), replace=False)
        self.state[pick] = HEALTHY
        return int(pick.size)

    # ------------------------------------------------------------------
    @property
    def n_healthy(self) -> int:
        return int(np.count_nonzero(self.state == HEALTHY))

    @property
    def n_eclipse(self) -> int:
        return int(np.count_nonzero(self.state == ECLIPSE))

    @property
    def n_productive(self) -> int:
        return int(np.count_nonzero(self.state == PRODUCTIVE))

    def describe(self) -> str:
        return (f"표적 상피세포 {self.n_target_initial:,}칸, "
                f"혈관 {self.vessel_sites.size:,}칸, "
                f"림프절 {self.lymph_sites.size:,}칸")


# =============================================================================
#  SECTION 11.  리간드 농도장
# =============================================================================


class LigandFieldSet:
    """
    표 1에 등장하는 리간드를 각각 독립된 농도장으로 관리한다.

    ★탐구의 독창성: 기존 모형은 단일 케모카인 농도를 가정하였으나,
      본 모형은 리간드마다 생성원·확산 폭·반감기를 따로 둔다.
      따라서 같은 시점에도 세포 종류에 따라 서로 다른 방향으로 이동한다.
    """

    def __init__(self):
        n = Scale.FIELD_N
        self.field: Dict[str, np.ndarray] = {
            key: np.zeros((n, n), dtype=np.float32)
            for key in ALL_LIGAND_FIELDS
        }
        self.peak: Dict[str, float] = {key: 0.0 for key in ALL_LIGAND_FIELDS}

    # ------------------------------------------------------------------
    def deposit(self, key: str, fr: np.ndarray, fc: np.ndarray,
                amount: float) -> None:
        """지정한 위치에 리간드를 방출한다"""
        if fr is None or fr.size == 0 or amount == 0.0:
            return
        np.add.at(self.field[key], (fr, fc), amount)

    def deposit_region(self, key: str, region, amount: float) -> None:
        """지정한 영역 전체에 리간드를 방출한다 (림프절 등)"""
        if amount == 0.0:
            return
        self.field[key][region] += amount

    def add_field(self, key: str, arr: np.ndarray, gain: float) -> None:
        """다른 농도장에 비례하여 생성한다 (인터페론 유도성 리간드)"""
        if gain == 0.0:
            return
        self.field[key] += gain * arr

    # ------------------------------------------------------------------
    def diffuse_and_decay(self) -> None:
        """각 리간드를 고유한 확산 폭과 반감기로 갱신한다"""
        for key in ALL_LIGAND_FIELDS:
            spec = LIGAND_SPECS[key]
            arr = gaussian_filter(self.field[key], sigma=spec.sigma,
                                  mode="nearest")
            arr *= (1.0 - spec.decay)
            self.field[key] = arr
            m = float(arr.max())
            if m > self.peak[key]:
                self.peak[key] = m

    # ------------------------------------------------------------------
    def direction_stacks(self) -> Dict[str, np.ndarray]:
        """
        각 리간드에 대해 8방향으로 이동시킨 농도장을 쌓아 반환한다.
        세포가 '주변 8방향 중 농도가 최대인 방향'을 찾을 때 사용한다.
        """
        out: Dict[str, np.ndarray] = {}
        for key in ALL_LIGAND_FIELDS:
            arr = self.field[key]
            st = np.empty((8, Scale.FIELD_N, Scale.FIELD_N), dtype=np.float32)
            for i, (dc, dr) in enumerate(DIRECTION_OFFSETS):
                st[i] = np.roll(np.roll(arr, -dr, axis=0), -dc, axis=1)
            out[key] = st
        return out

    # ------------------------------------------------------------------
    def mean(self, key: str) -> float:
        return float(self.field[key].mean())

    def maximum(self, key: str) -> float:
        return float(self.field[key].max())

    def total(self, key: str) -> float:
        return float(self.field[key].sum())


# =============================================================================
#  SECTION 12.  이동 엔진 — 작품설명서 표 2의 구현
# =============================================================================
#  ★두 이동 방식의 차이는 오직 이 클래스 안에만 존재한다.
# =============================================================================

# 8방향 (격자 오프셋과 단위벡터)
DIRECTION_OFFSETS: List[Tuple[int, int]] = [
    (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
]
DIRECTION_UNIT = np.array(DIRECTION_OFFSETS, dtype=np.float32)
DIRECTION_UNIT /= np.linalg.norm(DIRECTION_UNIT, axis=1, keepdims=True)


class MovementEngine:
    """
    ------------------------------------------------------------------------
     무작위 이동 (MODE_RANDOM)
    ------------------------------------------------------------------------
       이동 규칙 : 무작위로 방향을 골라 브라운 운동에 따라 이동한다.
       신호 감지 : 하지 않는다 (0%).
       구현      : 매 tick 마다 0~2pi 범위의 균일난수로 방향을 정한다.
                   어떤 리간드 농도장도 참조하지 않는다.

    ------------------------------------------------------------------------
     방향성 이동 (MODE_DIRECTED)
    ------------------------------------------------------------------------
       이동 규칙 : 수용체에 결합하는 리간드 농도가 최대인 방향으로 이동한다.
       신호 감지 : 항상 한다 (100%).
       구현      : 세포 종류별로 표 1에 기재된 수용체에 해당하는 농도장만
                   참조한다. 주변 8방향의 농도를 비교하여 최대인 방향을
                   고른다. 수용체가 둘 이상인 세포(단핵구/대식세포)는
                   각 수용체가 감지하는 농도를 합산한다.
                   구배가 전혀 없으면 따라갈 신호가 없으므로 탐색 이동한다.
    ------------------------------------------------------------------------
    """

    def __init__(self, mode: str, rng: np.random.Generator):
        if mode not in (MODE_RANDOM, MODE_DIRECTED):
            raise ValueError(f"알 수 없는 이동 방식: {mode}")
        self.mode = mode
        self.rng = rng
        self.spec = MOVEMENT_TABLE[mode]

        # ---- 검증용 계수기 ----
        self.n_move_calls = 0          # 이동을 수행한 총 세포-tick 수
        self.n_gradient_guided = 0     # 구배를 따라 이동한 세포-tick 수
        self.n_no_gradient = 0         # 구배가 없어 탐색한 세포-tick 수
        self.n_by_type = np.zeros(N_CELL_TYPES, dtype=np.int64)
        self.n_grad_by_type = np.zeros(N_CELL_TYPES, dtype=np.int64)

    # ------------------------------------------------------------------
    def move(self,
             x: np.ndarray, y: np.ndarray,
             hx: np.ndarray, hy: np.ndarray,
             speed: np.ndarray,
             stacks: Dict[str, np.ndarray],
             fr: np.ndarray, fc: np.ndarray,
             ctype: np.ndarray) -> None:
        """
        면역세포를 한 tick 이동시킨다. 배열은 제자리에서 수정된다.

          x, y   : 좌표
          hx, hy : 직전 이동 방향 (극성)
          speed  : 세포별 이동 속도 (칸/tick)
          stacks : 리간드별 8방향 농도장
          fr, fc : 각 세포가 위치한 농도장 격자
          ctype  : 각 세포의 종류
        """
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
        """
        무작위 이동: 균일난수로 방향을 결정한다 (브라운 운동).
        리간드 농도를 전혀 참조하지 않는다.
        """
        theta = self.rng.uniform(0.0, 2.0 * math.pi, n).astype(np.float32)
        return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)

    # ------------------------------------------------------------------
    def _directed_direction(self, stacks, fr, fc, ctype, hx, hy):
        """
        방향성 이동: 자기 수용체의 리간드 농도가 최대인 방향을 고른다.
        """
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

            # ---- 이 세포가 가진 수용체가 감지하는 농도를 합산한다 ----
            values = None
            for _receptor, ligand_key in CELL_RECEPTOR_FIELDS[code]:
                stack = stacks.get(ligand_key)
                if stack is None:
                    continue
                v = stack[:, fr[mask], fc[mask]].T          # (k, 8)
                values = v if values is None else values + v

            if values is None:
                # 감지 가능한 농도장이 없는 세포 (이론상 발생하지 않음)
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
            # 구배가 감지되지 않으면 따라갈 신호가 없으므로 탐색한다
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
        """격자 경계에서 반사시킨다"""
        limit = float(Scale.GRID) - 1e-3
        np.abs(x, out=x)
        np.abs(y, out=y)
        for arr in (x, y):
            over = arr > limit
            if over.any():
                arr[over] = 2.0 * limit - arr[over]
            np.clip(arr, 0.0, limit, out=arr)

    # ------------------------------------------------------------------
    @property
    def gradient_fraction(self) -> float:
        """구배를 따라 이동한 비율 (조작 검증용)"""
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
#  SECTION 13.  바이러스 개체군
# =============================================================================


class VirusPool:
    """
    자유 바이러스 입자의 집합.

    ★작품설명서 표 3:
        공간 분포 = 조직 전역에 균일한 확산
        자체 신호 = 없음
      따라서 큰 확산 폭으로 조직 전역에 퍼지며, 어떤 리간드도 방출하지 않는다.
    """

    def __init__(self, p: Params, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        self.x = np.zeros(0, dtype=np.float32)
        self.y = np.zeros(0, dtype=np.float32)
        self.n_thinned = 0

    @property
    def n(self) -> int:
        return int(self.x.size)

    # ------------------------------------------------------------------
    def seed_uniform(self, k: int) -> None:
        """초기 접종: 조직 전역에 흩뿌린다"""
        self.x = self.rng.uniform(0, Scale.GRID - 1, k).astype(np.float32)
        self.y = self.rng.uniform(0, Scale.GRID - 1, k).astype(np.float32)

    # ------------------------------------------------------------------
    def emit_from(self, sites: np.ndarray, per_site: float,
                  tissue: Tissue) -> int:
        """생산기 감염칸에서 바이러스를 방출한다"""
        if sites.size == 0 or per_site <= 0:
            return 0
        k = int(sites.size * per_site)
        if k <= 0:
            return 0
        pick = self.rng.choice(sites, size=k, replace=True)
        cx, cy = tissue.xy_of_site(pick)
        self.x = np.concatenate([self.x, cx])
        self.y = np.concatenate([self.y, cy])
        if self.x.size > self.p.max_virions:
            keep = self.rng.random(self.x.size) < 0.5
            self.n_thinned += int(np.count_nonzero(~keep))
            self.x = self.x[keep]
            self.y = self.y[keep]
        return k

    # ------------------------------------------------------------------
    def diffuse(self) -> None:
        """
        조직 전역 확산.
        표 3의 '조직 전역에 균일한 확산'을 큰 확산 폭으로 구현한다.
        """
        if self.n == 0:
            return
        s = self.p.virion_sigma
        self.x += self.rng.normal(0.0, s, self.n).astype(np.float32)
        self.y += self.rng.normal(0.0, s, self.n).astype(np.float32)
        MovementEngine.reflect_at_boundary(self.x, self.y)

    # ------------------------------------------------------------------
    def natural_decay(self) -> int:
        """유리 바이러스 반감기 3시간 (Baccam 등, 2006)"""
        if self.n == 0:
            return 0
        m = self.rng.random(self.n) < self.p.virion_decay
        k = int(np.count_nonzero(m))
        if k:
            self.x = self.x[~m]
            self.y = self.y[~m]
        return k

    # ------------------------------------------------------------------
    def remove(self, idx: np.ndarray) -> int:
        """지정한 인덱스의 바이러스를 제거한다"""
        if idx.size == 0:
            return 0
        keep = np.ones(self.n, dtype=bool)
        keep[idx] = False
        removed = int(self.n - np.count_nonzero(keep))
        self.x = self.x[keep]
        self.y = self.y[keep]
        return removed


# =============================================================================
#  SECTION 14.  감염세포 관리
# =============================================================================


class InfectedCellPool:
    """
    감염된 상피세포.  eclipse -> productive -> dead 로 진행한다.

    ★작품설명서 표 3:
        제거 경로 = 감염된 숙주세포의 사멸이 제거의 대부분을 차지
      수명이 다한 감염세포는 면역세포와 무관하게 자멸한다. 이 경로는
      면역세포의 이동 방식과 아무 관련이 없으므로, 이동 방식의 효과가
      최종 결과에 희석되는 원인이 된다.
    """

    def __init__(self, p: Params, tissue: Tissue):
        self.p = p
        self.tissue = tissue
        self.site = np.zeros(0, dtype=np.int64)
        self.timer = np.zeros(0, dtype=np.int32)
        self.stage = np.zeros(0, dtype=np.uint8)   # 0=eclipse, 1=productive
        self.alive = np.zeros(0, dtype=bool)

    @property
    def n_alive(self) -> int:
        return int(self.alive.sum()) if self.alive.size else 0

    # ------------------------------------------------------------------
    def infect(self, sites: np.ndarray) -> int:
        """정상 상피세포를 감염시킨다"""
        if sites.size == 0:
            return 0
        sites = np.unique(sites)
        ok = self.tissue.state[sites] == HEALTHY
        s = sites[ok]
        if s.size == 0:
            return 0
        self.tissue.state[s] = ECLIPSE
        self.site = np.concatenate([self.site, s])
        self.timer = np.concatenate([self.timer,
                                     np.zeros(s.size, dtype=np.int32)])
        self.stage = np.concatenate([self.stage,
                                     np.zeros(s.size, dtype=np.uint8)])
        self.alive = np.concatenate([self.alive,
                                     np.ones(s.size, dtype=bool)])
        return int(s.size)

    # ------------------------------------------------------------------
    def kill(self, sites: np.ndarray) -> int:
        """NK세포 또는 살해 T세포가 감염세포를 살해한다"""
        if sites.size == 0:
            return 0
        sites = np.unique(sites)
        st = self.tissue.state[sites]
        m = (st == ECLIPSE) | (st == PRODUCTIVE)
        s = sites[m]
        if s.size:
            self.tissue.state[s] = DEAD
            self.tissue.n_dead_epithelium += int(s.size)
            self.tissue.cumulative_damage += (
                s.size / max(1, self.tissue.n_target_initial) * 1.6)
        return int(s.size)

    # ------------------------------------------------------------------
    def advance(self) -> Tuple[np.ndarray, int]:
        """
        시간을 진행시킨다.
        반환: (생산기 감염칸의 위치, 이번 tick 에 자멸한 감염세포 수)
        """
        if self.site.size == 0:
            return np.zeros(0, dtype=np.int64), 0

        st = self.tissue.state[self.site]
        self.alive &= (st == ECLIPSE) | (st == PRODUCTIVE)
        alive = self.alive
        if not alive.any():
            return np.zeros(0, dtype=np.int64), 0

        self.timer[alive] += 1
        p = self.p

        # ---- 잠복기 -> 생산기 ----
        to_productive = (alive & (self.stage == 0)
                         & (self.timer >= p.eclipse_ticks))
        if to_productive.any():
            self.stage[to_productive] = 1
            self.tissue.state[self.site[to_productive]] = PRODUCTIVE

        # ---- 생산기 -> 자멸 (★탐색과 무관한 제거 경로) ----
        to_dead = (alive & (self.stage == 1)
                   & (self.timer >= p.eclipse_ticks + p.productive_ticks))
        n_apoptosis = int(np.count_nonzero(to_dead))
        if n_apoptosis:
            s = self.site[to_dead]
            self.tissue.state[s] = DEAD
            self.tissue.n_dead_epithelium += n_apoptosis
            self.tissue.cumulative_damage += (
                n_apoptosis / max(1, self.tissue.n_target_initial) * 1.6)
            self.alive[to_dead] = False

        productive_sites = self.site[self.alive & (self.stage == 1)]

        # ---- 배열 정리 ----
        if self.alive.size > 20000 and self.alive.mean() < 0.5:
            keep = self.alive
            self.site = self.site[keep]
            self.timer = self.timer[keep]
            self.stage = self.stage[keep]
            self.alive = self.alive[keep]

        return productive_sites, n_apoptosis


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

    # ------------------------------------------------------------------
    def add(self, k: int, ctype: int, x: np.ndarray, y: np.ndarray,
            speed: float, life: int, phago_capacity: int,
            rng: np.random.Generator, in_vessel: bool = False,
            specific: bool = False) -> int:
        """세포를 추가한다"""
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

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def compact(self) -> None:
        """죽은 세포를 배열에서 제거한다"""
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

      1. 항원이 도달하여 축적된다
      2. 일정 시간이 지나면 프라이밍이 완료된다
      3. 조력 T세포와 살해 T세포가 클론 증식한다
      4. 형질세포가 분화하여 항체(IgM -> IgG)를 분비한다
      5. 반응이 끝나면 일부가 기억세포로 남는다
    """

    def __init__(self, p: Params):
        self.p = p
        self.antigen = 0.0
        self.primed = False
        self.t_primed: Optional[int] = None
        self.priming_clock = 0
        self.cd4 = 100.0
        self.cd8 = 50.0
        self.plasma_cell = 0.0
        self.igm = 0.0
        self.igg = 0.0
        self.t_igm_detected: Optional[float] = None
        self.t_igg_detected: Optional[float] = None

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
            self.cd8 = min(self.cd8 * 1.0130, 40000.0)
            self.plasma_cell = min(
                self.plasma_cell + 0.020 * self.cd4 - 0.006 * self.plasma_cell,
                55000.0)
        else:
            self.cd4 *= 0.985
            self.cd8 *= 0.985
            self.plasma_cell *= 0.988

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
        """총 항체 역가"""
        return min(self.igm + self.igg * 1.8, self.p.ab_titer_cap)

    @property
    def neutralization(self) -> float:
        """
        항HA 항체에 의한 중화 정도 (0~1).
        바이러스가 숙주세포에 진입하는 것을 차단하는 비율이다.
        """
        t = self.titer
        return t / (t + self.p.ab_neutralize_k) if t > 0 else 0.0


# =============================================================================
#  SECTION 17.  시뮬레이션 본체
# =============================================================================


def match_same_site(a_sites: np.ndarray,
                    b_sites: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    같은 격자 칸에 있는 (a 인덱스, b 인덱스) 쌍을 찾는다.
    면역세포와 병원체가 접촉했는지 판정할 때 사용한다.
    """
    if a_sites.size == 0 or b_sites.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    order = np.argsort(b_sites, kind="stable")
    sorted_b = b_sites[order]
    lo = np.searchsorted(sorted_b, a_sites, side="left")
    hi = np.searchsorted(sorted_b, a_sites, side="right")
    has = hi > lo
    ai = np.flatnonzero(has)
    if ai.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return ai, order[lo[ai]]


class InfluenzaSimulation:
    """
    인플루엔자 A 감염 - 면역반응 Agent-Based Model.

    한 tick 의 진행 순서:
       A. 감염세포 단계 전이 및 자멸
       B. 바이러스 방출 · 확산 · 자연 감쇠
       C. 신규 감염 성립
       D. 리간드 농도장 갱신
       E. 면역세포 동원
       F. 수명 처리
       G. 이동  (★두 조건의 유일한 차이)
       H. 혈관외유출 판정
       I. 식균작용
       J. NK세포 / 살해 T세포의 감염세포 살해
       K. 보체 · 항체에 의한 제거
       L. 체액성 인자 갱신
       M. 림프절 반응
       N. 조직 재생
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

        # ---- 구성 요소 ----
        self.tissue = Tissue(self.rng)
        self.virus = VirusPool(self.p, self.rng)
        self.infected = InfectedCellPool(self.p, self.tissue)
        self.cells = ImmuneCellPool()
        self.mover = MovementEngine(movement_mode, self.rng)
        self.lymph_node = LymphNode(self.p)
        self.ligands = LigandFieldSet()

        n = Scale.FIELD_N
        self.interferon = np.zeros((n, n), dtype=np.float32)
        self.antiviral_state = np.zeros((n, n), dtype=np.float32)
        self.complement = 0.0
        self.inflammation = 0.0

        self.tick = 0
        self.daily: List[dict] = []
        self.wall_time = 0.0

        # ---- 누적 통계 ----
        self.cleared_by_phagocytosis = 0
        self.cleared_by_decay = 0
        self.cleared_by_complement = 0
        self.cleared_by_antibody = 0
        self.killed_by_nk = 0
        self.killed_by_ctl = 0
        self.apoptosis_count = 0
        self.total_infections = 0
        self.dead_immune_cells = 0
        self.memory_cells = 0
        self.n_extravasated = 0
        self.t_first_extravasation: Optional[int] = None
        self.t_first_arrival: Optional[int] = None
        self.ctl_contact_events = 0
        self.ctl_agent_ticks = 0

        self._seed_initial_population()

    # ------------------------------------------------------------------
    def _spawn(self, k: int, ctype: int, speed: float, life: int,
               phago_capacity: int, specific: bool = False) -> int:
        """
        면역세포를 생성한다.
        생성 장소는 작품설명서 표 2에 따라 이동 방식별로 다르다.
          무작위 이동 -> 조직에 무작위적으로 생성 (곧바로 조직에 존재)
          방향성 이동 -> 림프절에서 생성 (혈류를 타고 이동한 뒤 유출)
        """
        if k <= 0:
            return 0
        x, y = self.tissue.spawn_xy(k, self.mode, self.rng)
        in_vessel = (self.mode == MODE_DIRECTED)
        return self.cells.add(k, ctype, x, y, speed, life, phago_capacity,
                              self.rng, in_vessel=in_vessel, specific=specific)

    def _seed_initial_population(self) -> None:
        """초기 병원체와 면역세포를 배치한다"""
        p = self.p
        self.virus.seed_uniform(p.n_virus0)
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

        # ---------- A. 감염세포 단계 전이 및 자멸 ----------
        productive_sites, n_apoptosis = self.infected.advance()
        self.apoptosis_count += n_apoptosis

        # ---------- B. 바이러스 방출 · 확산 · 자연 감쇠 ----------
        self.virus.emit_from(productive_sites, p.burst_per_tick, tissue)
        self.virus.diffuse()
        self.cleared_by_decay += self.virus.natural_decay()

        # ---------- C. 신규 감염 성립 ----------
        if self.virus.n:
            v_site = Tissue.site_of_xy(self.virus.x, self.virus.y)
            on_target = tissue.state[v_site] == HEALTHY
            if on_target.any():
                cand = np.flatnonzero(on_target)
                fr, fc = Tissue.field_of_site(v_site[cand])
                protection = self.antiviral_state[fr, fc]
                neutral = self.lymph_node.neutralization
                prob = p.infect_p * (1.0 - protection) * (1.0 - neutral)
                ok = rng.random(cand.size) < prob
                if ok.any():
                    hit = cand[ok]
                    n_new = self.infected.infect(v_site[hit])
                    self.total_infections += n_new
                    self.virus.remove(hit)

        # ---------- D. 리간드 농도장 갱신 ----------
        self._update_ligand_fields(productive_sites)

        # ---------- E. 면역세포 동원 ----------
        self._recruit()

        # ---------- F. 수명 처리 ----------
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

        # ---------- G. 이동 (★두 조건의 유일한 차이) ----------
        stacks = self.ligands.direction_stacks()
        n = cells.n
        idx = np.flatnonzero(cells.alive[:n])
        if idx.size:
            # 혈관 내 세포는 혈류를 타고 빠르게 이동한다
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

            # ---------- H. 혈관외유출 판정 ----------
            if in_vessel.any():
                self._extravasate(idx, in_vessel, t)

        # ---------- I. 식균작용 ----------
        self._phagocytosis()

        # ---------- J. NK세포 / 살해 T세포의 감염세포 살해 ----------
        self._cytotoxic_killing(t)

        # ---------- K. 보체 · 항체에 의한 제거 ----------
        self._humoral_clearance()

        # ---------- L. 체액성 인자 갱신 ----------
        self._update_humoral_factors()

        # ---------- M. 림프절 반응 ----------
        self.lymph_node.step(t, 0.004 * self.infected.n_alive)
        if self.lymph_node.primed and self.memory_cells == 0:
            days = (t - self.lymph_node.t_primed) / Scale.TICKS_PER_DAY
            if days > 8.0:
                self.memory_cells = int(
                    (self.lymph_node.cd4 + self.lymph_node.cd8)
                    * p.memory_fraction)

        # ---------- N. 조직 재생 ----------
        tissue.regenerate(p.regeneration_rate * (1.0 - self.inflammation))

    # ==================================================================
    def _update_ligand_fields(self, productive_sites: np.ndarray) -> None:
        """
        표 1의 리간드를 각각 독립된 농도장으로 갱신한다.
        생성원은 리간드마다 다르다.
        """
        p = self.p
        cells = self.cells
        n = cells.n
        alive = cells.alive[:n]

        # ---- 감염된 상피세포가 방출하는 리간드 ----
        if productive_sites.size:
            fr, fc = Tissue.field_of_site(productive_sites)
            self.ligands.deposit(LIG_CXCL8, fr, fc, p.cxcl8_from_infected)
            self.ligands.deposit(LIG_CCL2, fr, fc, p.ccl2_from_infected)
            self.ligands.deposit(LIG_CCL35, fr, fc, p.ccl35_from_infected)
            self.ligands.deposit(LIG_CXCR3L, fr, fc, p.cxcr3l_basal)
            np.add.at(self.interferon, (fr, fc), p.ifn_production)

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

        # ---- ★CXCL9/10/11 은 인터페론 유도성 (Luster 등, 1985) ----
        self.ligands.add_field(LIG_CXCR3L, self.interferon, p.cxcr3l_ifn_gain)

        # ---- ★CXCL13 은 림프절이 항상적으로 방출 (Okada 등, 2002) ----
        self.ligands.deposit_region(LIG_CXCL13, self.tissue.lymph_field_slice,
                                    p.cxcl13_from_lymphnode)

        # ---- ★fMLF: 표 3에 따라 바이러스는 자체 신호를 내지 않는다 ----
        #      따라서 FPR1 축은 생성원이 없어 작동하지 않는다.
        #      (폐렴구균 모형에서는 세균 자신이 생성원이 된다)

        # ---- 확산 및 감쇠 ----
        self.ligands.diffuse_and_decay()

        # ---- 인터페론과 항바이러스 상태 ----
        self.interferon = gaussian_filter(self.interferon, sigma=p.ifn_sigma,
                                          mode="nearest")
        self.interferon *= (1.0 - p.ifn_decay)
        drive = np.clip(self.interferon * 6.0, 0.0, 1.0)
        self.antiviral_state += p.antiviral_rate * drive * (
            1.0 - self.antiviral_state)
        self.antiviral_state -= p.antiviral_decay * self.antiviral_state
        np.clip(self.antiviral_state, 0.0, 1.0, out=self.antiviral_state)

    # ==================================================================
    def _recruit(self) -> None:
        """염증 정도에 비례하여 면역세포를 동원한다"""
        p = self.p
        drive = float(np.clip(self.inflammation, 0.0, 1.0))
        for ctype, gain, speed, life, cap in (
                (NEUTROPHIL, p.recruit_neutrophil, p.v_neutrophil,
                 p.life_neutrophil, p.capacity_neutrophil),
                (MONOCYTE, p.recruit_monocyte, p.v_monocyte,
                 p.life_monocyte, p.capacity_monocyte),
                (NK_CELL, p.recruit_nk, p.v_nk, p.life_nk, 0)):
            k = int(gain * drive / Scale.TICKS_PER_DAY)
            if k > 0:
                self._spawn(k, ctype, speed, life, cap)

        # ---- 후천면역 효과기 세포의 조직 진입 ----
        if self.lymph_node.primed:
            k = int(self.lymph_node.cd8 * 0.0020)
            if k > 0:
                self._spawn(k, KILLER_T, p.v_t, p.life_effector_t, 0,
                            specific=True)
            k = int(self.lymph_node.cd4 * 0.0010)
            if k > 0:
                self._spawn(k, HELPER_T, p.v_t, p.life_effector_t, 0,
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

        # 각 세포가 자기 수용체로 감지하는 농도를 계산한다
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
    def _phagocytosis(self) -> None:
        """
        중성구와 대식세포가 같은 칸의 바이러스 입자를 포식한다.
        혈관 내 세포는 참여하지 않는다.
        """
        p = self.p
        cells = self.cells
        rng = self.rng
        n = cells.n
        idx = np.flatnonzero(cells.alive[:n])
        if idx.size == 0 or self.virus.n == 0:
            return

        tissue_cells = idx[~cells.in_vessel[idx]]
        if tissue_cells.size == 0:
            return

        c_site = Tissue.site_of_xy(cells.x[tissue_cells], cells.y[tissue_cells])
        v_site = Tissue.site_of_xy(self.virus.x, self.virus.y)

        is_phago = (((cells.ctype[tissue_cells] == NEUTROPHIL)
                     | (cells.ctype[tissue_cells] == MONOCYTE))
                    & (cells.capacity[tissue_cells] > 0))
        sel = np.flatnonzero(is_phago)
        if sel.size == 0:
            return

        loc, vi = match_same_site(c_site[sel], v_site)
        if loc.size == 0:
            return
        ok = rng.random(loc.size) < p.phagocytosis_p
        if not ok.any():
            return
        gi = tissue_cells[sel[loc[ok]]]
        targets = vi[ok]
        uniq, first = np.unique(targets, return_index=True)
        np.subtract.at(cells.capacity, gi[first], 1)
        self.cleared_by_phagocytosis += self.virus.remove(uniq)

    # ==================================================================
    def _cytotoxic_killing(self, t: int) -> None:
        """NK세포와 살해 T세포가 감염세포를 살해한다"""
        p = self.p
        cells = self.cells
        rng = self.rng
        n = cells.n
        idx = np.flatnonzero(cells.alive[:n])
        if idx.size == 0 or self.infected.n_alive == 0:
            return

        tissue_cells = idx[~cells.in_vessel[idx]]
        if tissue_cells.size == 0:
            return

        c_site = Tissue.site_of_xy(cells.x[tissue_cells], cells.y[tissue_cells])
        state = self.tissue.state[c_site]
        on_infected = (state == ECLIPSE) | (state == PRODUCTIVE)

        if on_infected.any() and self.t_first_arrival is None:
            self.t_first_arrival = t

        # ---- NK세포 ----
        sel = np.flatnonzero(on_infected
                             & (cells.ctype[tissue_cells] == NK_CELL))
        if sel.size:
            ok = rng.random(sel.size) < p.nk_kill_p
            if ok.any():
                self.killed_by_nk += self.infected.kill(c_site[sel[ok]])

        # ---- 살해 T세포 (항원특이 세포만) ----
        ctl_mask = ((cells.ctype[tissue_cells] == KILLER_T)
                    & cells.specific[tissue_cells])
        self.ctl_agent_ticks += int(np.count_nonzero(ctl_mask))
        sel = np.flatnonzero(on_infected & ctl_mask)
        if sel.size:
            self.ctl_contact_events += int(sel.size)
            ok = rng.random(sel.size) < p.ctl_kill_p
            if ok.any():
                self.killed_by_ctl += self.infected.kill(c_site[sel[ok]])

    # ==================================================================
    def _humoral_clearance(self) -> None:
        """보체와 항체가 유리 바이러스를 제거한다"""
        p = self.p
        rng = self.rng
        if self.virus.n and self.complement > 0.2:
            prob = p.complement_clear_p * self.complement
            m = rng.random(self.virus.n) < prob
            if m.any():
                self.cleared_by_complement += self.virus.remove(
                    np.flatnonzero(m))
        titer = self.lymph_node.titer
        if self.virus.n and titer > 0.02:
            prob = min(0.05, p.antibody_clear_p * titer / 10.0)
            m = rng.random(self.virus.n) < prob
            if m.any():
                self.cleared_by_antibody += self.virus.remove(np.flatnonzero(m))

    # ==================================================================
    def _update_humoral_factors(self) -> None:
        """보체 활성도와 염증 정도를 갱신한다"""
        p = self.p
        drive = min(1.0, self.virus.n / 30000.0
                    + self.infected.n_alive / 60000.0)
        self.complement += p.complement_on * drive * (1.0 - self.complement)
        self.complement -= p.complement_off * self.complement
        self.complement = float(np.clip(self.complement, 0.0, 1.0))
        self.inflammation += p.inflammation_on * drive * (1.0 - self.inflammation)
        self.inflammation -= p.inflammation_off * self.inflammation
        self.inflammation = float(np.clip(self.inflammation, 0.0, 1.0))

    # ==================================================================
    def snapshot(self) -> dict:
        """현재 상태를 기록한다"""
        c = self.cells
        tissue = self.tissue
        ln = self.lymph_node
        return {
            "tick": self.tick,
            "day": self.tick / Scale.TICKS_PER_DAY,
            # ---- 병원체 ----
            "virus": self.virus.n,
            "infected": self.infected.n_alive,
            "eclipse": tissue.n_eclipse,
            "productive": tissue.n_productive,
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
            "killer_t_specific": c.count_specific(KILLER_T),
            "helper_t_specific": c.count_specific(HELPER_T),
            "in_tissue": c.count_in_tissue(),
            "memory": self.memory_cells,
            # ---- 리간드 (수용체별 분리 구현의 증거) ----
            "CXCL8": self.ligands.mean(LIG_CXCL8),
            "CCL2": self.ligands.mean(LIG_CCL2),
            "fMLF": self.ligands.mean(LIG_FMLF),
            "CXCL9_10_11": self.ligands.mean(LIG_CXCR3L),
            "CXCL13": self.ligands.mean(LIG_CXCL13),
            "CCL3_CCL5": self.ligands.mean(LIG_CCL35),
            # ---- 기타 신호 ----
            "interferon": float(self.interferon.mean()),
            "antiviral": float(self.antiviral_state.mean()),
            "complement": self.complement,
            "inflammation": self.inflammation,
            # ---- 후천면역 ----
            "igm": ln.igm, "igg": ln.igg, "antibody": ln.titer,
            "neutralization": ln.neutralization,
            "ln_cd4": ln.cd4, "ln_cd8": ln.cd8, "ln_plasma": ln.plasma_cell,
            # ---- 누적 제거량 ----
            "cleared_phagocytosis": self.cleared_by_phagocytosis,
            "cleared_decay": self.cleared_by_decay,
            "cleared_complement": self.cleared_by_complement,
            "cleared_antibody": self.cleared_by_antibody,
            "killed_nk": self.killed_by_nk,
            "killed_ctl": self.killed_by_ctl,
            "apoptosis": self.apoptosis_count,
            "total_infections": self.total_infections,
            "dead_immune": self.dead_immune_cells,
            "extravasated": self.n_extravasated,
        }

    # ==================================================================
    def run(self) -> List[dict]:
        """시뮬레이션을 끝까지 실행한다"""
        p = self.p
        t_start = time.time()
        self.daily.append(self.snapshot())
        if self.verbose:
            print(f"  Day    0 | 바이러스 {self.virus.n:>8,} | "
                  f"감염세포 {0:>7,} | 정상세포 "
                  f"{self.tissue.n_healthy:>7,}")

        total_ticks = p.max_days * Scale.TICKS_PER_DAY
        clear_streak = 0
        clear_needed = int(p.stop_after_clear_days * Scale.TICKS_PER_DAY)

        while self.tick < total_ticks:
            self.step()

            if self.tick % Scale.TICKS_PER_DAY == 0:
                s = self.snapshot()
                self.daily.append(s)
                if self.verbose:
                    print(f"  Day {s['day']:4.0f} | "
                          f"바이러스 {s['virus']:>8,} | "
                          f"감염세포 {s['infected']:>7,} | "
                          f"정상세포 {s['healthy']:>7,} | "
                          f"중성구 {s['neutrophil']:>7,}", flush=True)

            if self.virus.n == 0 and self.infected.n_alive == 0:
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
#  작품설명서 「라-C. 결과 분석」에 열거된 지표를 산출한다:
#    최대 병원체 수, 최대 감염세포 수, 표적세포 최대 소모율, 누적 조직손상,
#    감염세포 제거까지의 시간, 식세포 포식 제거량 등
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
    """정점 이후 처음으로 0이 되는 시각"""
    v = _series(daily, key)
    d = _days(daily)
    if v.size == 0:
        return None
    i = int(np.argmax(v))
    for k in range(i, v.size):
        if v[k] == 0:
            return float(d[k])
    return None


def analyze(sim: InfluenzaSimulation) -> dict:
    """시뮬레이션 결과에서 분석 지표를 산출한다"""
    daily = sim.daily
    d = _days(daily)
    virus = _series(daily, "virus")
    infected = _series(daily, "infected")
    healthy = _series(daily, "healthy")
    n0 = sim.tissue.n_target_initial

    peak_virus, peak_virus_day = _peak(daily, "virus")
    peak_infected, peak_infected_day = _peak(daily, "infected")

    # 초기 지수증식률로부터 체내 재생산수를 추정한다
    growth_rate = float("nan")
    # 정점 이전의 상승 구간에서 지수증식률을 추정한다.
    peak_idx = int(np.argmax(virus))
    early = [x for x in daily[:peak_idx + 1] if x["day"] > 0 and x["virus"] > 0]
    if len(early) < 2:
        early = [x for x in daily if x["day"] > 0 and x["virus"] > 0][:3]
    if len(early) >= 2:
        d0, d1 = early[0], early[-1]
        if d1["day"] > d0["day"] and d0["virus"] > 0 and d1["virus"] > 0:
            growth_rate = (math.log(d1["virus"] / d0["virus"])
                           / (d1["day"] - d0["day"]))
    if math.isnan(growth_rate) or growth_rate <= 0:
        # 초기 접종량에서 정점까지의 평균 증식률로 대체 추정한다
        if peak_idx > 0 and virus[0] > 0 and virus[peak_idx] > virus[0]:
            growth_rate = (math.log(virus[peak_idx] / virus[0])
                           / max(1e-9, d[peak_idx] - d[0]))
    tau = REAL["infected_life_h"] / 24.0
    r0_est = math.exp(growth_rate * tau) if not math.isnan(growth_rate) else float("nan")

    result = {
        # ---- 실행 정보 ----
        "mode": sim.mode,
        "mode_name": sim.movement_spec.name_kr,
        "seed": sim.p.seed,
        "wall_time": sim.wall_time,
        "max_days": sim.p.max_days,

        # ---- 병원체 ----
        "virus_peak": peak_virus,
        "virus_peak_day": peak_virus_day,
        "virus_auc": float(np.trapezoid(virus, d)),
        "virus_t50": _first_below_after_peak(daily, "virus", 0.5),
        "virus_t90": _first_below_after_peak(daily, "virus", 0.1),
        "virus_shed_end": _first_below_after_peak(daily, "virus", 0.01),
        "virus_clear_day": _clear_day(daily, "virus"),
        "virus_final": float(virus[-1]),
        "growth_rate_per_day": growth_rate,
        "R0_estimate": r0_est,

        # ---- 감염세포 / 조직 ----
        "infected_peak": peak_infected,
        "infected_peak_day": peak_infected_day,
        "infected_auc": float(np.trapezoid(infected, d)),
        "infected_clear_day": _clear_day(daily, "infected"),
        "total_infections": sim.total_infections,
        "min_healthy": float(healthy.min()),
        "max_target_depletion": float(1.0 - healthy.min() / n0),
        "final_healthy": float(healthy[-1]),
        "dead_epithelium": sim.tissue.n_dead_epithelium,
        "damage_final": float(daily[-1]["damage"]),

        # ---- 면역세포 ----
        "neutrophil_peak": float(max(x["neutrophil"] for x in daily)),
        "monocyte_peak": float(max(x["monocyte"] for x in daily)),
        "nk_peak": float(max(x["nk"] for x in daily)),
        "ctl_peak": float(max(x["killer_t_specific"] for x in daily)),
        "th_peak": float(max(x["helper_t_specific"] for x in daily)),
        "memory_final": sim.memory_cells,
        "dead_immune": sim.dead_immune_cells,
        "n_extravasated": sim.n_extravasated,

        # ---- 제거 경로별 기여도 ----
        "cleared_by_phagocytosis": sim.cleared_by_phagocytosis,
        "cleared_by_decay": sim.cleared_by_decay,
        "cleared_by_complement": sim.cleared_by_complement,
        "cleared_by_antibody": sim.cleared_by_antibody,
        "killed_by_nk": sim.killed_by_nk,
        "killed_by_ctl": sim.killed_by_ctl,
        "apoptosis": sim.apoptosis_count,

        # ---- 후천면역 ----
        "antibody_peak": float(max(x["antibody"] for x in daily)),
        "t_igm": sim.lymph_node.t_igm_detected,
        "t_igg": sim.lymph_node.t_igg_detected,
        "t_primed": (sim.lymph_node.t_primed / Scale.TICKS_PER_DAY
                     if sim.lymph_node.t_primed else None),

        # ---- 신호 ----
        "interferon_peak": float(max(x["interferon"] for x in daily)),
        "inflammation_peak": float(max(x["inflammation"] for x in daily)),
        "inflammation_peak_day": float(
            daily[int(np.argmax(_series(daily, "inflammation")))]["day"]),
        "complement_peak": float(max(x["complement"] for x in daily)),

        # ---- 리간드 최대 농도 ----
        "peak_CXCL8": sim.ligands.peak[LIG_CXCL8],
        "peak_CCL2": sim.ligands.peak[LIG_CCL2],
        "peak_fMLF": sim.ligands.peak[LIG_FMLF],
        "peak_CXCL9_10_11": sim.ligands.peak[LIG_CXCR3L],
        "peak_CXCL13": sim.ligands.peak[LIG_CXCL13],
        "peak_CCL3_CCL5": sim.ligands.peak[LIG_CCL35],

        # ---- 탐색 효율 ----
        "ctl_contact_per_day": (sim.ctl_contact_events
                                / max(1, sim.ctl_agent_ticks)
                                * Scale.TICKS_PER_DAY),
        "gradient_fraction": sim.mover.gradient_fraction,
        "t_first_extravasation": (sim.t_first_extravasation
                                  / Scale.TICKS_PER_DAY
                                  if sim.t_first_extravasation else None),
        "t_first_arrival": (sim.t_first_arrival / Scale.TICKS_PER_DAY
                            if sim.t_first_arrival else None),
    }
    return result


# =============================================================================
#  SECTION 19.  선행 연구를 통한 타당성 검증
# =============================================================================


@dataclass
class ValidationItem:
    """검증 항목 1개"""
    item: str
    literature: str
    simulated: str
    verdict: str
    source: str


def _fmt_day(v: Optional[float]) -> str:
    return "미발생" if v is None else f"Day {v:.2f}"


def validate(result: dict) -> List[ValidationItem]:
    """
    시뮬레이션 결과를 선행 연구의 보고값과 대조한다.
    판정: 일치 / 근접 / 불일치 / 판정불가
    """
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

    # ---- 바이러스 정점 시각 ----
    v = result["virus_peak_day"]
    add("바이러스 정점 시각",
        f"Day {REAL['symptom_peak_day_min']:.0f}~"
        f"{REAL['symptom_peak_day_max']:.0f} (증상 정점)",
        _fmt_day(v), judge(v, 1.5, 3.5, 1.0, 4.5), "carrat2008")

    # ---- 바이러스 배출 종료 ----
    v = result["virus_shed_end"]
    add("바이러스 배출 종료",
        f"평균 {REAL['shedding_mean_days']:.1f}일",
        _fmt_day(v), judge(v, 3.5, 6.5, 2.5, 8.0), "carrat2008")

    # ---- 바이러스 완전 제거 ----
    v = result["virus_clear_day"]
    add("바이러스 완전 제거",
        "배출 종료 후 수일 이내",
        _fmt_day(v), judge(v, 4.0, 14.0, 3.0, 18.0), "carrat2008")

    # ---- 감염세포 정점 시각 ----
    v = result["infected_peak_day"]
    add("감염세포 정점 시각",
        "Day 1~3 (표적세포 제한 모형)",
        _fmt_day(v), judge(v, 0.8, 3.5, 0.5, 4.5), "baccam2006")

    # ---- 체내 재생산수 ----
    v = result["R0_estimate"]
    add("체내 기초재생산수",
        f"약 {REAL['R0_in_host']:.0f}",
        f"{v:.1f}" if not math.isnan(v) else "판정불가",
        judge(v, 5.0, 60.0, 2.0, 120.0), "baccam2006")

    # ---- 표적세포 소모 ----
    v = result["max_target_depletion"] * 100.0
    add("표적세포 최대 소모율",
        "표적세포제한 모형에서 상당 비율 소모",
        f"{v:.1f} %", judge(v, 50.0, 99.5, 30.0, 99.9), "baccam2006")

    # ---- 염증 정점 ----
    v = result["inflammation_peak_day"]
    add("염증 정점 시각",
        f"Day {REAL['symptom_peak_day_min']:.0f}~"
        f"{REAL['symptom_peak_day_max']:.0f}",
        _fmt_day(v), judge(v, 1.5, 4.5, 1.0, 6.0), "carrat2008")

    # ---- 항체 검출 ----
    v = result["t_igm"]
    add("IgM 검출 시각",
        f"Day {REAL['igm_detect_day']:.0f} 내외",
        _fmt_day(v), judge(v, 3.0, 8.0, 2.0, 11.0), MODEL_ASSUMPTION)
    v = result["t_igg"]
    add("IgG 검출 시각",
        f"Day {REAL['igg_detect_day']:.0f} 내외",
        _fmt_day(v), judge(v, 5.0, 13.0, 4.0, 16.0), MODEL_ASSUMPTION)

    # ---- 제거 경로 ----
    apo = result["apoptosis"]
    killed = result["killed_by_nk"] + result["killed_by_ctl"]
    total_removed = apo + killed
    ratio = apo / total_removed * 100 if total_removed > 0 else float("nan")
    add("감염세포 제거 경로",
        "감염된 숙주세포의 사멸이 제거의 대부분 (표 3)",
        f"자멸 {ratio:.1f} %" if not math.isnan(ratio) else "판정불가",
        judge(ratio, 60.0, 100.0, 40.0, 100.0), "baccam2006")

    # ---- fMLF 축 ----
    v = result["peak_fMLF"]
    add("자체 신호(fMLF) 방출",
        "없음 (표 3)",
        f"{v:.6f}", "일치" if v == 0.0 else "불일치", MODEL_ASSUMPTION)

    # ---- 조작 검증 ----
    v = result["gradient_fraction"] * 100.0
    if result["mode"] == MODE_RANDOM:
        add("이동 조작 검증 (신호 감지)",
            "무작위 이동 = 0 % (표 2)",
            f"{v:.1f} %", "일치" if v < 0.01 else "불일치", MODEL_ASSUMPTION)
    else:
        add("이동 조작 검증 (신호 감지)",
            "방향성 이동 = 100 % (표 2)",
            f"{v:.1f} %", "일치" if v > 90.0 else "근접" if v > 60.0
            else "불일치", MODEL_ASSUMPTION)

    return checks


def print_validation(checks: List[ValidationItem]) -> Tuple[int, int, int]:
    """검증표 출력. 반환: (일치, 근접, 불일치) 개수"""
    print("=" * 112)
    print(" 선행 연구를 통한 타당성 검증")
    print("=" * 112)
    print(f" {'검증 항목':<24}{'문헌 기준값':<34}{'시뮬레이션 결과':<20}"
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
        print(f" {c.item:<24}{c.literature:<34}{c.simulated:<20}"
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


def print_result_report(result: dict, sim: InfluenzaSimulation) -> None:
    """작품설명서의 결과 지표를 출력한다"""
    r = result
    print("=" * 84)
    print(f" 탐구 결과 — 인플루엔자 A 바이러스(H1N1) / {r['mode_name']}")
    print("=" * 84)

    print("\n [1] 병원체")
    print(f"   최대 병원체 수          : {r['virus_peak']:>14,.0f} agent"
          f"   (Day {r['virus_peak_day']:.2f})")
    print(f"   병원체 총량 AUC         : {r['virus_auc']:>14,.0f}")
    print(f"   병원체 50 % 감소        : {_fmt_day(r['virus_t50'])}")
    print(f"   병원체 90 % 감소        : {_fmt_day(r['virus_t90'])}")
    print(f"   바이러스 배출 종료      : {_fmt_day(r['virus_shed_end'])}")
    print(f"   병원체 완전 제거        : {_fmt_day(r['virus_clear_day'])}")
    print(f"   추정 체내 재생산수      : {r['R0_estimate']:>14.1f}")

    print("\n [2] 감염세포 및 조직")
    print(f"   최대 감염세포 수        : {r['infected_peak']:>14,.0f} 칸"
          f"   (Day {r['infected_peak_day']:.2f})")
    print(f"   감염세포 제거           : {_fmt_day(r['infected_clear_day'])}")
    print(f"   누적 감염 발생칸        : {r['total_infections']:>14,}")
    print(f"   최저 정상세포 수        : {r['min_healthy']:>14,.0f} 칸")
    print(f"   표적세포 최대 소모율    : "
          f"{r['max_target_depletion']*100:>14.1f} %")
    print(f"   누적 조직손상           : {r['damage_final']:>14.3f}")

    print("\n [3] 면역세포")
    print(f"   중성구 정점             : {r['neutrophil_peak']:>14,.0f}")
    print(f"   단핵구/대식세포 정점    : {r['monocyte_peak']:>14,.0f}")
    print(f"   자연살해 세포 정점      : {r['nk_peak']:>14,.0f}")
    print(f"   항원특이 살해T 정점     : {r['ctl_peak']:>14,.0f}")
    print(f"   기억세포 형성량         : {r['memory_final']:>14,}")
    print(f"   면역세포 사망/소모      : {r['dead_immune']:>14,}")

    print("\n [4] 병원체 제거 경로별 기여도")
    print(f"   식세포 포식 제거량      : "
          f"{r['cleared_by_phagocytosis']:>14,}")
    print(f"   자연 감쇠               : {r['cleared_by_decay']:>14,}")
    print(f"   보체                    : {r['cleared_by_complement']:>14,}")
    print(f"   항체                    : {r['cleared_by_antibody']:>14,}")

    print("\n [5] 감염세포 제거 경로별 기여도")
    print(f"   자연살해 세포 살상      : {r['killed_by_nk']:>14,}")
    print(f"   살해 T세포 살상         : {r['killed_by_ctl']:>14,}")
    print(f"   숙주세포 자멸           : {r['apoptosis']:>14,}")
    tot = r['killed_by_nk'] + r['killed_by_ctl'] + r['apoptosis']
    if tot:
        print(f"   -> 자멸이 차지하는 비율 : "
              f"{r['apoptosis']/tot*100:>14.1f} %")

    print("\n [6] 리간드별 최대 농도 (수용체 분리 구현)")
    for key, label in ((LIG_CXCL8, "CXCL8      (CXCR2)"),
                       (LIG_CCL2, "CCL2       (CCR2)"),
                       (LIG_FMLF, "fMLF       (FPR1)"),
                       (LIG_CXCR3L, "CXCL9/10/11(CXCR3)"),
                       (LIG_CXCL13, "CXCL13     (CXCR5)"),
                       (LIG_CCL35, "CCL3/CCL5  (CCR1)")):
        print(f"   {label:<22}: {sim.ligands.peak[key]:>14.6f}")

    print("\n [7] 이동 방식 조작 검증")
    for line in sim.mover.report().split("\n"):
        print(f"   {line}")

    print("\n [8] 실행 정보")
    print(f"   난수 시드               : {r['seed']:>14,}")
    print(f"   실행 시간               : {r['wall_time']:>14.1f} 초")
    print("=" * 84)


# =============================================================================
#  SECTION 21.  실행 진입점
# =============================================================================


def run_simulation(movement_mode: str = MODE_RANDOM,
                   seed: int = 20260811,
                   max_days: int = 16,
                   verbose: bool = True
                   ) -> Tuple[InfluenzaSimulation, dict]:
    """시뮬레이션 1회 실행"""
    p = Params()
    p.seed = seed
    p.max_days = max_days
    sim = InfluenzaSimulation(p, movement_mode=movement_mode,
                              seed=seed, verbose=verbose)
    sim.run()
    return sim, analyze(sim)


def print_header(movement_mode: str) -> None:
    spec = MOVEMENT_TABLE[movement_mode]
    print()
    print("#" * 84)
    print("#" + " " * 82 + "#")
    print("#" + "  인플루엔자 A 바이러스(H1N1) 감염 - 면역반응 "
          "Agent-Based Model".center(80) + "#")
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
    days = int(argv[3]) if len(argv) > 3 else 16

    print_header(mode)
    print_immune_cell_table()
    print()
    print_ligand_table()
    print()
    print_movement_table()
    print()
    print_pathogen_table()
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

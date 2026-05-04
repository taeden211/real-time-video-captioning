import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any

from dotenv import load_dotenv

try:
    import openai
except ImportError as exc:
    raise ImportError("pip install openai") from exc


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = os.getenv("OPENAI_SCENE_MODEL", "gpt-5.4-mini")
IMAGE_DIR = os.getenv("IMAGE_DIR", "sample_data")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output2")
MAX_IMAGES = None
CANDIDATES_PER_IMAGE = 2
TEMPERATURE = 0.0
MAX_WORKERS = 6
MAX_RETRIES = 3
RETRY_DELAY = 5
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------

WORKER_LABEL = "근로자"

TIER1_LABELS = {
    "근로자",
    "굴착기",
    "타워크레인",
    "이동식크레인",
    "덤프트럭",
    "지게차",
    "로드롤러",
    "콘크리트펌프카",
    "레미콘",
    "파일드라이버",
    "고소작업차",
    "불도저",
    "천공기",
    "로더",
    "그레이더",
    "페이버",
    "소형트럭",
    "승용차",
    "살수차",
}

TIER2_LABELS = {
    "철근",
    "합판",
    "파이프",
    "H빔",
    "토사",
    "거푸집",
    "목재",
    "자재",
    "개구부",
}

TIER3_LABELS = {
    "난간",
    "안전망",
    "방호울타리",
    "가설울타리",
    "동바리",
    "비계",
    "작업발판",
}

PREFERRED_LABELS = sorted(
    TIER1_LABELS
    | TIER2_LABELS
    | TIER3_LABELS
    | {
        "콘크리트구조물",
        "교각",
        "거더",
        "옹벽",
        "슬래브",
        "기초",
        "경고표지",
        "신호수",
        "사다리",
        "수역",
        "안전모",
        "안전대",
        "가스배관",
    }
)

FORBIDDEN_LABELS = {"장비", "차량", "기계", "구조물", "시설", "물체", "물건"}

CATEGORY_PREDICATES = {
    "functional": {"operating", "loading", "carrying", "working_on", "walking_on"},
    "structural": {"on", "inside", "attached_to", "supported_by", "connected_to"},
    "spatial": {"next_to", "above", "below", "behind", "in_front_of"},
    "safety": {"too_close_to", "approaching", "blocking"},
}

RELATION_CATEGORY_PRIORITY = {
    "spatial": 1,
    "structural": 2,
    "functional": 3,
    "safety": 4,
}

VALID_PREDICATES = sorted({p for preds in CATEGORY_PREDICATES.values() for p in preds})
VALID_CATEGORIES = sorted(CATEGORY_PREDICATES)
VALID_HAZARDS = ["추락", "낙하물", "충돌", "협착", "전도", "감전", "익수"]

EQUIPMENT_LABELS = TIER1_LABELS - {WORKER_LABEL}
WATER_LABEL = "수역"
SLOPE_CONTEXT_LABELS = {"토사", "콘크리트구조물", "슬래브", "비계", "작업발판"}

ID_PREFIXES = {
    "근로자": "worker",
    "비계": "scaffold",
    "동바리": "prop",
    "작업발판": "platform",
    "난간": "guardrail",
    "안전망": "safety_net",
    "방호울타리": "barrier",
    "가설울타리": "temp_fence",
    "굴착기": "excavator",
    "타워크레인": "tower_crane",
    "이동식크레인": "mobile_crane",
    "덤프트럭": "dump_truck",
    "지게차": "forklift",
    "로드롤러": "roller",
    "콘크리트펌프카": "pump_car",
    "레미콘": "mixer_truck",
    "파일드라이버": "pile_driver",
    "고소작업차": "aerial_lift",
    "불도저": "bulldozer",
    "천공기": "drill",
    "로더": "loader",
    "그레이더": "grader",
    "페이버": "paver",
    "소형트럭": "small_truck",
    "승용차": "car",
    "살수차": "sprinkler",
    "철근": "rebar",
    "합판": "plywood",
    "파이프": "pipe",
    "H빔": "hbeam",
    "토사": "soil",
    "거푸집": "formwork",
    "목재": "wood",
    "자재": "material",
    "개구부": "opening",
    "콘크리트구조물": "structure",
    "교각": "pier",
    "거더": "girder",
    "옹벽": "retaining_wall",
    "슬래브": "slab",
    "기초": "foundation",
    "경고표지": "sign",
    "신호수": "signal_worker",
    "사다리": "ladder",
    "수역": "water",
    "안전모": "helmet",
    "안전대": "safety_belt",
    "가스배관": "gas_pipe",
}

SYNONYM_MAP = {
    "백호": "굴착기",
    "포크레인": "굴착기",
    "굴삭기": "굴착기",
    "크레인": "이동식크레인",
    "기중기": "이동식크레인",
    "롤러": "로드롤러",
    "머캐덤롤러": "로드롤러",
    "진동롤러": "로드롤러",
    "믹서트럭": "레미콘",
    "콘크리트믹서트럭": "레미콘",
    "펌프카": "콘크리트펌프카",
    "콘크리트펌프": "콘크리트펌프카",
    "고소차": "고소작업차",
    "스카이차": "고소작업차",
    "표지판": "경고표지",
    "안전표지": "경고표지",
    "임시펜스": "가설울타리",
    "가설펜스": "가설울타리",
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

OBJECT_SYSTEM_PROMPT = f"""당신은 건설현장 안전 장면을 분석하는 시각 언어 주석 모델입니다.

목표는 이미지에 실제로 보이는 객체만 추출하고, 각 객체의 위치와 시각적 근거를 구조화하는 것입니다.

출력 규칙:
- 반드시 JSON 스키마를 따르세요.
- 보이지 않는 객체, 가려진 PPE, 추정 위험요소는 만들지 마세요.
- 애매하면 더 보수적인 라벨과 낮은 confidence를 사용하세요.
- 한국어 문장은 건조하고 사실적으로 작성하세요.
- bbox는 전체 이미지 기준 normalized 좌표 [x1, y1, x2, y2]이며 각 값은 0.0~1.0입니다.

객체 그룹화 정책:
- Tier 1: 근로자와 모든 중장비/차량은 항상 개별 객체입니다. 절대 group으로 묶지 마세요.
- Tier 2: 철근, 합판, 파이프, H빔, 토사, 거푸집, 목재, 자재, 개구부는 같은 종류가 같은 구역에 3개 이상 있으면 묶을 수 있습니다.
- Tier 2 예외: 특정 자재 하나가 근로자나 장비와 직접 상호작용하면 그 자재는 별도 객체로 분리하세요.
- Tier 3: 난간, 안전망, 방호울타리, 가설울타리, 동바리, 비계, 작업발판은 연속/반복 설치물일 때 하나의 group 객체로 묶으세요.
- count는 개별 객체면 1, group이면 보이는 대략적 개수를 넣으세요. 연속 구조물은 2 이상으로 두세요.

라벨 규칙:
- 가능한 한 다음 선호 라벨을 사용하세요: {", ".join(PREFERRED_LABELS)}
- 금지 라벨은 사용하지 마세요: {", ".join(sorted(FORBIDDEN_LABELS))}
- '자재'는 최후의 수단입니다. 가능하면 철근, 합판, 파이프, 목재, 거푸집 등 구체 라벨을 쓰세요.
- PPE를 착용 중인 경우에는 근로자의 attributes.ppe에만 기록하세요. 착용 중인 안전모/안전대는 별도 객체로 만들지 마세요.

중장비 구분:
- 굴착기: 붐-암-버킷 구조가 보입니다.
- 지게차: 전방 포크 두 개가 보입니다.
- 타워크레인: 고정 타워와 수평 지브가 보입니다.
- 이동식크레인: 트럭/크롤러 위에 붐이 장착되어 있습니다.
- 덤프트럭: 적재함이 있는 대형 트럭입니다.
- 로드롤러: 원통형 다짐 드럼이 보입니다.
- 콘크리트펌프카: 접이식 붐이 있는 콘크리트 펌프 차량입니다.
- 레미콘: 회전 드럼이 있는 믹서 트럭입니다.

스캔 전략:
- 이미지를 좌상, 우상, 좌하, 우하 4분할로 나누어 확인한 뒤 결과를 병합하세요.
- 주변부의 작은 근로자와 안전시설을 놓치지 마세요.
"""

OBJECT_USER_PROMPT = """이미지에서 보이는 건설현장 객체를 추출하세요.

작업 순서:
1. 전체 장면을 짧게 묘사합니다.
2. 근로자, 중장비/차량, 구조물, 가설시설, 자재, 위험 맥락에 중요한 객체를 찾습니다.
3. 각 객체에 id, label, count, bbox, location, attributes, confidence, evidence를 부여합니다.
4. 보이지 않거나 추정에 가까운 객체는 제외합니다.
5. 모든 id는 label에 맞는 영어 prefix와 번호를 사용합니다. 예: worker_1, excavator_1, rebar_group_1.
"""

RELATION_SYSTEM_PROMPT = f"""당신은 건설현장 객체 목록을 입력받아 씬그래프 관계와 위험요소를 생성하는 모델입니다.

중요한 제한:
- 새 객체를 만들지 마세요.
- relationships와 hazards는 반드시 입력된 object id만 참조해야 합니다.
- 이미지에서 직접 확인되는 관계만 생성하세요.
- 같은 (sub_id, obj_id) 쌍에는 가장 중요한 관계 하나만 생성하세요.
- 우선순위는 safety > functional > structural > spatial입니다.

관계 카테고리와 predicate:
- functional: operating, loading, carrying, working_on, walking_on
- structural: on, inside, attached_to, supported_by, connected_to
- spatial: next_to, above, below, behind, in_front_of
- safety: too_close_to, approaching, blocking

방향 규칙:
- functional: 행위자 또는 움직이는/들고 있는 객체가 sub_id, 대상이 obj_id입니다.
- structural on/inside: 위나 안에 있는 객체가 sub_id, 받치는/담는 객체가 obj_id입니다.
- attached_to/supported_by/connected_to: 더 작거나 의존적인 객체가 sub_id입니다.
- safety: 위험에 노출된 근로자/장비/객체가 sub_id, 위험원 또는 장애물이 obj_id입니다.
- spatial: 중요 객체를 sub_id로 두고, 중요도가 같으면 좌측 객체를 sub_id로 둡니다.

위험요소:
- 허용 hazard: {", ".join(VALID_HAZARDS)}
- 추락: 근로자가 높은 위치에 있고, 해당 위치의 난간/안전망 부재가 명확하거나 좁고 개방된 작업면에 있을 때만 생성합니다.
- 낙하물: 매달린 하중, 고소 위치 자재, 상부 작업과 하부 객체가 함께 보일 때만 생성합니다.
- 충돌: 작동/이동 중인 장비와 근로자/장비가 매우 가까울 때만 생성합니다.
- 협착: 근로자가 장비와 고정물 사이 또는 끼임 지점에 있을 때만 생성합니다.
- 전도: 사다리, 장비, 적재물이 시각적으로 불안정할 때만 생성합니다.
- 감전: 노출 전선, 전기설비, 송전선 인접 작업이 보일 때만 생성합니다.
- 익수: 근로자가 수역 가장자리에 있고 방호가 부족한 것이 보일 때만 생성합니다.
"""

RELATION_USER_PROMPT = """아래 객체 목록과 이미지를 함께 보고 relationships와 hazards를 생성하세요.

주의:
- objects를 다시 출력하지 마세요. relationships와 hazards만 출력하세요.
- 입력된 object id만 참조하세요. 새 object id나 새 객체를 만들지 마세요.
- 관계와 위험은 시각적 근거가 있을 때만 추가하세요.
- evidence와 reason은 짧은 한국어 사실 문장으로 작성하세요.
- 관계가 없으면 relationships는 []로 둡니다.
- 위험이 없으면 hazards는 []로 둡니다.
"""

RELATION_SYSTEM_PROMPT += """

추가 보수 판정 규칙:
- hazards는 "그럴 수 있음"이 아니라 "현재 이미지에서 명확히 확인되는 위험 상태"만 기록합니다.
- 현장 종류, 객체의 일반적 위험성, 단순 근접, 원거리 차량, 정지 장비, 협업 작업만으로 hazard를 만들지 마세요.
- 근로자의 몸 위치, 장비의 작동/이동 상태, 위험원과의 거리, 방호시설 부재가 동시에 명확하지 않으면 hazards는 []로 둡니다.
- 익수는 근로자가 수역 경계 바로 옆에 있고, 물 쪽으로 떨어질 수 있는 개방 가장자리와 방호 부족이 명확할 때만 생성합니다.
- 추락은 근로자가 실제 낙하 가능한 굴착면/구조물/비계 가장자리에 있고, 고저차와 방호 부족이 명확할 때만 생성합니다.
- 충돌은 근로자 또는 장비가 작동/이동 경로 안에 매우 가까이 있을 때만 생성합니다. 멀리 보이는 차량은 제외합니다.
- 협착은 근로자가 장비와 고정물 사이, 장비와 자재 사이, 또는 명확한 끼임 공간 안에 있을 때만 생성합니다.
- 전도는 기울어진 사다리, 불안정한 적재물, 경사면 위 장비처럼 불안정성이 시각적으로 뚜렷할 때만 생성합니다.
"""


# ---------------------------------------------------------------------------
# Quality-focused prompts used by the pipeline.
# These override the draft prompts above and keep the final saved JSON aligned
# with the legacy output/ scene-graph format.
# ---------------------------------------------------------------------------

OBJECT_SYSTEM_PROMPT = f"""당신은 건설현장 안전 장면을 분석하는 시각 언어 주석 모델입니다.

목표:
- 이미지에 실제로 보이는 객체만 추출합니다.
- 결과는 텍스트와 씬그래프 임베딩 유사도 비교에 사용됩니다.
- 같은 시각 장면은 같은 라벨, 같은 id prefix, 같은 관계 방향으로 표현될 수 있도록 일관성을 최우선으로 합니다.

출력 규칙:
- 반드시 JSON 스키마를 따르세요.
- 보이지 않는 객체, 가려진 PPE, 추정 위험요소는 만들지 마세요.
- 애매한 객체는 제외하거나 더 보수적인 라벨과 낮은 confidence를 사용하세요.
- scene_description은 2~4개의 건조한 한국어 문장으로 작성하세요.
- scene_description에는 장면 종류, 주요 장비/근로자, 작업 행위, 주변 구조/토사/수역 맥락을 포함하세요.
- scene_description에서 위험을 단정하지 마세요. 위험은 hazards에서만, 명확할 때만 기록합니다.
- bbox는 전체 이미지 기준 normalized 좌표 [x1, y1, x2, y2]이며 각 값은 0.0~1.0입니다.

객체 그룹화:
- Tier 1: 근로자와 모든 중장비/차량은 항상 개별 객체입니다. 절대 그룹화하지 마세요.
- Tier 2: 철근, 합판, 파이프, H빔, 토사, 거푸집, 목재, 자재, 개구부는 같은 종류가 같은 구역에 3개 이상 있을 때만 group으로 묶을 수 있습니다.
- Tier 2 예외: 특정 자재 하나가 근로자나 장비와 직접 상호작용하면 별도 객체로 분리합니다.
- Tier 3: 난간, 안전망, 방호울타리, 가설울타리, 동바리, 비계, 작업발판은 연속/반복 설치물일 때 group으로 묶습니다.
- count는 내부 검증용입니다. 개별 객체는 1, group은 보이는 대략적 개수를 넣습니다.

라벨 규칙:
- 가능한 한 다음 선호 라벨을 사용하세요: {", ".join(PREFERRED_LABELS)}
- 금지 라벨은 사용하지 마세요: {", ".join(sorted(FORBIDDEN_LABELS))}
- '자재'는 최후의 수단입니다. 가능하면 철근, 합판, 파이프, 목재, 거푸집, H빔, 토사 등 구체 라벨을 쓰세요.
- PPE 착용은 근로자의 attributes.ppe에만 기록합니다. 착용 중인 안전모/안전대는 별도 객체로 만들지 마세요.

id 규칙:
- id는 반드시 라벨별 표준 prefix를 사용하세요.
- 예: worker_1, excavator_1, tower_crane_1, mobile_crane_1, dump_truck_1, roller_1, mixer_truck_1, barrier_group_1, guardrail_group_1, rebar_group_1.
- 왼쪽에서 오른쪽 순서로 번호를 매기세요.

중장비 구분:
- 굴착기: 붐-암-버킷 구조가 보입니다.
- 지게차: 전방 포크 두 개가 보입니다.
- 타워크레인: 고정 타워와 수평 지브가 보입니다.
- 이동식크레인: 트럭/크롤러 위에 붐이 장착되어 있습니다.
- 덤프트럭: 적재함이 있는 대형 트럭입니다.
- 로드롤러: 원통형 다짐 드럼이 보입니다.
- 콘크리트펌프카: 접이식 붐이 있는 콘크리트 펌프 차량입니다.
- 레미콘: 회전 드럼이 있는 믹서 트럭입니다.

스캔 전략:
- 이미지를 좌상, 우상, 좌하, 우하 4분할로 나누어 확인한 뒤 결과를 병합하세요.
- 작은 근로자, 수역, 굴착면, 난간, 방호울타리처럼 안전 맥락에 중요한 객체를 놓치지 마세요.
"""

OBJECT_USER_PROMPT = """이미지에서 보이는 건설현장 객체를 추출하세요.

작업 순서:
1. 전체 장면을 먼저 요약합니다.
2. 근로자, 중장비/차량, 구조물, 가설시설, 안전시설, 자재, 수역/토사처럼 장면 의미에 중요한 객체를 찾습니다.
3. 각 객체에 id, label, count, bbox, location, attributes, confidence, evidence를 부여합니다.
4. 보이지 않거나 추정에 가까운 객체는 제외합니다.
5. id prefix를 표준화하고, 같은 종류 객체는 왼쪽에서 오른쪽 순서로 번호를 매깁니다.
"""

RELATION_SYSTEM_PROMPT = f"""당신은 건설현장 객체 목록과 이미지를 함께 보고 씬그래프 관계와 위험요소를 생성하는 모델입니다.

중요 제한:
- 새 객체를 만들지 마세요.
- relationships와 hazards는 반드시 입력된 object id만 참조해야 합니다.
- 직접 보이는 관계만 생성하세요.
- 관계는 적지만 의미 있게 생성하세요. 임베딩 비교에 불필요한 spatial 관계를 많이 만들지 마세요.
- 같은 (sub_id, obj_id) 쌍에는 가장 중요한 관계 하나만 생성합니다.
- 같은 의미의 양방향 spatial 관계를 중복 생성하지 마세요.
- 우선순위는 safety > functional > structural > spatial입니다.
- functional 관계가 명확하면 같은 쌍의 spatial 관계는 추가하지 마세요.

관계 카테고리와 predicate:
- functional: operating, loading, carrying, working_on, walking_on
- structural: on, inside, attached_to, supported_by, connected_to
- spatial: next_to, above, below, behind, in_front_of
- safety: too_close_to, approaching, blocking

방향 규칙:
- functional: 행위자 또는 움직이는/들고 있는 객체가 sub_id, 대상이 obj_id입니다.
- structural on/inside: 위나 안에 있는 객체가 sub_id, 받치는/담는 객체가 obj_id입니다.
- attached_to/supported_by/connected_to: 더 작거나 의존적인 객체가 sub_id입니다.
- safety: 위험에 노출된 근로자/장비/객체가 sub_id, 위험원 또는 장애물이 obj_id입니다.
- spatial: 중요 객체를 sub_id로 두고, 중요도가 같으면 왼쪽 객체를 sub_id로 둡니다.

위험요소 생성 원칙:
- 허용 hazard: {", ".join(VALID_HAZARDS)}
- 위험요소는 가능성이나 일반 상식이 아니라 이미지에서 명확히 보이는 위험 상태일 때만 생성합니다.
- 근거가 애매하면 hazards는 []로 둡니다.
- 추락: 근로자가 높은 위치 또는 굴착면 가장자리에 있고, 실제 낙하 방향과 방호 부족이 명확할 때만 생성합니다.
- 낙하물: 매달린 하중, 고소 위치 자재, 상부 작업과 하부 객체가 함께 명확히 보일 때만 생성합니다.
- 충돌: 작동/이동 중인 장비와 근로자/장비가 매우 가까울 때만 생성합니다.
- 협착: 근로자가 장비와 고정물 사이 또는 끼임 지점에 있을 때만 생성합니다.
- 전도: 사다리, 장비, 적재물이 시각적으로 불안정할 때만 생성합니다.
- 감전: 노출 전선, 전기설비, 송전선 인접 작업이 명확할 때만 생성합니다.
- 익수: 근로자가 수역 가장자리 또는 수중에 있고 방호가 부족한 것이 보일 때만 생성합니다.
- 수역만 보이거나 장비가 물가/얕은 물 위에 있는 것만으로는 익수를 생성하지 않습니다.
- 장비 간 협업 작업은 functional 관계입니다. 굴착기와 덤프트럭이 함께 작업한다는 이유만으로 충돌을 생성하지 않습니다.
- 승용차와 장비가 같은 현장에 보이는 것만으로 충돌을 생성하지 않습니다.
- 굴착면이나 토사 사면이 보이는 것만으로 추락/전도를 생성하지 않습니다.
"""

RELATION_USER_PROMPT = """아래 객체 목록과 이미지를 함께 보고 relationships와 hazards를 생성하세요.

주의:
- objects를 다시 출력하지 마세요. relationships와 hazards만 출력하세요.
- 입력된 object id만 참조하세요. 새 object id나 새 객체를 만들지 마세요.
- 관계와 위험은 시각적 근거가 있을 때만 추가하세요.
- evidence와 reason은 짧은 한국어 사실 문장으로 작성하세요.
- 관계가 없으면 relationships는 []로 둡니다.
- 위험이 없으면 hazards는 []로 둡니다.
"""


# ---------------------------------------------------------------------------
# JSON schemas
# ---------------------------------------------------------------------------

OBJECT_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "label": {"type": "string"},
        "count": {"type": "integer", "minimum": 1},
        "bbox": {
            "type": "array",
            "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "minItems": 4,
            "maxItems": 4,
        },
        "location": {"type": "string"},
        "attributes": {
            "type": "object",
            "properties": {
                "ppe": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "state": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["ppe", "state"],
            "additionalProperties": False,
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence": {"type": "string"},
    },
    "required": [
        "id",
        "label",
        "count",
        "bbox",
        "location",
        "attributes",
        "confidence",
        "evidence",
    ],
    "additionalProperties": False,
}

OBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scene_description": {"type": "string"},
        "objects": {
            "type": "array",
            "items": OBJECT_ITEM_SCHEMA,
        },
    },
    "required": ["scene_description", "objects"],
    "additionalProperties": False,
}

RELATION_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sub_id": {"type": "string"},
        "predicate": {"type": "string", "enum": VALID_PREDICATES},
        "obj_id": {"type": "string"},
        "category": {"type": "string", "enum": VALID_CATEGORIES},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence": {"type": "string"},
    },
    "required": ["sub_id", "predicate", "obj_id", "category", "score", "evidence"],
    "additionalProperties": False,
}

HAZARD_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "related_object_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "hazard": {"type": "string", "enum": VALID_HAZARDS},
        "reason": {"type": "string"},
    },
    "required": ["related_object_ids", "hazard", "reason"],
    "additionalProperties": False,
}

SCENE_GRAPH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scene_description": {"type": "string"},
        "objects": {
            "type": "array",
            "items": OBJECT_ITEM_SCHEMA,
        },
        "relationships": {
            "type": "array",
            "items": RELATION_ITEM_SCHEMA,
        },
        "hazards": {
            "type": "array",
            "items": HAZARD_ITEM_SCHEMA,
        },
    },
    "required": ["scene_description", "objects", "relationships", "hazards"],
    "additionalProperties": False,
}

RELATION_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relationships": {
            "type": "array",
            "items": RELATION_ITEM_SCHEMA,
        },
        "hazards": {
            "type": "array",
            "items": HAZARD_ITEM_SCHEMA,
        },
    },
    "required": ["relationships", "hazards"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def image_to_data_uri(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime_map = {
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    mime = mime_map.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def find_images() -> list[str]:
    images = []
    for root, dirs, files in os.walk(IMAGE_DIR):
        dirs.sort()
        for filename in sorted(files):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                images.append(os.path.join(root, filename))
    return images


def output_stem_for_image(img_path: str) -> str:
    rel_path = os.path.relpath(img_path, IMAGE_DIR)
    stem = os.path.splitext(rel_path)[0]
    return stem.replace(os.sep, "__").replace("/", "__")


def output_paths_for_image(img_path: str) -> list[str]:
    stem = output_stem_for_image(img_path)
    return [
        os.path.join(OUTPUT_DIR, f"{stem}_{idx}.json")
        for idx in range(1, CANDIDATES_PER_IMAGE + 1)
    ]


def slug_prefix(label: str) -> str:
    prefix = ID_PREFIXES.get(label)
    if prefix:
        return prefix
    slug = re.sub(r"[^a-zA-Z0-9가-힣]+", "_", label).strip("_").lower()
    return slug or "object"


def clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, number))


def normalize_bbox(bbox: Any) -> list[float]:
    if not isinstance(bbox, list):
        return [0.0, 0.0, 1.0, 1.0]
    values = [clamp_float(v, 0.0, 1.0, 0.0) for v in bbox[:4]]
    while len(values) < 4:
        values.append(0.0)
    x1, y1, x2, y2 = values
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def normalize_object(
    obj: dict[str, Any], index: int
) -> tuple[dict[str, Any], list[str]]:
    warnings = []
    label = str(obj.get("label", "")).strip() or "자재"
    original_label = label
    label = SYNONYM_MAP.get(label, label)
    if label != original_label:
        warnings.append(f"Normalized label '{original_label}' -> '{label}'")

    obj["label"] = label
    obj["count"] = max(1, int(obj.get("count") or 1))
    obj["bbox"] = normalize_bbox(obj.get("bbox"))
    obj["location"] = str(obj.get("location", "")).strip() or "이미지 내부"
    obj["confidence"] = clamp_float(obj.get("confidence"), 0.0, 1.0, 0.5)
    obj["evidence"] = (
        str(obj.get("evidence", "")).strip() or "이미지에서 해당 객체가 보인다."
    )

    attrs = obj.get("attributes")
    if not isinstance(attrs, dict):
        attrs = {}
    ppe = attrs.get("ppe") if isinstance(attrs.get("ppe"), list) else []
    state = attrs.get("state") if isinstance(attrs.get("state"), list) else []
    obj["attributes"] = {
        "ppe": [str(v) for v in ppe if str(v).strip()],
        "state": [str(v) for v in state if str(v).strip()],
    }
    if label != WORKER_LABEL:
        obj["attributes"]["ppe"] = []

    id_value = str(obj.get("id", "")).strip()
    if not id_value:
        id_value = f"{slug_prefix(label)}_{index}"
    if (
        obj["count"] > 1
        and label in (TIER2_LABELS | TIER3_LABELS)
        and "group_" not in id_value
    ):
        id_value = f"{slug_prefix(label)}_group_{index}"
    obj["id"] = re.sub(r"[^a-zA-Z0-9_]+", "_", id_value).strip("_") or f"object_{index}"

    if label in FORBIDDEN_LABELS:
        warnings.append(f"Forbidden generic label '{label}' on {obj['id']}")
    if label in TIER1_LABELS and obj["count"] > 1:
        warnings.append(
            f"Tier 1 object '{obj['id']}' should be individual; count reset to 1"
        )
        obj["count"] = 1
    return obj, warnings


def canonicalize_object_ids(
    objects: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    counters: dict[str, int] = {}
    id_map: dict[str, str] = {}
    canonical_objects = []

    for obj in objects:
        old_id = obj["id"]
        label = obj["label"]
        prefix = slug_prefix(label)
        is_group = obj.get("count", 1) > 1 and label in (TIER2_LABELS | TIER3_LABELS)
        counter_key = f"{prefix}_group" if is_group else prefix
        counters[counter_key] = counters.get(counter_key, 0) + 1
        new_id = (
            f"{prefix}_group_{counters[counter_key]}"
            if is_group
            else f"{prefix}_{counters[counter_key]}"
        )

        obj = deepcopy(obj)
        obj["id"] = new_id
        id_map[old_id] = new_id
        canonical_objects.append(obj)

    return canonical_objects, id_map


def normalize_object_result(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(data, dict):
        raise ValueError("Top-level output must be an object")
    if (
        not isinstance(data.get("scene_description"), str)
        or not data["scene_description"].strip()
    ):
        raise ValueError("scene_description must be a non-empty string")
    if not isinstance(data.get("objects"), list):
        raise ValueError("objects must be a list")

    warnings = []
    clean_objects = []
    seen_ids = set()
    for index, obj in enumerate(data["objects"], 1):
        if not isinstance(obj, dict):
            warnings.append("Skipped non-object item in objects")
            continue
        clean_obj, obj_warnings = normalize_object(obj, index)
        warnings.extend(obj_warnings)
        if clean_obj["label"] in FORBIDDEN_LABELS:
            warnings.append(
                f"Skipped object with forbidden generic label: {clean_obj['id']} "
                f"({clean_obj['label']})"
            )
            continue
        base_id = clean_obj["id"]
        suffix = 2
        while clean_obj["id"] in seen_ids:
            clean_obj["id"] = f"{base_id}_{suffix}"
            suffix += 1
        seen_ids.add(clean_obj["id"])
        clean_objects.append(clean_obj)

    clean_objects, id_map = canonicalize_object_ids(clean_objects)
    for old_id, new_id in id_map.items():
        if old_id != new_id:
            warnings.append(f"Canonicalized object id '{old_id}' -> '{new_id}'")

    return {
        "scene_description": data["scene_description"].strip(),
        "objects": clean_objects,
    }, warnings


def validate_scene_graph(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    object_result, warnings = normalize_object_result(
        {
            "scene_description": data.get("scene_description", ""),
            "objects": data.get("objects", []),
        }
    )
    valid_ids = {obj["id"] for obj in object_result["objects"]}
    labels_by_id = {obj["id"]: obj["label"] for obj in object_result["objects"]}

    rel_candidates = []
    for rel in data.get("relationships", []):
        if not isinstance(rel, dict):
            warnings.append("Skipped non-object relationship")
            continue
        sid = str(rel.get("sub_id", "")).strip()
        oid = str(rel.get("obj_id", "")).strip()
        pred = str(rel.get("predicate", "")).strip()
        cat = str(rel.get("category", "")).strip()
        if sid not in valid_ids or oid not in valid_ids:
            warnings.append(f"Skipped relationship with unknown id: {sid} -> {oid}")
            continue
        if sid == oid:
            warnings.append(f"Skipped self relationship: {sid}")
            continue
        if pred not in VALID_PREDICATES:
            warnings.append(
                f"Skipped invalid predicate '{pred}' between {sid} and {oid}"
            )
            continue
        if cat not in CATEGORY_PREDICATES or pred not in CATEGORY_PREDICATES[cat]:
            warnings.append(f"Skipped predicate/category mismatch: {pred}/{cat}")
            continue
        rel_candidates.append(
            {
                "sub_id": sid,
                "predicate": pred,
                "obj_id": oid,
                "category": cat,
                "score": clamp_float(rel.get("score"), 0.0, 1.0, 0.5),
                "evidence": str(rel.get("evidence", "")).strip(),
            }
        )

    best_rels_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for rel in rel_candidates:
        pair = (rel["sub_id"], rel["obj_id"])
        current = best_rels_by_pair.get(pair)
        if current is None:
            best_rels_by_pair[pair] = rel
            continue

        rel_rank = (
            RELATION_CATEGORY_PRIORITY.get(rel["category"], 0),
            rel["score"],
        )
        current_rank = (
            RELATION_CATEGORY_PRIORITY.get(current["category"], 0),
            current["score"],
        )
        if rel_rank > current_rank:
            warnings.append(
                f"Replaced duplicate relationship {pair[0]} -> {pair[1]} "
                f"with higher-priority {rel['category']}/{rel['predicate']}"
            )
            best_rels_by_pair[pair] = rel
        else:
            warnings.append(
                f"Skipped lower-priority duplicate relationship {pair[0]} -> {pair[1]} "
                f"{rel['category']}/{rel['predicate']}"
            )

    clean_rels = list(best_rels_by_pair.values())

    functional_pairs = {
        frozenset((rel["sub_id"], rel["obj_id"]))
        for rel in clean_rels
        if rel["category"] == "functional"
    }

    clean_hazards = []
    for haz in data.get("hazards", []):
        if not isinstance(haz, dict):
            warnings.append("Skipped non-object hazard")
            continue
        ref_ids = haz.get("related_object_ids", [])
        if not isinstance(ref_ids, list):
            warnings.append("Skipped hazard with invalid related_object_ids")
            continue
        ref_ids = [str(rid).strip() for rid in ref_ids if str(rid).strip()]
        if not ref_ids or any(rid not in valid_ids for rid in ref_ids):
            warnings.append(f"Skipped hazard with unknown ids: {ref_ids}")
            continue
        hazard = str(haz.get("hazard", "")).strip()
        if hazard not in VALID_HAZARDS:
            warnings.append(f"Skipped invalid hazard '{hazard}'")
            continue
        ref_labels = [labels_by_id.get(rid, "") for rid in ref_ids]
        reason = str(haz.get("reason", "")).strip()
        ref_label_set = set(ref_labels)
        if hazard in {"익수", "추락", "협착"} and WORKER_LABEL not in ref_label_set:
            warnings.append(
                f"Skipped unsupported {hazard} hazard without worker: {ref_ids}"
            )
            continue
        if hazard == "익수":
            if WATER_LABEL not in ref_label_set:
                warnings.append(
                    f"Skipped unsupported 익수 hazard without water: {ref_ids}"
                )
                continue
            has_water_edge = any(
                keyword in reason for keyword in ("가장자리", "물가", "수역", "개방")
            )
            has_missing_protection = any(
                keyword in reason for keyword in ("방호", "난간", "울타리", "안전")
            )
            if not (has_water_edge and has_missing_protection):
                warnings.append(
                    f"Skipped weak 익수 hazard without edge and protection evidence: {ref_ids}"
                )
                continue
        if hazard == "추락":
            if not (ref_label_set & SLOPE_CONTEXT_LABELS):
                warnings.append(
                    f"Skipped unsupported 추락 hazard without fall context: {ref_ids}"
                )
                continue
            has_fall_edge = any(
                keyword in reason
                for keyword in ("가장자리", "고저차", "굴착면", "높", "낙하")
            )
            has_missing_protection = any(
                keyword in reason for keyword in ("방호", "난간", "울타리", "안전망")
            )
            if not (has_fall_edge and has_missing_protection):
                warnings.append(
                    f"Skipped weak 추락 hazard without fall edge and protection evidence: {ref_ids}"
                )
                continue
        if hazard == "협착":
            if not (
                ref_label_set
                & (EQUIPMENT_LABELS | {"콘크리트구조물", "자재", "파이프"})
            ):
                warnings.append(
                    f"Skipped unsupported 협착 hazard without pinch source: {ref_ids}"
                )
                continue
            if not any(
                keyword in reason
                for keyword in ("사이", "끼", "협착", "좁", "장비", "고정")
            ):
                warnings.append(
                    f"Skipped weak 협착 hazard without explicit pinch evidence: {ref_ids}"
                )
                continue
        if hazard == "낙하물" and not any(
            keyword in reason for keyword in ("매달", "상부", "고소", "낙하", "하부")
        ):
            warnings.append(
                f"Skipped weak 낙하물 hazard without overhead evidence: {ref_ids}"
            )
            continue
        if hazard == "전도" and not any(
            keyword in reason for keyword in ("기울", "불안정", "경사", "전도", "넘어")
        ):
            warnings.append(
                f"Skipped weak 전도 hazard without instability evidence: {ref_ids}"
            )
            continue
        if hazard == "충돌":
            has_worker = WORKER_LABEL in ref_label_set
            has_functional_pair = any(
                frozenset(pair) in functional_pairs
                for pair in (
                    (ref_ids[i], ref_ids[j])
                    for i in range(len(ref_ids))
                    for j in range(i + 1, len(ref_ids))
                )
            )
            if has_functional_pair and not has_worker:
                warnings.append(
                    f"Skipped collision hazard for equipment collaboration: {ref_ids}"
                )
                continue
            has_safety_relation = any(
                rel["category"] == "safety"
                and rel["predicate"] in {"too_close_to", "approaching"}
                and rel["sub_id"] in ref_ids
                and rel["obj_id"] in ref_ids
                for rel in clean_rels
            )
            if not has_safety_relation:
                warnings.append(
                    f"Skipped weak collision hazard without safety relation: {ref_ids}"
                )
                continue
            if not any(
                keyword in reason
                for keyword in ("가까", "접근", "경로", "이동", "작동", "충돌")
            ):
                warnings.append(
                    f"Skipped weak collision hazard without proximity/motion evidence: {ref_ids}"
                )
                continue
        clean_hazards.append(
            {
                "related_object_ids": ref_ids,
                "hazard": hazard,
                "reason": reason,
            }
        )

    return {
        "scene_description": object_result["scene_description"],
        "objects": object_result["objects"],
        "relationships": clean_rels,
        "hazards": clean_hazards,
    }, warnings


def to_legacy_output_format(graph: dict[str, Any]) -> dict[str, Any]:
    legacy_objects = []
    for obj in graph.get("objects", []):
        attrs = obj.get("attributes") if isinstance(obj.get("attributes"), dict) else {}
        state = [str(v) for v in attrs.get("state", []) if str(v).strip()]
        legacy_attrs = {"state": state}
        if obj.get("label") == WORKER_LABEL:
            ppe = [str(v) for v in attrs.get("ppe", []) if str(v).strip()]
            legacy_attrs = {"ppe": ppe, "state": state}

        legacy_objects.append(
            {
                "id": obj.get("id", ""),
                "label": obj.get("label", ""),
                "attributes": legacy_attrs,
                "location": obj.get("location", ""),
            }
        )

    return {
        "scene_description": graph.get("scene_description", ""),
        "objects": legacy_objects,
        "relationships": graph.get("relationships", []),
        "hazards": graph.get("hazards", []),
    }


def extract_json_from_response(resp: Any) -> str:
    output_text = getattr(resp, "output_text", None)
    if output_text:
        return output_text
    if getattr(resp, "choices", None):
        return resp.choices[0].message.content
    if getattr(resp, "output", None):
        chunks = []
        for item in resp.output:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
        if chunks:
            return "".join(chunks)
    raise ValueError("Could not extract text from API response")


def call_json_api(
    client: openai.OpenAI,
    system_prompt: str,
    user_text: str,
    data_uri: str,
    schema_name: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema,
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            try:
                resp = client.responses.create(
                    model=MODEL,
                    input=[
                        {
                            "role": "system",
                            "content": [{"type": "input_text", "text": system_prompt}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": user_text},
                                {
                                    "type": "input_image",
                                    "image_url": data_uri,
                                    "detail": "high",
                                },
                            ],
                        },
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": schema_name,
                            "strict": True,
                            "schema": schema,
                        }
                    },
                    temperature=TEMPERATURE,
                )
            except (AttributeError, TypeError):
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_text},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": data_uri, "detail": "high"},
                                },
                            ],
                        },
                    ],
                    response_format=response_format,
                    temperature=TEMPERATURE,
                )
            return json.loads(extract_json_from_response(resp))
        except (openai.RateLimitError, openai.APITimeoutError) as exc:
            print(
                f"  API retryable error ({attempt}/{MAX_RETRIES}): {exc}",
                file=sys.stderr,
            )
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY * attempt)
        except (json.JSONDecodeError, ValueError) as exc:
            print(
                f"  Invalid model output ({attempt}/{MAX_RETRIES}): {exc}",
                file=sys.stderr,
            )
            if attempt == MAX_RETRIES:
                raise

    raise RuntimeError("unreachable")


def call_scene_graph_api(
    client: openai.OpenAI, data_uri: str
) -> tuple[dict[str, Any], list[str]]:
    raw_objects = call_json_api(
        client=client,
        system_prompt=OBJECT_SYSTEM_PROMPT,
        user_text=OBJECT_USER_PROMPT,
        data_uri=data_uri,
        schema_name="construction_objects",
        schema=OBJECT_SCHEMA,
    )
    object_result, warnings = normalize_object_result(raw_objects)

    relation_user_text = (
        RELATION_USER_PROMPT
        + "\n\n입력 objects JSON:\n"
        + json.dumps(object_result, ensure_ascii=False, indent=2)
    )
    raw_relations = call_json_api(
        client=client,
        system_prompt=RELATION_SYSTEM_PROMPT,
        user_text=relation_user_text,
        data_uri=data_uri,
        schema_name="construction_scene_relations",
        schema=RELATION_RESULT_SCHEMA,
    )

    raw_graph = {
        "scene_description": object_result["scene_description"],
        "objects": deepcopy(object_result["objects"]),
        "relationships": raw_relations.get("relationships", []),
        "hazards": raw_relations.get("hazards", []),
    }

    graph, graph_warnings = validate_scene_graph(raw_graph)
    warnings.extend(graph_warnings)
    return graph, warnings


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_image(
    client: openai.OpenAI, img_path: str, index: int, total: int
) -> tuple[bool, list[str], str]:
    name = os.path.basename(img_path)
    out_paths = output_paths_for_image(img_path)

    if all(os.path.exists(out_path) for out_path in out_paths):
        return True, [], f"[{index}/{total}] {name} ... already exists (skipped)"

    data_uri = image_to_data_uri(img_path)
    messages = []
    warnings_all = []

    for candidate_idx, out_path in enumerate(out_paths, 1):
        result, warnings = call_scene_graph_api(client, data_uri)
        warnings_all.extend(f"c{candidate_idx}: {w}" for w in warnings)
        output_result = to_legacy_output_format(result)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_result, f, ensure_ascii=False, indent=2)

        obj_count = len(output_result["objects"])
        group_count = sum(1 for obj in result["objects"] if obj.get("count", 1) > 1)
        rel_count = len(output_result["relationships"])
        haz_count = len(output_result["hazards"])
        messages.append(
            f"c{candidate_idx}: objects={obj_count} groups={group_count} "
            f"rels={rel_count} hazards={haz_count}"
        )

    return (
        True,
        warnings_all,
        f"[{index}/{total}] {name} ... OK " + " | ".join(messages),
    )


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = find_images()
    if MAX_IMAGES is not None:
        images = images[:MAX_IMAGES]

    if not images:
        print(f"ERROR: No images in {IMAGE_DIR}/", file=sys.stderr)
        sys.exit(1)

    print(
        f"Processing {len(images)} images with {MODEL} "
        f"({CANDIDATES_PER_IMAGE} candidates/image, {MAX_WORKERS} workers)...\n"
    )

    success = 0
    fail = 0
    all_warnings = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_image, client, img_path, i, len(images)): img_path
            for i, img_path in enumerate(images, 1)
        }

        for future in as_completed(futures):
            img_path = futures[future]
            try:
                ok, warnings, message = future.result()
                print(message)
                for warning in warnings:
                    print(f"    WARN: {warning}", file=sys.stderr)
                all_warnings.extend(warnings)
                success += int(ok)
                fail += int(not ok)
            except Exception as exc:
                print(f"{os.path.basename(img_path)} ... FAIL {exc}", file=sys.stderr)
                fail += 1

    if all_warnings:
        report_path = os.path.join(OUTPUT_DIR, "_warnings.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(all_warnings, f, ensure_ascii=False, indent=2)
        print(f"\nSaved warnings to {report_path}")

    print(f"\nDone. success={success} fail={fail}")


if __name__ == "__main__":
    main()

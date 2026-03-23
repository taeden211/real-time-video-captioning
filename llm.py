import base64
import json
import os
import sys
import time

from dotenv import load_dotenv

try:
    import openai
except ImportError:
    raise ImportError("pip install openai")

# ── Prompts ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a vision-language annotation model for construction safety scene understanding.

Your task is to analyze one construction-site image and return exactly one valid JSON object only.

Hard output rules:
- Output JSON only.
- Do not output markdown, code fences, explanations, notes, or conversational text.
- Do not omit required keys. Do not add extra keys.
- Never use null. Use empty arrays [] when needed.

Grounding rules:
- Use only visually verifiable facts from the image.
- Do not infer invisible objects, hidden PPE, hidden body parts, or unconfirmed hazards.
- If something is unclear due to occlusion, distance, blur, cropping, lighting, or low resolution, describe it conservatively.
- Ignore tiny, heavily occluded, or ambiguous background objects.

Scene description rules:
- "scene_description": factual, dry Korean sentences summarizing the whole visible work situation.

─────────────────────────────────────────────
OBJECT RULES
─────────────────────────────────────────────

Include all clearly visible construction-safety-relevant objects as separate entries.
Do not collapse multiple objects of the same type into one unless they are truly indistinguishable.

Anti-hallucination rules (CRITICAL):
- NEVER create an object you cannot see in the image.
- Every object MUST have a specific, visible location description in Korean (e.g., "화면 좌측 하단", "중앙 비계 위").
- If you would write "보이지 않음", "확인 불가", or any phrase meaning the object is not visible, do NOT include that object at all.
- Do not create objects just because the scene type implies they should exist.
- Do not create objects to match the scene_description. Only include what you can directly point to in the image.

Object label rules:
- Use a short, specific Korean noun for each object.
- Be as specific as possible (e.g., "로드롤러" not "장비", "타워크레인" not "크레인").
- PREFERRED VOCABULARY — use these exact terms when the object matches:

  중장비: 굴착기, 타워크레인, 이동식크레인, 덤프트럭, 지게차, 로드롤러,
          콘크리트펌프카, 레미콘, 항타기, 고소작업차, 불도저, 천공기, 로더,
          그레이더, 페이버
  차량: 소형트럭, 승용차, 살수차
  구조물: 콘크리트구조물, 교각, 거더, 옹벽, 슬래브, 기초
  가시설: 비계, 동바리, 거푸집, 작업발판, 가설울타리, 임시지보
  안전시설: 난간, 안전네트, 경고표지판, 방호울타리, 신호등
  자재: 철근, 합판, 파이프, H빔, 토사
  기타: 근로자, 사다리, 개구부, 수역, 안전모, 안전대, 가스배관

- If the object does not match any preferred term, use the most standard Korean construction-industry term for it.
- FORBIDDEN generic labels — do NOT use: "장비", "차량", "기계", "구조물", "시설", "물체", "물건"

Id prefixes:
  근로자→worker_  비계→scaffold_  동바리→prop_  지게차→forklift_
  굴착기→excavator_  타워크레인→tower_crane_  이동식크레인→mobile_crane_
  덤프트럭→dump_truck_  로드롤러→roller_  콘크리트펌프카→pump_car_
  레미콘→mixer_  항타기→pile_driver_  고소작업차→aerial_lift_
  불도저→bulldozer_  천공기→drill_  로더→loader_  그레이더→grader_
  페이버→paver_  소형트럭→small_truck_  승용차→car_  살수차→sprinkler_
  콘크리트구조물→structure_  교각→pier_  거더→girder_  옹벽→retaining_wall_
  슬래브→slab_  기초→foundation_  거푸집→formwork_  작업발판→platform_
  가설울타리→temp_fence_  임시지보→temp_support_  안전네트→safety_net_
  경고표지판→sign_  방호울타리→barrier_  신호등→signal_
  철근→rebar_  합판→plywood_  파이프→pipe_  H빔→hbeam_  토사→soil_
  자재→material_  개구부→opening_  난간→guardrail_  사다리→ladder_
  수역→water_  안전모→helmet_  안전대→belt_  가스배관→gas_pipe_
  For unlisted labels, create a reasonable English snake_case prefix.

Number instances left-to-right when possible.
"location": short Korean phrase describing where in the image (must reference a visible position).

Heavy equipment disambiguation (CRITICAL):
- "지게차": has a fork (two prongs) at the front for lifting pallets. Compact body.
- "굴착기": has a boom-arm-bucket structure for digging. Cab rotates on tracks or wheels. Do NOT label this as 지게차 or 크레인.
- "타워크레인": fixed tower with horizontal jib and trolley at top.
- "이동식크레인": truck or crawler mounted crane with telescopic/lattice boom.
- "덤프트럭": large truck with a tiltable cargo bed.
- "로드롤러": heavy cylindrical drum(s) for compacting soil or asphalt.
- "콘크리트펌프카": truck with folding boom arm for pumping concrete.
- "레미콘": truck with rotating drum for mixing/transporting concrete.
- "불도저": tracked vehicle with a wide flat blade at front for pushing soil.
- "로더": wheeled/tracked vehicle with a front-mounted bucket for scooping.
- "고소작업차": vehicle with extendable platform/basket for elevated work.
- "항타기": tall rig with vertical leads for driving piles.

PPE handling:
- PPE being worn → record in that worker's attributes.ppe only. Do NOT create a separate object.
- Create "안전모"/"안전대" as objects only if clearly visible as independent items, not being worn.

─────────────────────────────────────────────
ATTRIBUTE RULES
─────────────────────────────────────────────

Worker objects: 
  "attributes": {"ppe": [...], "state": [...]}

Non-worker objects: "attributes": {"state": [...]}
- Do NOT include the "ppe" key for non-worker objects.

- PPE values (only for workers, only when visually confirmed worn): ["안전모", "안전대"]
- "state": short Korean descriptors only when visually supported.
  Examples: ["가동 중", "정지", "적재 중", "굴착 중", "앉아 있음", "서 있음", "이동 중",
             "설치됨", "해체됨", "손상됨", "노출됨", "미설치", "채워짐", "비어 있음"]

─────────────────────────────────────────────
RELATIONSHIP RULES (CRITICAL FOR SCENE GRAPH QUALITY)
─────────────────────────────────────────────

"relationships" must be an array. If none are clearly visible, use [].

Goal: capture ALL visually evident pairwise spatial, functional, and safety-critical relationships.

Relationship extraction strategy — check these in order:
1. Worker ↔ Equipment: Is a worker operating, approaching, or dangerously close to equipment?
2. Worker ↔ Structure/Edge: Is a worker on, at the edge of a structure, opening, or water?
3. Equipment ↔ Equipment: Are two machines collaborating (e.g., excavator loading dump truck)?
4. Equipment ↔ Structure: Is equipment on, attached to, or working on a structure?
5. Object ↔ Hazard zone: Is any object at an opening, edge, water, or unprotected area?
6. Spatial: General positioning with clear directionality — on, next_to, above, behind, etc.

FORBIDDEN relations — do NOT use:
- "near" — too vague, no directionality. Use "next_to", "too_close_to", or a functional relation instead.
- "at_risk_of" — hazard assessment belongs in the "hazards" field, not in relationships.

Allowed relations (use exactly):
["on", "under", "above", "below", "next_to", "inside",
 "attached_to", "behind", "in_front_of",
 "walking_on", "working_on", "carrying", "operating",
 "approaching", "blocking", "too_close_to",
 "loading", "supported_by", "connected_to", "along"]

Relation tiers (prefer higher tiers — use the most specific relation that applies):
- Tier 1 — Functional: operating, loading, carrying, working_on, walking_on
- Tier 2 — Structural: on, attached_to, supported_by, connected_to, inside
- Tier 3 — Spatial (directional): next_to, above, below, under, behind, in_front_of, along
- Safety-critical: too_close_to, approaching, blocking

For each object pair, fill in the appropriate relation per tier within a single relationship entry. 
If no relation exists for a tier, use "". 
Do not duplicate semantically equivalent relations across tiers (e.g., "on" in Structural and "above" in Spatial for the same pair — pick one tier only).

Do not duplicate semantically redundant relations between the same pair.
Do not invent relationships not visible in the image.

Minimum relationship guideline:
- If 2+ objects are visible, there should usually be at least 1 relationship.
- Equipment working together (e.g., excavator + dump truck) MUST have a relationship like "loading".

─────────────────────────────────────────────
HAZARD RULES
─────────────────────────────────────────────

"hazards" must be an array. If no clear hazard is visually supported, use [].
Only create a hazard when there is direct visual evidence.

Allowed hazard labels:
["추락", "낙하물", "충돌", "협착", "전도", "감전", "익수"]

Hazard detection checklist — evaluate each:
- 추락: worker at height, near unprotected edge, on structure without guardrail
- 낙하물: suspended load, material on elevated surface, overhead work
- 충돌: worker next to operating/moving equipment, equipment next to each other
- 협착: worker between machine and fixed object, pinch points
- 전도: unstable ladder, top-heavy load, equipment on slope
- 감전: exposed wiring, electrical equipment, work near power lines
- 익수: worker at water edge without barriers

Each hazard must reference valid object ids in "related_object_ids".
"reason": one short factual Korean sentence.

─────────────────────────────────────────────
JSON SCHEMA
─────────────────────────────────────────────

{
  "scene_description": "string",
  "objects": [
    {
      "id": "string",
      "label": "string",
      "attributes": {
        "ppe": ["string"], // workers only
        "state": ["string"]
      },
      "location": "string (must describe a visible position in the image)"
    }
  ],
  "relationships": [
    {
        "subject_id": "string",
        "Functional": "string",
        "Structural": "string",
        "Spatial": "string",
        "Safety-critical": "string",
        "object_id": "string"
    }
  ],
  "hazards": [
    {
      "related_object_ids": ["string"],
      "hazard": "string",
      "reason": "string"
    }
  ]
}"""

USER_PROMPT = """Analyze the provided construction-site image and generate exactly one JSON object following the schema and rules in your system instructions.

Step-by-step:
1. Scan the entire image and list all safety-relevant objects you can actually SEE. Do not add objects that are not visible.
2. Identify each piece of heavy equipment by its defining visual features (boom-arm-bucket = 굴착기, fork prongs = 지게차, wire-hook boom = 타워크레인/이동식크레인, tiltable bed = 덤프트럭, cylindrical drum = 로드롤러, rotating drum on truck = 레미콘, folding pump boom = 콘크리트펌프카, front blade = 불도저, front bucket on wheels = 로더, tall vertical leads = 항타기, elevated platform = 고소작업차).
3. For every pair of nearby objects, determine if a spatial, functional, or safety-critical relationship exists. Do NOT use "near" — always choose a specific directional or functional relation. Prefer Tier 1 (functional) over Tier 3 (spatial).
4. Check the hazard checklist against every object and relationship.
5. Output valid JSON only."""

# ── Config ───────────────────────────────────────────────

MODEL = "gpt-5.4"
IMAGE_DIR = "sample_data"
OUTPUT_DIR = "sample_output"
MAX_IMAGES = 10
MAX_RETRIES = 3
RETRY_DELAY = 5

# ── Ontology ─────────────────────────────────────────────

PREFERRED_LABELS = {
    # 중장비
    "굴착기",
    "타워크레인",
    "이동식크레인",
    "덤프트럭",
    "지게차",
    "로드롤러",
    "콘크리트펌프카",
    "레미콘",
    "항타기",
    "고소작업차",
    "불도저",
    "천공기",
    "로더",
    "그레이더",
    "페이버",
    # 차량
    "소형트럭",
    "승용차",
    "살수차",
    # 구조물
    "콘크리트구조물",
    "교각",
    "거더",
    "옹벽",
    "슬래브",
    "기초",
    # 가시설
    "비계",
    "동바리",
    "거푸집",
    "작업발판",
    "가설울타리",
    "임시지보",
    # 안전시설
    "난간",
    "안전네트",
    "경고표지판",
    "방호울타리",
    "신호등",
    # 자재
    "철근",
    "합판",
    "파이프",
    "H빔",
    "토사",
    "자재",
    # 기타
    "근로자",
    "사다리",
    "개구부",
    "수역",
    "안전모",
    "안전대",
    "가스배관",
}

SYNONYM_MAP = {
    # 굴착기 동의어
    "백호": "굴착기",
    "포클레인": "굴착기",
    "유압셔블": "굴착기",
    "엑스카베이터": "굴착기",
    "파워셔블": "굴착기",
    # 크레인 동의어
    "크레인": "이동식크레인",
    "카고크레인": "이동식크레인",
    "기중기": "이동식크레인",
    # 롤러 동의어
    "롤러": "로드롤러",
    "다짐롤러": "로드롤러",
    "진동롤러": "로드롤러",
    "머캐덤롤러": "로드롤러",
    "탠덤롤러": "로드롤러",
    "타이어롤러": "로드롤러",
    # 레미콘 동의어
    "콘크리트믹서트럭": "레미콘",
    "믹서트럭": "레미콘",
    "콘크리트믹서": "레미콘",
    # 기타 동의어
    "봉고차": "소형트럭",
    "화물차": "소형트럭",
    "휠로더": "로더",
    "프론트로더": "로더",
    "모터그레이더": "그레이더",
    "아스팔트피니셔": "페이버",
    "피니셔": "페이버",
    "콘크리트펌프": "콘크리트펌프카",
    "펌프카": "콘크리트펌프카",
    "말뚝항타기": "항타기",
    "파일드라이버": "항타기",
    "고소차": "고소작업차",
    "스카이차": "고소작업차",
    "스카이": "고소작업차",
    "안전표지판": "경고표지판",
    "표지판": "경고표지판",
    "가설펜스": "가설울타리",
    "임시펜스": "가설울타리",
}

FORBIDDEN_LABELS = {"장비", "차량", "기계", "구조물", "시설", "물체", "물건", "중장비"}

VALID_RELATIONS = {
    "on",
    "under",
    "above",
    "below",
    "next_to",
    "inside",
    "attached_to",
    "behind",
    "in_front_of",
    "walking_on",
    "working_on",
    "carrying",
    "operating",
    "approaching",
    "blocking",
    "too_close_to",
    "loading",
    "supported_by",
    "connected_to",
    "along",
}

FORBIDDEN_RELATIONS = {"near", "at_risk_of"}

VALID_HAZARDS = {"추락", "낙하물", "충돌", "협착", "전도", "감전", "익수"}

REQUIRED_KEYS = {"scene_description", "objects", "relationships", "hazards"}

INVISIBLE_KEYWORDS = {"보이지 않음", "확인 불가", "보이지 않는", "식별 불가", "없음"}

# ── Helpers ──────────────────────────────────────────────


def image_to_data_uri(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def validate_json(text: str) -> dict:
    data = json.loads(text)
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    if not isinstance(data["objects"], list):
        raise ValueError("objects must be a list")

    # ── Pass 1: Remove hallucinated objects + normalize labels ──
    valid_ids = set()
    removed_ids = set()
    clean_objects = []
    warnings = []
    novel_labels = []

    for obj in data["objects"]:
        if not all(k in obj for k in ("id", "label", "attributes", "location")):
            raise ValueError(f"Object missing required fields: {obj.get('id', '?')}")

        loc = obj.get("location", "")
        if any(kw in loc for kw in INVISIBLE_KEYWORDS):
            removed_ids.add(obj["id"])
            warnings.append(
                f"Removed hallucinated object '{obj['id']}' (location: '{loc}')"
            )
            continue

        # Synonym normalization
        original_label = obj["label"]
        obj["label"] = SYNONYM_MAP.get(obj["label"], obj["label"])
        if original_label != obj["label"]:
            warnings.append(
                f"Normalized label '{original_label}' → '{obj['label']}' on {obj['id']}"
            )

        # Forbidden generic labels
        if obj["label"] in FORBIDDEN_LABELS:
            warnings.append(
                f"FORBIDDEN generic label '{obj['label']}' on {obj['id']} — needs manual review"
            )

        # Novel label detection
        if (
            obj["label"] not in PREFERRED_LABELS
            and obj["label"] not in FORBIDDEN_LABELS
        ):
            novel_labels.append(obj["label"])
            warnings.append(
                f"Novel label '{obj['label']}' on {obj['id']} — not in preferred vocabulary"
            )

        valid_ids.add(obj["id"])
        clean_objects.append(obj)

    data["objects"] = clean_objects

    # ── Pass 2: Remove relationships referencing removed objects ──
    clean_rels = []
    TIER_KEYS = ["Functional", "Structural", "Spatial", "Safety-critical"]

    for rel in data.get("relationships", []):
        sid, oid = rel.get("subject_id"), rel.get("object_id")
        if sid in removed_ids or oid in removed_ids:
            warnings.append(
                f"Removed relationship {sid}→{oid} (references removed object)"
            )
            continue
        for tier in TIER_KEYS:
            r = rel.get(tier, "")
            if not r:
                continue
            if r in FORBIDDEN_RELATIONS:
                warnings.append(
                    f"Forbidden relation '{r}' in {tier} between {sid}→{oid}"
                )
            elif r not in VALID_RELATIONS:
                warnings.append(f"Invalid relation '{r}' in {tier}")
        if sid not in valid_ids:
            warnings.append(f"Unknown subject_id '{sid}'")
        if oid not in valid_ids:
            warnings.append(f"Unknown object_id '{oid}'")
        clean_rels.append(rel)

    data["relationships"] = clean_rels

    # ── Pass 3: Remove hazards referencing removed objects ──
    clean_hazards = []
    for haz in data.get("hazards", []):
        ref_ids = haz.get("related_object_ids", [])
        if any(rid in removed_ids for rid in ref_ids):
            warnings.append(f"Removed hazard referencing removed object(s): {ref_ids}")
            continue
        if haz.get("hazard") not in VALID_HAZARDS:
            warnings.append(f"Invalid hazard '{haz.get('hazard')}'")
        for rid in ref_ids:
            if rid not in valid_ids:
                warnings.append(f"Hazard references unknown id '{rid}'")
        clean_hazards.append(haz)

    data["hazards"] = clean_hazards

    if warnings:
        for w in warnings:
            print(f"    WARN: {w}", file=sys.stderr)

    return data, novel_labels


def call_api(client: openai.OpenAI, data_uri: str) -> tuple:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_uri, "detail": "high"},
                            },
                        ],
                    },
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            return validate_json(raw)
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            print(
                f"  ⚠ API error (attempt {attempt}/{MAX_RETRIES}): {e}", file=sys.stderr
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise
        except (json.JSONDecodeError, ValueError) as e:
            print(
                f"  ⚠ Invalid JSON (attempt {attempt}/{MAX_RETRIES}): {e}",
                file=sys.stderr,
            )
            if attempt < MAX_RETRIES:
                continue
            else:
                raise


# ── Main ─────────────────────────────────────────────────


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = sorted(
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )[:MAX_IMAGES]

    if not images:
        print(f"ERROR: No images in {IMAGE_DIR}/", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(images)} images with {MODEL}...\n")

    success, fail = 0, 0
    all_novel_labels = {}  # label -> count

    for i, img_path in enumerate(images, 1):
        name = os.path.basename(img_path)
        out_name = os.path.splitext(name)[0] + ".json"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        # 이미 파일이 존재하면 건너뛰는 코드 추가
        if os.path.exists(out_path):
            print(f"[{i}/{len(images)}] {name} ... 이미 존재함 (건너뜀)")
            success += 1
            continue
        try:
            data_uri = image_to_data_uri(img_path)
            result, novel = call_api(client, data_uri)

            for lbl in novel:
                all_novel_labels[lbl] = all_novel_labels.get(lbl, 0) + 1

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            obj_count = len(result["objects"])
            rel_count = len(result["relationships"])
            haz_count = len(result["hazards"])
            print(f"✓  objects={obj_count} rels={rel_count} hazards={haz_count}")
            success += 1

        except Exception as e:
            print(f"✗  {e}", file=sys.stderr)
            fail += 1

    # ── Novel label report ──
    if all_novel_labels:
        print("\n── Novel labels (not in preferred vocabulary) ──")
        for lbl, cnt in sorted(all_novel_labels.items(), key=lambda x: -x[1]):
            print(f"  {lbl}: {cnt}")
        report_path = os.path.join(OUTPUT_DIR, "_novel_labels.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(all_novel_labels, f, ensure_ascii=False, indent=2)
        print(f"Saved to {report_path}")

    print(f"\nDone. success={success} fail={fail}")


if __name__ == "__main__":
    main()

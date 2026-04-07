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

Grouping rules:
- When 3 or more objects share the SAME label AND occupy the SAME spatial zone AND have similar states, merge them into ONE group entry with a "count" field.
- Group id uses the standard prefix + "group_" + number (e.g., "worker_group_1", "opening_group_1").
- "location" describes the zone the group occupies (e.g., "슬래브 가장자리 일대", "비계 상부 전반").
- Objects that are individually distinguishable by unique actions, unique PPE status, or unique relationships to other objects MUST remain as separate entries even if label matches.
- If count is 1, omit the "count" field entirely (default single object).

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
  For group entries, insert "group_" after the prefix (e.g., worker_group_1).
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
RELATIONSHIP RULES
─────────────────────────────────────────────

"relationships" must be an array. If none are clearly visible, use [].

Goal: capture ALL visually evident pairwise relationships using a flat per-edge schema.

Each relationship is one edge: {sub_id, predicate, obj_id, category, score, evidence}.

Category → allowed predicates (closed sets, use EXACTLY one):
  functional  : operating, loading, carrying, working_on, walking_on
  structural  : on, inside, attached_to, supported_by, connected_to
  spatial     : next_to, above, below, behind, in_front_of
  safety      : too_close_to, approaching, blocking

Rules:
- "predicate" MUST belong to the closed set of its "category". No cross-category predicates.
- For a given (sub_id, obj_id) pair, output at most ONE relationship — choose the highest-priority category:
    safety > functional > structural > spatial
- "score": float 0.0–1.0 indicating visual confidence for this relation.
- "evidence": one short factual Korean sentence describing the visual basis for the relation.
- Do NOT use "near" or "at_risk_of" — these are FORBIDDEN.
- Do not invent relationships not visible in the image.
- If 2+ objects are visible, there should usually be at least 1 relationship.
- Equipment working together (e.g., excavator + dump truck) MUST have a relationship like "loading".

Relationship extraction strategy — check in order:
1. Worker ↔ Equipment: operating, approaching, too_close_to?
2. Worker ↔ Structure/Edge: on, walking_on, too_close_to?
3. Equipment ↔ Equipment: loading, next_to, too_close_to?
4. Equipment ↔ Structure: on, working_on, attached_to?
5. Object ↔ Hazard zone: too_close_to, approaching?
6. Remaining spatial: next_to, above, below, behind, in_front_of

─────────────────────────────────────────────
HAZARD RULES
─────────────────────────────────────────────

"hazards" must be an array. If no clear hazard is visually supported, use [].
Only create a hazard when there is direct visual evidence.

Allowed hazard labels:
["추락", "낙하물", "충돌", "협착", "전도", "감전", "익수"]

Hazard detection checklist — evaluate each:
- 추락: BOTH conditions must be visually confirmed simultaneously:
    (1) A worker is visibly at an elevated position (비계, 슬래브 가장자리, 구조물 상부 등).
    (2) The absence of guardrail/safety net at that specific location is CLEARLY VISIBLE in the image.
    If the edge area is cropped, occluded, too distant, or ambiguous — do NOT generate a 추락 hazard.
    "가드레일이 보이지 않는다" ≠ "가드레일이 없다". Only confirm absence when the area is fully visible and clearly unprotected.
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
      "count": int,           // optional, only when ≥ 2 (grouped objects)
      "attributes": {
        "ppe": ["string"],    // workers only
        "state": ["string"]
      },
      "location": "string"
    }
  ],
  "relationships": [
    {
      "sub_id": "string",
      "predicate": "string",
      "obj_id": "string",
      "category": "string",
      "score": float,
      "evidence": "string"
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
1. Scan the entire image. List all safety-relevant objects you can actually SEE. Group 3+ identical objects in the same zone into one group entry with "count". Keep individually distinguishable objects separate.
2. Identify heavy equipment by defining visual features (boom-arm-bucket = 굴착기, fork prongs = 지게차, wire-hook boom = 타워크레인/이동식크레인, tiltable bed = 덤프트럭, cylindrical drum = 로드롤러, rotating drum on truck = 레미콘, folding pump boom = 콘크리트펌프카, front blade = 불도저, front bucket on wheels = 로더, tall vertical leads = 항타기, elevated platform = 고소작업차).
3. For every pair of nearby objects, determine the single best relationship. Pick the highest-priority category (safety > functional > structural > spatial). Assign a confidence score and one-sentence Korean evidence.
4. Check the hazard checklist. For 추락: confirm BOTH (a) worker at height AND (b) clearly visible absence of guardrail at that exact location.
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
    "소형트럭",
    "승용차",
    "살수차",
    "콘크리트구조물",
    "교각",
    "거더",
    "옹벽",
    "슬래브",
    "기초",
    "비계",
    "동바리",
    "거푸집",
    "작업발판",
    "가설울타리",
    "임시지보",
    "난간",
    "안전네트",
    "경고표지판",
    "방호울타리",
    "신호등",
    "철근",
    "합판",
    "파이프",
    "H빔",
    "토사",
    "자재",
    "근로자",
    "사다리",
    "개구부",
    "수역",
    "안전모",
    "안전대",
    "가스배관",
}

SYNONYM_MAP = {
    "백호": "굴착기",
    "포클레인": "굴착기",
    "유압셔블": "굴착기",
    "엑스카베이터": "굴착기",
    "파워셔블": "굴착기",
    "크레인": "이동식크레인",
    "카고크레인": "이동식크레인",
    "기중기": "이동식크레인",
    "롤러": "로드롤러",
    "다짐롤러": "로드롤러",
    "진동롤러": "로드롤러",
    "머캐덤롤러": "로드롤러",
    "탠덤롤러": "로드롤러",
    "타이어롤러": "로드롤러",
    "콘크리트믹서트럭": "레미콘",
    "믹서트럭": "레미콘",
    "콘크리트믹서": "레미콘",
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

CATEGORY_PREDICATES = {
    "functional": {"operating", "loading", "carrying", "working_on", "walking_on"},
    "structural": {"on", "inside", "attached_to", "supported_by", "connected_to"},
    "spatial": {"next_to", "above", "below", "behind", "in_front_of"},
    "safety": {"too_close_to", "approaching", "blocking"},
}

ALL_VALID_PREDICATES = set()
for s in CATEGORY_PREDICATES.values():
    ALL_VALID_PREDICATES |= s

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


def validate_json(text: str) -> tuple:
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

        original_label = obj["label"]
        obj["label"] = SYNONYM_MAP.get(obj["label"], obj["label"])
        if original_label != obj["label"]:
            warnings.append(
                f"Normalized label '{original_label}' → '{obj['label']}' on {obj['id']}"
            )

        if obj["label"] in FORBIDDEN_LABELS:
            warnings.append(
                f"FORBIDDEN generic label '{obj['label']}' on {obj['id']} — needs manual review"
            )

        if (
            obj["label"] not in PREFERRED_LABELS
            and obj["label"] not in FORBIDDEN_LABELS
        ):
            novel_labels.append(obj["label"])
            warnings.append(
                f"Novel label '{obj['label']}' on {obj['id']} — not in preferred vocabulary"
            )

        # Validate count field
        count = obj.get("count")
        if count is not None:
            if not isinstance(count, int) or count < 2:
                warnings.append(
                    f"Invalid count={count} on {obj['id']} — removed count field"
                )
                obj.pop("count", None)

        valid_ids.add(obj["id"])
        clean_objects.append(obj)

    data["objects"] = clean_objects

    # ── Pass 2: Validate relationships (new flat schema) ──
    clean_rels = []
    seen_pairs = set()

    for rel in data.get("relationships", []):
        sid = rel.get("sub_id", "")
        oid = rel.get("obj_id", "")
        pred = rel.get("predicate", "")
        cat = rel.get("category", "")
        score = rel.get("score", 0.0)

        if sid in removed_ids or oid in removed_ids:
            warnings.append(
                f"Removed relationship {sid}→{oid} (references removed object)"
            )
            continue

        if sid not in valid_ids:
            warnings.append(f"Unknown sub_id '{sid}'")
        if oid not in valid_ids:
            warnings.append(f"Unknown obj_id '{oid}'")

        if pred in FORBIDDEN_RELATIONS:
            warnings.append(
                f"Forbidden predicate '{pred}' between {sid}→{oid} — skipped"
            )
            continue

        if pred not in ALL_VALID_PREDICATES:
            warnings.append(f"Invalid predicate '{pred}' between {sid}→{oid}")

        if cat not in CATEGORY_PREDICATES:
            warnings.append(f"Invalid category '{cat}' between {sid}→{oid}")
        elif pred not in CATEGORY_PREDICATES.get(cat, set()):
            warnings.append(
                f"Predicate '{pred}' does not belong to category '{cat}' between {sid}→{oid}"
            )

        if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
            warnings.append(f"Invalid score={score} between {sid}→{oid}, clamping")
            score = max(
                0.0, min(1.0, float(score) if isinstance(score, (int, float)) else 0.5)
            )
            rel["score"] = score

        pair_key = (sid, oid)
        if pair_key in seen_pairs:
            warnings.append(f"Duplicate pair {sid}→{oid} — keeping first only")
            continue
        seen_pairs.add(pair_key)

        clean_rels.append(rel)

    data["relationships"] = clean_rels

    # ── Pass 3: Validate hazards ──
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
    all_novel_labels = {}

    for i, img_path in enumerate(images, 1):
        name = os.path.basename(img_path)
        out_name = os.path.splitext(name)[0] + ".json"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        if os.path.exists(out_path):
            print(f"[{i}/{len(images)}] {name} ... 이미 존재함 (건너뜀)")
            success += 1
            continue

        print(f"[{i}/{len(images)}] {name} ... ", end="", flush=True)
        try:
            data_uri = image_to_data_uri(img_path)
            result, novel = call_api(client, data_uri)

            for lbl in novel:
                all_novel_labels[lbl] = all_novel_labels.get(lbl, 0) + 1

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            obj_count = len(result["objects"])
            grp_count = sum(1 for o in result["objects"] if o.get("count", 0) >= 2)
            rel_count = len(result["relationships"])
            haz_count = len(result["hazards"])
            print(
                f"✓  objects={obj_count} (groups={grp_count}) rels={rel_count} hazards={haz_count}"
            )
            success += 1

        except Exception as e:
            print(f"✗  {e}", file=sys.stderr)
            fail += 1

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

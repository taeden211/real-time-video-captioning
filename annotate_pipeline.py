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

SYSTEM_PROMPT = """You are a construction-site image annotation model.

Return exactly one JSON object and nothing else.

ABSOLUTE OUTPUT RULES:
- Output must be valid JSON.
- Do not output markdown, code fences, comments, explanations, notes, or extra text.
- Do not omit required top-level keys.
- Do not add extra keys.
- Never use null. Use [] when appropriate.
- All strings must be valid JSON strings with double quotes.
- If unsure about any field value, choose the most conservative valid value or omit that object, relation, or hazard.
- If a relation or hazard is not clearly supported, output [] for that part instead of guessing.

TOP-LEVEL SCHEMA:
{
  "scene_description": "string",
  "objects": [
    {
      "id": "string",
      "label": "string",
      "count": int,
      "attributes": {
        "ppe": ["string"],
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
}

GROUNDING RULES:
- Use only visually verifiable facts from the image.
- Never infer hidden objects, hidden PPE, hidden body parts, hidden edges, or hidden hazards.
- If an object is tiny, blurred, cropped, backlit, heavily occluded, or ambiguous, do not include it.
- If you cannot point to the object's visible position in the image, do not include it.
- Do not create objects just because they are common in construction scenes.

SCENE DESCRIPTION RULES:
- Write 1 to 3 short factual Korean sentences.
- Describe only what is clearly visible at the whole-scene level.
- Do not mention hazards unless directly visible.

OBJECT RULES:
- Every object must have a visible location in Korean, a visually grounded label, and attributes supported by the image.
- Use short specific Korean nouns.
- Do not use generic labels like "장비", "차량", "기계", "구조물", "시설", "물체", "물건".
- If 3 or more same-label objects are in the same zone and visually similar, group them into one entry with "count".
- If a worker is individually distinguishable by action, PPE, or relation, keep that worker separate.
- If count is 1, omit "count".
- Use these preferred labels when the visible object matches:
  중장비: 굴착기, 타워크레인, 이동식크레인, 덤프트럭, 지게차, 로드롤러, 콘크리트펌프카, 레미콘, 항타기, 고소작업차, 불도저, 천공기, 로더, 그레이더, 페이버
  차량: 소형트럭, 승용차, 살수차
  구조물: 콘크리트구조물, 교각, 거더, 옹벽, 슬래브, 기초
  가시설: 비계, 동바리, 거푸집, 작업발판, 가설울타리, 임시지보
  안전시설: 난간, 안전네트, 경고표지판, 방호울타리, 신호등
  자재: 철근, 합판, 파이프, H빔, 토사
  기타: 근로자, 사다리, 개구부, 수역, 안전모, 안전대, 가스배관

ID RULES:
- Use the standard English snake_case prefix for ids.
- For grouped entries, insert "group_" after the prefix.
- Number instances left-to-right when possible.

LOCATION RULES:
- "location" must be a short Korean phrase tied to visible image position.
- Good examples: "화면 상단 중앙", "화면 우측 하단", "화면 중앙 좌측 비계 위"
- Bad examples: "잘 안 보임", "확인 불가", "있을 것으로 보임"

WORKER ATTRIBUTE RULES:
- Worker objects use {"ppe": [...], "state": [...]}.
- Non-worker objects use {"state": [...]}.
- Never include "ppe" for non-worker objects.
- Record PPE only when clearly visible as worn.
- If helmet or safety belt is not clearly visible, do not guess.
- Allowed worker PPE values: ["안전모", "안전대"]
- "state" values must be short Korean descriptors supported directly by the image.

RELATIONSHIP RULES:
- "relationships" must be an array. If no clear relation is visible, use [].
- Only include pairwise relations with direct visual evidence.
- For each (sub_id, obj_id), output at most one relation.
- Use the highest-priority category:
  safety > functional > structural > spatial
- Allowed predicates:
  functional: operating, loading, carrying, working_on, walking_on
  structural: on, inside, attached_to, supported_by, connected_to
  spatial: next_to, above, below, behind, in_front_of
  safety: too_close_to, approaching, blocking
- "evidence" must be one short factual Korean sentence.
- Describe what is visibly seen, not inferred intent.
- If motion or operation is not visually clear, do not use operating or loading.

HEAVY EQUIPMENT DISAMBIGUATION:
- 지게차: front fork prongs clearly visible
- 굴착기: boom-arm-bucket clearly visible
- 타워크레인: fixed tower mast and jib at top
- 이동식크레인: vehicle-mounted boom crane
- 덤프트럭: tilting dump bed
- 로드롤러: large cylindrical roller drum
- 콘크리트펌프카: truck with folding concrete pump boom
- 레미콘: truck with rotating drum
- 로더: front bucket loader
- 불도저: front blade pushing vehicle
- 고소작업차: elevated basket or platform
- 항타기: tall vertical pile-driving rig
- If the diagnostic feature is not clearly visible, choose a more conservative label or omit the object.

HAZARD RULES:
- "hazards" must be an array. If no clear hazard is visible, use [].
- Only create a hazard when there is direct visual evidence.
- Never create a hazard from scene expectation alone.
- Allowed hazard labels: ["추락", "낙하물", "충돌", "협착", "전도", "감전", "익수"]
- Hazard criteria:
  - 추락: worker is clearly at height, the exact nearby edge or opening is clearly visible, and the lack of guardrail or safety protection at that exact point is clearly visible. If edge visibility is partial or ambiguous, do not output 추락.
  - 낙하물: suspended load, overhead work, or elevated unsecured material is clearly visible.
  - 충돌: worker is clearly very close to operating or moving equipment, or two machines are clearly very close during operation or movement. Do not use 충돌 for a worker merely standing near static stacked material.
  - 협착: worker is visibly between machine and fixed object, or at a clear pinch point.
  - 전도: unstable ladder, unstable top-heavy load, or machine on risky tilt or slope is clearly visible.
  - 감전: exposed wiring, electrical equipment, or work near power lines is clearly visible.
  - 익수: worker is at water edge and lack of barrier is clearly visible.
- Each hazard must reference valid object ids in "related_object_ids".
- "reason" must be one short factual Korean sentence.

FINAL CHECK BEFORE OUTPUT:
1. Remove any object that is not clearly visible.
2. Remove any relation that is not directly supported.
3. Remove any hazard based on assumption.
4. Ensure every referenced id exists.
5. Ensure the result is valid JSON only.
"""
USER_PROMPT = """Analyze this construction-site image and output exactly one valid JSON object.

Work in this order:

1. Scan the whole image.
- List only clearly visible safety-relevant objects.
- Ignore tiny, ambiguous, heavily occluded, or uncertain items.

2. Build objects.
- Use specific Korean labels.
- Give every object a precise visible location in Korean.
- Group only when 3 or more same-label objects are in the same zone with similar appearance or state.
- Keep distinguishable workers separate.

3. Add attributes.
- For workers, record only clearly visible worn PPE.
- Do not guess hidden helmets or belts.
- Add short Korean state words only when directly visible.

4. Add relationships.
- Add only visually obvious pairwise relations.
- Prefer safety > functional > structural > spatial.
- If no strong relation is visible, use [].

5. Add hazards conservatively.
- Only add a hazard when the checklist is visually satisfied.
- Do not create hazard labels from static material alone.
- For 추락, confirm both height and a clearly unprotected edge or opening at that exact location.

6. Validate before output.
- Make sure every string is properly quoted.
- Make sure every referenced id exists.
- Make sure the final output is valid JSON only.

Return JSON only."""
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
    if not isinstance(data, dict):
        raise ValueError("Top-level output must be a JSON object")

    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    if not isinstance(data["scene_description"], str) or not data["scene_description"].strip():
        raise ValueError("scene_description must be a non-empty string")

    if not isinstance(data["objects"], list):
        raise ValueError("objects must be a list")
    if not isinstance(data["relationships"], list):
        raise ValueError("relationships must be a list")
    if not isinstance(data["hazards"], list):
        raise ValueError("hazards must be a list")

    # ── Pass 1: Remove hallucinated objects + normalize labels ──
    valid_ids = set()
    removed_ids = set()
    clean_objects = []
    warnings = []
    novel_labels = []

    for obj in data["objects"]:
        if not all(k in obj for k in ("id", "label", "attributes", "location")):
            raise ValueError(f"Object missing required fields: {obj.get('id', '?')}")
        if not isinstance(obj["id"], str) or not obj["id"].strip():
            raise ValueError("Object id must be a non-empty string")
        if not isinstance(obj["label"], str) or not obj["label"].strip():
            raise ValueError(f"Object label must be a non-empty string: {obj['id']}")
        if not isinstance(obj["location"], str) or not obj["location"].strip():
            raise ValueError(f"Object location must be a non-empty string: {obj['id']}")
        if not isinstance(obj["attributes"], dict):
            raise ValueError(f"Object attributes must be a dict: {obj['id']}")

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

        attrs = obj["attributes"]
        state = attrs.get("state")
        if not isinstance(state, list) or not all(isinstance(x, str) for x in state):
            raise ValueError(f"attributes.state must be a string list: {obj['id']}")

        if obj["label"] == "근로자":
            if "ppe" not in attrs:
                raise ValueError(f"Worker missing ppe field: {obj['id']}")
            if not isinstance(attrs["ppe"], list) or not all(
                isinstance(x, str) for x in attrs["ppe"]
            ):
                raise ValueError(f"attributes.ppe must be a string list: {obj['id']}")
        elif "ppe" in attrs:
            warnings.append(
                f"Removed non-worker ppe field from {obj['id']} ({obj['label']})"
            )
            attrs.pop("ppe", None)

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
        if not isinstance(rel, dict):
            raise ValueError("Each relationship must be an object")
        sid = rel.get("sub_id", "")
        oid = rel.get("obj_id", "")
        pred = rel.get("predicate", "")
        cat = rel.get("category", "")
        score = rel.get("score", 0.0)

        if not all(
            isinstance(val, str) and val.strip()
            for val in (sid, oid, pred, cat, rel.get("evidence", ""))
        ):
            raise ValueError("Relationship fields must be non-empty strings")

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
        if not isinstance(haz, dict):
            raise ValueError("Each hazard must be an object")
        ref_ids = haz.get("related_object_ids", [])
        if not isinstance(ref_ids, list) or not all(
            isinstance(rid, str) and rid.strip() for rid in ref_ids
        ):
            raise ValueError("related_object_ids must be a string list")
        if not isinstance(haz.get("hazard"), str) or not isinstance(
            haz.get("reason"), str
        ):
            raise ValueError("Hazard fields must be strings")
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


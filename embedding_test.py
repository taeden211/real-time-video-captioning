import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
DEFAULT_INPUT_DIR = Path("./sample_output")
REQUIRED_KEYS = {"scene_description", "objects", "relationships", "hazards"}

CATEGORY_ORDER = ["safety", "functional", "structural", "spatial"]
CATEGORY_WEIGHTS = {
    "safety": 1.5,
    "functional": 1.2,
    "structural": 1.0,
    "spatial": 0.6,
}
SCENE_WEIGHT = 0.55
GRAPH_WEIGHT = 0.45


print("KoSBERT 모델을 로드 중입니다...")
kosbert_model = SentenceTransformer(MODEL_NAME)
TEXT_EMBED_DIM = (
    kosbert_model.get_embedding_dimension()
    if hasattr(kosbert_model, "get_embedding_dimension")
    else kosbert_model.get_sentence_embedding_dimension()
)


def encode_texts(texts):
    if not texts:
        return torch.zeros((0, TEXT_EMBED_DIM), dtype=torch.float32)
    return torch.tensor(kosbert_model.encode(texts), dtype=torch.float32)


def weighted_mean(embeddings, weights=None):
    if embeddings.numel() == 0:
        return torch.zeros((1, TEXT_EMBED_DIM), dtype=torch.float32)

    if weights is None:
        return embeddings.mean(dim=0, keepdim=True)

    weight_tensor = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
    weight_sum = weight_tensor.sum().clamp_min(1e-8)
    return (embeddings * weight_tensor).sum(dim=0, keepdim=True) / weight_sum


def load_scene_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not REQUIRED_KEYS.issubset(data):
        return None
    return data


def object_context(obj):
    attrs = obj.get("attributes", {})
    state = ", ".join(attrs.get("state", [])) or "상태없음"
    ppe = ", ".join(attrs.get("ppe", []))
    count = obj.get("count")
    count_text = f"; 수량={count}" if isinstance(count, int) and count >= 2 else ""
    ppe_text = f"; 보호구={ppe}" if ppe else ""
    location = obj.get("location", "")
    return (
        f"객체={obj.get('label', '')}; 상태={state}{ppe_text}"
        f"{count_text}; 위치={location}"
    )


def relation_text(rel, object_by_id):
    sub = object_by_id.get(rel.get("sub_id", ""), {})
    obj = object_by_id.get(rel.get("obj_id", ""), {})
    sub_label = sub.get("label", rel.get("sub_id", "unknown"))
    obj_label = obj.get("label", rel.get("obj_id", "unknown"))
    sub_state = ", ".join(sub.get("attributes", {}).get("state", []))
    obj_state = ", ".join(obj.get("attributes", {}).get("state", []))
    predicate = rel.get("predicate", "")
    category = rel.get("category", "")

    return (
        f"{sub_label}[{sub_state}] -- {category}:{predicate} --> "
        f"{obj_label}[{obj_state}]"
    )


def hazard_text(hazard, object_by_id):
    labels = [
        object_by_id.get(obj_id, {}).get("label", obj_id)
        for obj_id in hazard.get("related_object_ids", [])
    ]
    return f"위험={hazard.get('hazard', '')}; 관련객체={', '.join(labels)}"


def category_relation_embedding(relationships, object_by_id, category):
    texts = []
    weights = []

    for rel in relationships:
        if rel.get("category") != category:
            continue
        score = rel.get("score", 0.0)
        if not isinstance(score, (int, float)) or score < 0.5:
            continue
        texts.append(relation_text(rel, object_by_id))
        weights.append(float(score) * CATEGORY_WEIGHTS.get(category, 1.0))

    return weighted_mean(encode_texts(texts), weights)


def process_scene_template(template_dict):
    scene_text = template_dict.get("scene_description", "")
    scene_embedding = encode_texts([scene_text])

    objects = template_dict.get("objects", [])
    relationships = template_dict.get("relationships", [])
    hazards = template_dict.get("hazards", [])
    object_by_id = {obj.get("id", ""): obj for obj in objects}

    object_texts = [object_context(obj) for obj in objects]
    object_weights = [
        max(1.0, float(obj.get("count", 1)) ** 0.5)
        if isinstance(obj.get("count", 1), int)
        else 1.0
        for obj in objects
    ]
    object_embedding = weighted_mean(encode_texts(object_texts), object_weights)

    relation_embeddings = [
        category_relation_embedding(relationships, object_by_id, category)
        for category in CATEGORY_ORDER
    ]

    hazard_texts = [hazard_text(hazard, object_by_id) for hazard in hazards]
    hazard_embedding = weighted_mean(encode_texts(hazard_texts))

    graph_embedding = torch.cat(
        [object_embedding, *relation_embeddings, hazard_embedding],
        dim=1,
    )

    scene_part = F.normalize(scene_embedding, p=2, dim=1) * SCENE_WEIGHT
    graph_part = F.normalize(graph_embedding, p=2, dim=1) * GRAPH_WEIGHT
    combined_embedding = torch.cat([scene_part, graph_part], dim=1)

    return scene_embedding, graph_embedding, combined_embedding


def embed_json_file(json_path):
    template_data = load_scene_json(json_path)
    if template_data is None:
        raise ValueError(f"scene JSON이 아닙니다: {json_path}")
    return process_scene_template(template_data)


def cosine_similarity_matrix(embeddings):
    normalized = F.normalize(embeddings, p=2, dim=1)
    return torch.mm(normalized, normalized.t())


def print_similarity_table(title, file_names, sim_matrix):
    width = 12
    print(f"\n=== {title} ===")
    header = "".ljust(width) + " ".join(name[:10].ljust(width) for name in file_names)
    print(header)
    for i, name in enumerate(file_names):
        row = [
            f"{sim_matrix[i, j].item():.4f}".ljust(width)
            for j in range(len(file_names))
        ]
        print(name[:10].ljust(width) + " ".join(row))


def top_pairs(file_names, sim_matrix, k=5):
    pairs = []
    n = len(file_names)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((file_names[i], file_names[j], sim_matrix[i, j].item()))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]


def find_scene_json_files(input_dir, output_path):
    json_files = []
    for json_path in sorted(input_dir.glob("*.json")):
        if json_path.resolve() == output_path.resolve():
            continue
        if json_path.name.startswith("_"):
            continue
        if load_scene_json(json_path) is None:
            continue
        json_files.append(json_path)
    return json_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_path = args.output_path or (input_dir / "similarity_matrix.json")
    json_files = find_scene_json_files(input_dir, output_path)

    if not json_files:
        raise FileNotFoundError(f"scene JSON 파일을 찾지 못했습니다: {input_dir.resolve()}")

    print(f"총 {len(json_files)}개 파일 임베딩을 진행합니다.")
    scene_embeddings = []
    graph_embeddings = []
    combined_embeddings = []
    file_names = []

    for json_file in json_files:
        scene_emb, graph_emb, combined_emb = embed_json_file(json_file)
        scene_embeddings.append(scene_emb)
        graph_embeddings.append(graph_emb)
        combined_embeddings.append(combined_emb)
        file_names.append(json_file.stem)
        print(
            f"- 임베딩 완료: {json_file.name} "
            f"scene={tuple(scene_emb.shape)} graph={tuple(graph_emb.shape)} "
            f"combined={tuple(combined_emb.shape)}"
        )

    scene_tensor = torch.cat(scene_embeddings, dim=0)
    graph_tensor = torch.cat(graph_embeddings, dim=0)
    combined_tensor = torch.cat(combined_embeddings, dim=0)

    scene_matrix = cosine_similarity_matrix(scene_tensor)
    graph_matrix = cosine_similarity_matrix(graph_tensor)
    combined_matrix = cosine_similarity_matrix(combined_tensor)

    print_similarity_table("Combined Similarity Matrix", file_names, combined_matrix)

    best_pairs = top_pairs(
        file_names,
        combined_matrix,
        k=min(10, len(file_names) * (len(file_names) - 1) // 2),
    )
    print("\n=== Top Similar Pairs (combined) ===")
    for a, b, score in best_pairs:
        print(f"{a} <-> {b}: {score:.4f}")

    result = {
        "model": MODEL_NAME,
        "strategy": {
            "scene": "scene_description KoSBERT embedding",
            "graph": "object summary + category-wise weighted relationship embeddings + hazard embedding",
            "relation_weight": "score * category_weight, score < 0.5 filtered out",
            "category_weights": CATEGORY_WEIGHTS,
            "fusion": {
                "scene_weight": SCENE_WEIGHT,
                "graph_weight": GRAPH_WEIGHT,
            },
        },
        "files": file_names,
        "similarity_matrices": {
            "scene": scene_matrix.tolist(),
            "graph": graph_matrix.tolist(),
            "combined": combined_matrix.tolist(),
        },
        "top_pairs": [
            {"file_a": a, "file_b": b, "score": score} for a, b, score in best_pairs
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n유사도 결과 저장 완료: {output_path}")


if __name__ == "__main__":
    main()

from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDINGS_PATH = Path("vector_store/embeddings.npy")
METADATA_PATH = Path("vector_store/metadata.json")

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# Đọc embeddings đã tạo trước đó
def load_embeddings() -> np.ndarray:
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file embeddings: {EMBEDDINGS_PATH}"
        )

    embeddings = np.load(EMBEDDINGS_PATH)

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings phải là ma trận 2 chiều."
        )

    return embeddings.astype(np.float32)


# Đọc metadata tương ứng với từng vector
def load_metadata() -> list[dict]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file metadata: {METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    if not isinstance(metadata, list):
        raise ValueError(
            "metadata.json phải chứa một danh sách."
        )

    return metadata


# Tải model embedding
def load_model() -> SentenceTransformer:
    print(f"Đang tải model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    return model


# Tìm các chunk liên quan nhất
def retrieve(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    metadata: list[dict],
    top_k: int = 5,
) -> list[dict]:
    if not query.strip():
        return []

    if top_k <= 0:
        raise ValueError("top_k phải lớn hơn 0.")

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Số lượng embeddings và metadata không khớp: "
            f"{len(embeddings)} embeddings, "
            f"{len(metadata)} metadata."
        )

    # Tạo embedding cho câu hỏi.
    # normalize_embeddings=True giúp tích vô hướng
    # tương đương với cosine similarity.
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = query_embedding.astype(np.float32)

    # Tính điểm tương đồng giữa query và tất cả chunk.
    scores = embeddings @ query_embedding

    # Không lấy quá số lượng chunk hiện có.
    top_k = min(top_k, len(metadata))

    # Lấy index của các kết quả có điểm cao nhất.
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, index in enumerate(top_indices, start=1):
        result = metadata[index].copy()

        result["rank"] = rank
        result["score"] = float(scores[index])

        results.append(result)

    return results


# In kết quả ra terminal để kiểm tra
def print_results(
    query: str,
    results: list[dict],
) -> None:
    print("\n" + "=" * 80)
    print("Câu hỏi:")
    print(query)
    print("=" * 80)

    if not results:
        print("Không tìm thấy kết quả.")
        return

    for result in results:
        print(
            f"\nTop {result['rank']} "
            f"- Score: {result['score']:.4f}"
        )

        print(
            f"Chunk ID: {result.get('chunk_id', 'Không có')}"
        )

        print(
            f"Tài liệu: {result.get('file_name', 'Không có')}"
        )

        print(
            f"Trang: {result.get('page', 'Không có')}"
        )

        print("\nNội dung:")
        print(result.get("text", ""))

        print("-" * 80)


def main() -> None:
    model = load_model()
    embeddings = load_embeddings()
    metadata = load_metadata()

    print(f"Đã tải {len(metadata)} chunk.")

    # Có thể thay câu hỏi này để test.
    query = input("\nNhập câu hỏi: ").strip()

    results = retrieve(
        query=query,
        model=model,
        embeddings=embeddings,
        metadata=metadata,
        top_k=5,
    )

    print_results(
        query=query,
        results=results,
    )


if __name__ == "__main__":
    main()
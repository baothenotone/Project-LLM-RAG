from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer


# Đường dẫn đến vector store.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTOR_FOLDER = PROJECT_ROOT / "vector_store"
EMBEDDINGS_FILE = VECTOR_FOLDER / "embeddings.npy"
METADATA_FILE = VECTOR_FOLDER / "metadata.json"


# Mô hình dùng để embedding tài liệu và câu hỏi.
MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# Đọc embeddings đã tạo.
@lru_cache(maxsize=1)
def load_embeddings():
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {EMBEDDINGS_FILE}"
        )

    embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=False)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings phải là ma trận 2 chiều: {embeddings.shape}"
        )

    embeddings = embeddings.astype(np.float32, copy=False)

    print("Đã đọc embeddings:", embeddings.shape)

    return embeddings


# Đọc metadata tương ứng với embeddings.
@lru_cache(maxsize=1)
def load_metadata():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {METADATA_FILE}"
        )

    try:
        with METADATA_FILE.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"File metadata.json không hợp lệ: {error}"
        ) from error

    if not isinstance(metadata, list):
        raise ValueError(
            "Metadata phải được lưu dưới dạng danh sách."
        )

    print("Đã đọc metadata:", len(metadata))

    return metadata


# Tải mô hình đã dùng để embedding tài liệu.
@lru_cache(maxsize=1)
def load_embedding_model():
    print("Đang tải mô hình embedding...")
    print("Tên mô hình:", MODEL_NAME)

    return SentenceTransformer(MODEL_NAME)


# Tạo embedding cho câu hỏi.
def embed_question(model, question):
    question = question.strip()

    if not question:
        raise ValueError(
            "Câu hỏi không được để trống."
        )

    question_embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return question_embedding.astype(np.float32, copy=False)


# Kiểm tra embeddings và metadata có khớp nhau không.
def check_vector_store(embeddings, metadata, question_embedding):
    if len(embeddings) != len(metadata):
        raise ValueError(
            "Số embeddings và metadata không khớp: "
            f"{len(embeddings)} embeddings, "
            f"{len(metadata)} metadata"
        )

    if len(metadata) == 0:
        raise ValueError(
            "Vector store không có dữ liệu."
        )

    if embeddings.shape[1] != question_embedding.shape[0]:
        raise ValueError(
            "Số chiều embedding không khớp: "
            f"{embeddings.shape[1]} và "
            f"{question_embedding.shape[0]}"
        )


# Chuẩn hóa embeddings để tính cosine similarity.
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError(
            "Vector store chứa embedding có độ dài bằng 0."
        )

    return embeddings / norms


# Tìm các chunk có nội dung gần câu hỏi nhất.
def retrieve(question, top_k=5):
    if not isinstance(top_k, int):
        raise TypeError(
            "top_k phải là số nguyên."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k phải lớn hơn 0."
        )

    embeddings = load_embeddings()
    metadata = load_metadata()
    model = load_embedding_model()

    question_embedding = embed_question(model, question)

    check_vector_store(
        embeddings,
        metadata,
        question_embedding,
    )

    embeddings = normalize_embeddings(embeddings)

    similarity_scores = embeddings @ question_embedding

    top_k = min(top_k, len(metadata))

    sorted_indices = np.argsort(similarity_scores)[::-1]
    top_indices = sorted_indices[:top_k]

    results = []

    for index in top_indices:
        index = int(index)

        if not isinstance(metadata[index], dict):
            raise ValueError(
                "Mỗi phần tử metadata phải là một object."
            )

        item = metadata[index].copy()
        item["score"] = float(similarity_scores[index])

        results.append(item)

    return results


# Hiển thị kết quả truy xuất.
def print_results(results):
    if not results:
        print("Không tìm thấy kết quả.")
        return

    for rank, item in enumerate(results, start=1):
        print("=" * 80)
        print("TOP:", rank)
        print("Score:", round(item["score"], 4))
        print("File:", item.get("file_name", ""))
        print("Pages:", item.get("pages", []))
        print("Chunk ID:", item.get("chunk_id", ""))
        print("Text:")
        print(item.get("content") or item.get("embedding_text") or "")


# Chạy thử chức năng truy xuất.
def main():
    question = input("Nhập câu hỏi: ").strip()

    if not question:
        print("Câu hỏi không được để trống.")
        return

    try:
        results = retrieve(question, top_k=5)
        print_results(results)

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        OSError,
        RuntimeError,
    ) as error:
        print("Lỗi:", error)


if __name__ == "__main__":
    main()
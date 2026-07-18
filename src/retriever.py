from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

VECTOR_FOLDER = Path("vector_store")
EMBEDDINGS_FILE = VECTOR_FOLDER / "embeddings.npy"
METADATA_FILE = VECTOR_FOLDER / "metadata.json"

MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Đọc embeddings đã tạo.
def load_embeddings():
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {EMBEDDINGS_FILE}"
        )

    embeddings = np.load(EMBEDDINGS_FILE)

    print("Đã đọc embeddings:", embeddings.shape)

    return embeddings

# Đọc metadata tương ứng với embeddings.
def load_metadata():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {METADATA_FILE}"
        )

    with METADATA_FILE.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    print("Đã đọc metadata:", len(metadata))

    return metadata

# Tải mô hình đã dùng để embedding tài liệu.
def load_embedding_model():
    print("Đang tải mô hình embedding...")
    print("Tên mô hình:", MODEL_NAME)

    return SentenceTransformer(MODEL_NAME)

# Tạo embedding cho câu hỏi.
def embed_question(model, question):
    return model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

# Kiểm tra vector và metadata có khớp nhau không.
def check_vector_store(embeddings, metadata):
    if len(embeddings) != len(metadata):
        raise ValueError(
            "Số embeddings và metadata không khớp: "
            f"{len(embeddings)} embeddings, "
            f"{len(metadata)} metadata"
        )

# Tìm các chunk có nội dung gần câu hỏi nhất.
def retrieve(question, top_k=5):
    embeddings = load_embeddings()
    metadata = load_metadata()

    check_vector_store(embeddings, metadata)

    model = load_embedding_model()
    question_embedding = embed_question(model, question)

    similarity_scores = embeddings @ question_embedding
    sorted_indices = np.argsort(similarity_scores)[::-1]
    top_indices = sorted_indices[:top_k]

    results = []

    for index in top_indices:
        item = metadata[index].copy()
        item["score"] = float(similarity_scores[index])

        results.append(item)

    return results

# Hiển thị kết quả truy xuất.
def print_results(results):
    for rank, item in enumerate(results, start=1):
        print("=" * 80)
        print("TOP:", rank)
        print("Score:", item["score"])
        print("File:", item.get("file_name", ""))
        print("Page:", item.get("page", ""))
        print("Chunk ID:", item.get("chunk_id", ""))
        print("Text:")
        print(item.get("text", ""))

def main():
    question = input("Nhập câu hỏi: ").strip()

    if not question:
        print("Câu hỏi không được để trống.")
        return

    try:
        results = retrieve(question, top_k=5)
        print_results(results)

    except (FileNotFoundError, ValueError) as error:
        print("Lỗi:", error)

if __name__ == "__main__":
    main()
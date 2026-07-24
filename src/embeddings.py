from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer


# Đường dẫn đến các file
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "chunks.json"
VECTOR_FOLDER = PROJECT_ROOT / "vector_store"
EMBEDDINGS_FILE = VECTOR_FOLDER / "embeddings.npy"
METADATA_FILE = VECTOR_FOLDER / "metadata.json"


# Mô hình dùng để embedding văn bản tiếng Việt
MODEL_NAME = ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Đọc danh sách chunk
def load_chunks():
    print("Đang đọc file:", CHUNKS_FILE)

    # Kiểm tra file chunks.json có tồn tại hay không
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {CHUNKS_FILE}"
        )

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print("Số chunk đã đọc:", len(chunks))
    return chunks


# Loại bỏ những chunk không có nội dung embedding
def filter_valid_chunks(chunks):
    valid_chunks = []

    for chunk in chunks:
        text = str(
            chunk.get("embedding_text") or ""
        ).strip()

        # Bỏ qua chunk có embedding_text rỗng
        if text == "":
            continue

        # Chuẩn hóa lại nội dung embedding_text
        chunk["embedding_text"] = text

        valid_chunks.append(chunk)

    print("Số chunk hợp lệ:", len(valid_chunks))
    print("Số chunk bị bỏ qua:", len(chunks) - len(valid_chunks))

    return valid_chunks


# Tải mô hình embedding
def load_embedding_model():
    print("Đang tải mô hình...")
    print("Tên mô hình:", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)
    return model


# Lấy phần text trong từng chunk
# Lấy phần embedding_text trong từng chunk
def get_chunk_texts(chunks):
    texts = []

    for chunk in chunks:
        texts.append(chunk["embedding_text"])

    return texts


# Tạo embedding cho toàn bộ chunk
def create_embeddings(model, texts):
    print("Đang tạo embedding...")
    print("Số lượng đoạn văn bản:", len(texts))

    embeddings = model.encode(texts,batch_size=32,show_progress_bar=True,convert_to_numpy=True,normalize_embeddings=True)
    return embeddings


# Tạo embedding cho câu hỏi của người dùng
def create_query_embedding(model, query):
    query = query.strip()

    # Không cho phép query rỗng
    if query == "":
        raise ValueError("Câu hỏi không được để trống")

    query_embedding = model.encode(query,convert_to_numpy=True,normalize_embeddings=True)
    return query_embedding


# Tạo metadata tương ứng với từng vector embedding
# Tạo metadata tương ứng với từng vector embedding
def create_metadata(chunks):
    metadata_list = []

    for vector_index, chunk in enumerate(chunks):
        source_file = chunk.get("source_file")

        metadata = {
            "vector_index": vector_index,
            "chunk_id": chunk.get("chunk_id"),
            "doc_id": (
                Path(source_file).stem
                if source_file
                else None
            ),
            "file_name": source_file,
            "source": chunk.get("source_path"),
            "chunk_index": chunk.get(
                "original_chunk_index"
            ),
            "pages": chunk.get("pages"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "document_title": chunk.get(
                "document_title"
            ),
            "content": chunk.get("content"),
            "embedding_text": chunk.get(
                "embedding_text"
            ),
            "chapter": chunk.get("chapter"),
            "section": chunk.get("section"),
            "article": chunk.get("article"),
            "clause": chunk.get("clause"),
            "point": chunk.get("point"),
            "labels": chunk.get("labels")
        }

        metadata_list.append(metadata)

    return metadata_list


# Lưu vector và metadata
def save_vector_data(embeddings, metadata):
    # Kiểm tra số vector và metadata có bằng nhau hay không
    if len(embeddings) != len(metadata):
        raise ValueError(
            "Số embedding không bằng số metadata: "
            f"{len(embeddings)} != {len(metadata)}"
        )

    # Tạo thư mục vector_store nếu chưa tồn tại
    VECTOR_FOLDER.mkdir(parents=True,exist_ok=True)

    # Lưu ma trận embedding
    np.save(EMBEDDINGS_FILE,embeddings)

    # Lưu metadata của từng vector
    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata,file,ensure_ascii=False,indent=2)

    print("Đã lưu vector tại:", EMBEDDINGS_FILE)
    print("Đã lưu metadata tại:", METADATA_FILE)


def main():
    chunks = load_chunks()

    if len(chunks) == 0:
        print("Không có chunk để embedding")
        return

    # Dùng cùng một danh sách chunk cho embedding và metadata
    valid_chunks = filter_valid_chunks(chunks)

    if len(valid_chunks) == 0:
        print("Không có chunk hợp lệ để embedding")
        return

    texts = get_chunk_texts(valid_chunks)
    model = load_embedding_model()
    embeddings = create_embeddings(model,texts)
    metadata = create_metadata(valid_chunks)
    save_vector_data(embeddings,metadata)

    print("Hoàn thành tạo vector store")
    print("Số embedding:", len(embeddings))
    print("Số metadata:", len(metadata))


if __name__ == "__main__":
    main()
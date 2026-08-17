import json
import logging
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "chunks.json"
VECTOR_FOLDER = PROJECT_ROOT / "vector_store"
FAISS_INDEX_FILE = VECTOR_FOLDER / "faiss_index.bin"
METADATA_FILE = VECTOR_FOLDER / "metadata.json"

MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

def build_vector_store():
    # 1. Đọc dữ liệu
    if not CHUNKS_FILE.exists():
        logging.error(f"Không tìm thấy file: {CHUNKS_FILE}")
        return

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        raw_chunks = json.load(file)

    # Lọc lấy các chunk hợp lệ (có nội dung text)
    valid_chunks = []
    texts_to_embed = []
    for chunk in raw_chunks:
        text = str(chunk.get("embedding_text", "")).strip()
        if text != "":
            valid_chunks.append(chunk)
            texts_to_embed.append(text)

    if not valid_chunks:
        logging.warning("Không có chunk hợp lệ để tạo embedding.")
        return

    # 2. Tạo Embeddings
    logging.info(f"Đang tải mô hình {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    logging.info(f"Đang mã hóa (encode) {len(texts_to_embed)} đoạn văn bản...")
    # Bật normalize_embeddings=True để có thể dùng hàm Inner Product (tương đương Cosine) trong FAISS
    embeddings = model.encode(
        texts_to_embed, 
        batch_size=32, 
        show_progress_bar=True, 
        convert_to_numpy=True, 
        normalize_embeddings=True
    )

    # 3. Khởi tạo và lưu FAISS Index
    VECTOR_FOLDER.mkdir(parents=True, exist_ok=True)
    
    dimension = embeddings.shape[1] # Số chiều của vector, ví dụ: 768
    index = faiss.IndexFlatIP(dimension) # IP = Inner Product
    index.add(embeddings)
    
    faiss.write_index(index, str(FAISS_INDEX_FILE))

    # 4. Lưu Metadata
    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(valid_chunks, file, ensure_ascii=False, indent=2)

    logging.info(f"Đã tạo xong FAISS Index tại: {FAISS_INDEX_FILE}")
    logging.info(f"Đã lưu Metadata tại: {METADATA_FILE}")

if __name__ == "__main__":
    build_vector_store()
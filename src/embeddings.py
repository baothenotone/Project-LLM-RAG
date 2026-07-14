from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

#Đường dẫn đến các file 
CHUNKS_FILE = Path("data/chunks/chunks.json")
VECTOR_FOLDER = Path("vector_store")
EMBEDDINGS_FILE = VECTOR_FOLDER / "embeddings.npy"
METADATA_FILE = VECTOR_FOLDER / "metadata.json" 

#Mô hình dùng để embedding văn bản tiếng việt
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

#Đọc danh sách chunk 
def load_chunks():
    print("Đang đọc file:", CHUNKS_FILE)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)
    print("Số chunk đã đọc:", len(chunks))
    return chunks

#Tải mô hình embedding
def load_embedding_model():
    print("Đang tải mô hình...")
    print("Tên mô hình:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    return model

#Lấy phần text trong từng chunk
def get_chunk_texts(chunks):
    texts = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        #chỉ thêm những đoạn không rỗng
        if text!="":
            texts.append(text)
    return texts

# Tạo embedding cho toàn bộ chunk
def create_embedding(model, texts):
    print("Đang tạo embedding...")
    print("Số lượng đoạn văn bản:", len(texts))
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy = True, normalize_embeddings=True)
    return embeddings

# Tạo metadata tương ứng với từng vector embedding
def create_metadata(chunks):
    metadata_list = []
    for vector_index, chunk in enumerate(chunks):
        metadata = {
              "vector_index": vector_index,
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "file_name": chunk["file_name"],
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "source": chunk["source"]
        }
        metadata_list.append(metadata)
    return metadata_list

# Lưu vector và metadata
def save_vector_data(embeddings, metadata):
    VECTOR_FOLDER.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, embeddings)
    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("Đã lưu vector tại:", EMBEDDINGS_FILE)
    print("Đã lưu metadata tại:", METADATA_FILE)
def main():
    chunks = load_chunks()
    if len(chunks)==0:
        print("Không có chunk để embedding")
        return
    texts = get_chunk_texts(chunks)
    model = load_embedding_model()
    embeddings = create_embedding(model, texts)
    meatadata = create_metadata(chunks)
    save_vector_data(embeddings, meatadata)
if __name__ == "__main__":
    main()
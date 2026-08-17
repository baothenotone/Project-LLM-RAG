import json
import logging
import os
import re
import unicodedata
import faiss
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from query_expansion import expand_query

logging.basicConfig(level=logging.INFO, format="%(message)s")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

def load_retriever_resources():
    root = Path(__file__).resolve().parent.parent
    metadata_file = root / "vector_store" / "metadata.json"
    faiss_file = root / "vector_store" / "faiss_index.bin"
    
    if not faiss_file.exists() or not metadata_file.exists():
        raise FileNotFoundError("Chưa tìm thấy Database. Hãy chạy file embeddings.py trước.")
        
    # Đọc Metadata
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    # Đọc FAISS Index
    faiss_index = faiss.read_index(str(faiss_file))
    
    # Chuẩn bị dữ liệu cho BM25 (Từ khóa)
    search_texts = []
    tokenized_texts = []
    
    for item in metadata:
        title = str(item.get("document_title") or "")
        headings = str(item.get("headings") or "")
        content = str(item.get("embedding_text") or "")
        
        # Gộp các trường lại thành một văn bản dài để tìm kiếm
        full_text = f"{title}\n{headings}\n{content}".strip()
        search_texts.append(full_text)
        
        # Tokenize văn bản cho BM25
        normalized_text = unicodedata.normalize("NFC", full_text).lower()
        cleaned_text = re.sub(r"[^\w]+", " ", normalized_text, flags=re.UNICODE).strip()
        tokenized_texts.append(cleaned_text.split())
        
    bm25_model = BM25Okapi(tokenized_texts)
    
    # Load các Model AI
    logging.info("Đang tải mô hình Bi-Encoder...")
    embed_model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")
    
    reranker_model = None
    enable_reranker = os.getenv("ENABLE_RERANKER", "true").lower() in {"true", "1"}
    if enable_reranker:
        logging.info("Đang tải mô hình CrossEncoder (Reranker)...")
        reranker_name = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        reranker_model = CrossEncoder(reranker_name)
    
    return metadata, faiss_index, search_texts, bm25_model, embed_model, reranker_model

def retrieve(question, resources, top_k=5, pool_size=50, rrf_k=60):
    metadata, faiss_index, search_texts, bm25_model, embed_model, reranker_model = resources
    
    # 1. Mở rộng câu hỏi
    question = question.strip()
    if len(question) > 0:
        question = question[0].upper() + question[1:]
    query_variants = expand_query(question)
    
    if not query_variants:
        return []

    # 2. DENSE SEARCH (Tìm theo Ngữ nghĩa qua FAISS)
    q_embs = embed_model.encode(query_variants, convert_to_numpy=True, normalize_embeddings=True)
    if q_embs.ndim == 1: 
        q_embs = q_embs.reshape(1, -1)
    
    # D: Distance (Điểm số), I: Index (Vị trí tài liệu)
    distances, indices = faiss_index.search(q_embs, pool_size)
    
    dense_scores_dict = {}
    for q_idx in range(len(query_variants)):
        for rank_idx, doc_id in enumerate(indices[q_idx]):
            if doc_id >= 0: # Tránh trường hợp FAISS trả về -1 khi thiếu dữ liệu
                score = distances[q_idx][rank_idx]
                if doc_id not in dense_scores_dict or score > dense_scores_dict[doc_id]:
                    dense_scores_dict[doc_id] = score
                    
    dense_ranked_indices = sorted(dense_scores_dict.keys(), key=lambda x: dense_scores_dict[x], reverse=True)[:pool_size]

    # 3. SPARSE SEARCH (Tìm theo Từ khóa qua BM25)
    bm25_all_scores = []
    for q in query_variants:
        norm_q = unicodedata.normalize("NFC", q).lower()
        clean_q = re.sub(r"[^\w]+", " ", norm_q).strip().split()
        scores = bm25_model.get_scores(clean_q)
        bm25_all_scores.append(scores)
        
    # Lấy điểm cao nhất của mỗi văn bản trong tất cả các query variant
    bm25_max_scores = np.max(np.vstack(bm25_all_scores), axis=0)
    bm25_ranked_indices = np.argsort(bm25_max_scores)[::-1][:pool_size]

    # 4. RECIPROCAL RANK FUSION (RRF - Hợp nhất kết quả)
    fused_scores = {}
    
    for rank, doc_id in enumerate(dense_ranked_indices, start=1):
        fused_scores[int(doc_id)] = fused_scores.get(int(doc_id), 0.0) + (1.0 / (rrf_k + rank))
        
    for rank, doc_id in enumerate(bm25_ranked_indices, start=1):
        fused_scores[int(doc_id)] = fused_scores.get(int(doc_id), 0.0) + (1.0 / (rrf_k + rank))
            
    # Lấy top 60 ứng viên tốt nhất sau khi hợp nhất
    sorted_fused_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:60]
    candidate_indices = [item[0] for item in sorted_fused_items]

    # 5. RERANKER (Chấm điểm lại bằng CrossEncoder)
    if reranker_model:
        pairs = []
        for doc_idx in candidate_indices:
            for q in query_variants:
                pairs.append((q, search_texts[doc_idx]))
                
        # Chấm điểm toàn bộ các cặp (Câu hỏi - Tài liệu)
        rerank_scores = reranker_model.predict(pairs)
        rerank_scores = rerank_scores.reshape(len(candidate_indices), len(query_variants))
        
        # Lấy điểm cao nhất cho mỗi tài liệu
        best_rerank_scores = np.max(rerank_scores, axis=1)
        
        # --- ĐOẠN CODE ĐÃ SỬA LỖI ---
        # Tạo Dictionary map giữa doc_id và điểm rerank tương ứng
        rerank_dict = {doc_id: float(best_rerank_scores[i]) for i, doc_id in enumerate(candidate_indices)}
        
        # Sắp xếp lại danh sách dựa trên điểm số trong Dictionary
        candidate_indices.sort(key=lambda doc_id: rerank_dict[doc_id], reverse=True)
        final_scores = rerank_dict
        # -----------------------------
    else:
        final_scores = {doc_id: float(fused_scores[doc_id]) for doc_id in candidate_indices}

    # 6. Trả về kết quả
    results = []
    for doc_id in candidate_indices[:top_k]:
        item = metadata[doc_id].copy()
        item["score"] = final_scores[doc_id]
        results.append(item)
        
    return results
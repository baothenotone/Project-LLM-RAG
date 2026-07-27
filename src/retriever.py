import json
import os
import re
import unicodedata
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from query_expansion import QueryExpander

class HybridRetriever:
    def __init__(self):
        self.root = Path(__file__).resolve().parent.parent
        load_dotenv(self.root / ".env")
        
        self.embeddings_file = self.root / "vector_store" / "embeddings.npy"
        self.metadata_file = self.root / "vector_store" / "metadata.json"
        
        # Mô hình chuẩn tiếng Việt
        self.embed_model_name = "bkai-foundation-models/vietnamese-bi-encoder"
        self.reranker_name = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        self.enable_reranker = os.getenv("ENABLE_RERANKER", "true").lower() in {"true", "1"}
        
        self.pool_size = 50
        self.rrf_k = 60
        self.expander = QueryExpander()
        
        print("Đang khởi tạo hệ thống Retrieval...")
        self._load_data()
        self._load_models()
        print("Hoàn tất khởi tạo!")

    def _load_data(self):
        if not self.embeddings_file.exists() or not self.metadata_file.exists():
            raise FileNotFoundError("Không tìm thấy database. Hãy chạy lại embeddings.py trước.")
            
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        emb = np.load(self.embeddings_file)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        self.embeddings = emb / norms
        
        self.search_texts = [self._build_search_text(m) for m in self.metadata]
        self.bm25 = BM25Okapi([self._tokenize(t) for t in self.search_texts])

    def _load_models(self):
        self.embed_model = SentenceTransformer(self.embed_model_name)
        self.reranker = CrossEncoder(self.reranker_name) if self.enable_reranker else None

    def _normalize(self, text):
        text = unicodedata.normalize("NFC", str(text or "")).lower()
        return re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()

    def _tokenize(self, text):
        return self._normalize(text).split()

    def _build_search_text(self, item):
        content = str(item.get("content") or item.get("embedding_text") or "").strip()
        fields = [item.get("document_title"), item.get("headings"), content]
        return "\n".join([str(f) for f in fields if f])

    def _rrf(self, ranked_lists):
        fused = {}
        for ranked_indices in ranked_lists:
            for rank, idx in enumerate(ranked_indices, 1):
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, question, top_k=5):
        # 1. Expand query và tự động viết hoa chữ cái đầu
        question = question.strip()
        question = question[0].upper() + question[1:] if question else ""
        query_variants = self.expander.expand(question)
        
        if not query_variants:
            return []

        # 2. Dense Search
        q_embs = self.embed_model.encode(list(query_variants), convert_to_numpy=True, normalize_embeddings=True)
        if q_embs.ndim == 1: q_embs = q_embs.reshape(1, -1)
        
        dense_scores = np.max(self.embeddings @ q_embs.T, axis=1)
        dense_indices = np.argsort(dense_scores)[::-1][:self.pool_size]

        # 3. Sparse Search (BM25)
        bm25_matrix = np.vstack([self.bm25.get_scores(self._tokenize(q)) for q in query_variants]).T
        bm25_scores = np.max(bm25_matrix, axis=1)
        bm25_indices = np.argsort(bm25_scores)[::-1][:self.pool_size]

        # 4. RRF
        ranked_items = self._rrf([dense_indices, bm25_indices])
        candidate_indices = [idx for idx, _ in ranked_items[:60]]

        # 5. Reranker
        if self.reranker:
            pairs = [(q, self.search_texts[i]) for i in candidate_indices for q in query_variants]
            scores = self.reranker.predict(pairs).reshape(len(candidate_indices), len(query_variants))
            best_scores = np.max(scores, axis=1)
            candidate_indices = sorted(candidate_indices, key=lambda i: best_scores[candidate_indices.index(i)], reverse=True)

        # 6. Format kết quả
        results = []
        for i, idx in enumerate(candidate_indices[:top_k]):
            item = self.metadata[idx].copy()
            item["score"] = float(best_scores[candidate_indices.index(idx)]) if self.reranker else float(ranked_items[i][1])
            results.append(item)
            
        return results 
    # Hiển thị kết quả truy xuất.
def print_results(results):
    if not results:
        print("Không tìm thấy kết quả phù hợp.")
        return

    for rank, item in enumerate(results, start=1):
        content = (
            item.get("content")
            or item.get("text")
            or item.get("embedding_text")
            or ""
        )

        source_file = (
            item.get("source_file")
            or item.get("file_name")
            or "Không xác định"
        )

        pages = (
            item.get("pages")
            or item.get("page")
            or "Không xác định"
        )

        print("\n" + "=" * 80)
        print("TOP:", rank)
        print("Score:", f"{item.get('score', 0):.4f}")
        print("File:", source_file)
        print("Trang:", pages)
        print("Chunk ID:", item.get("chunk_id", ""))
        print("Nội dung:")
        print(content)


# Chạy thử hệ thống truy xuất.
def main():
    try:
        retriever = HybridRetriever()
    except Exception as error:
        print("Không thể khởi tạo hệ thống:")
        print(error)
        return

    print("\nHệ thống Retrieval đã sẵn sàng.")
    print("Nhập 'exit' để kết thúc.\n")

    while True:
        try:
            question = input("Nhập câu hỏi: ").strip()

            if question.lower() == "exit":
                print("Đã kết thúc.")
                break

            if not question:
                print("Câu hỏi không được để trống.\n")
                continue

            results = retriever.retrieve(
                question=question,
                top_k=5,
            )

            print_results(results)
            print()

        except KeyboardInterrupt:
            print("\nĐã kết thúc.")
            break

        except Exception as error:
            print("Lỗi khi truy xuất:", error)
            print()


if __name__ == "__main__":
    main()
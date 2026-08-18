import json
import re
import unicodedata
from datetime import date
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DIR = PROJECT_ROOT / "vector_store"

EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

TOP_K = 5
SEARCH_POOL_SIZE = 30
RERANK_CANDIDATES = 20
RRF_K = 60
RELEVANCE_THRESHOLD = -2


# Chuẩn hóa văn bản thành token cho BM25
def tokenize(text):
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return text.strip().split()


# Kiểm tra truy vấn có khớp trực tiếp với tài liệu
def has_exact_match(question, text):
    query_tokens = tokenize(question)
    text_tokens = set(tokenize(text))

    return bool(query_tokens) and all(
        token in text_tokens
        for token in query_tokens
    )


# Kiểm tra văn bản sửa đổi đã có hiệu lực
def is_active_amendment(item):
    if item.get("document_type") != "quyet_dinh_sua_doi":
        return False

    try:
        return (
            date.fromisoformat(item.get("effective_date", ""))
            <= date.today()
        )
    except ValueError:
        return False


# Thêm văn bản sửa đổi khi kết quả chứa Điều cũ
def add_amendment_candidates(candidate_ids, metadata):
    scopes = {
        (
            metadata[doc_id].get("document_number"),
            metadata[doc_id].get("article"),
        )
        for doc_id in candidate_ids
        if metadata[doc_id].get("document_number")
        and metadata[doc_id].get("article")
    }

    result = list(candidate_ids)
    seen = set(result)

    for doc_id, item in enumerate(metadata):
        target = (
            item.get("amends_document"),
            item.get("amends_article"),
        )

        if (
            is_active_amendment(item)
            and target in scopes
            and doc_id not in seen
        ):
            result.append(doc_id)
            seen.add(doc_id)

    return result


# Nạp FAISS, BM25 và các mô hình retrieval
def load_retriever_resources():
    metadata_file = VECTOR_DIR / "metadata.json"
    faiss_file = VECTOR_DIR / "faiss_index.bin"

    if not metadata_file.exists() or not faiss_file.exists():
        raise FileNotFoundError(
            "Chưa tìm thấy Database. Hãy chạy embeddings.py trước."
        )

    with open(metadata_file, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    faiss_index = faiss.read_index(str(faiss_file))

    if faiss_index.ntotal != len(metadata):
        raise ValueError(
            "Số vector FAISS không khớp với metadata."
        )

    search_texts = []

    for item in metadata:
        title = item.get("document_title") or ""
        headings = item.get("headings") or []
        content = item.get("embedding_text") or ""

        if isinstance(headings, list):
            headings = "\n".join(headings)

        search_texts.append(
            f"{title}\n{headings}\n{content}".strip()
        )

    bm25 = BM25Okapi([
        tokenize(text)
        for text in search_texts
    ])

    embed_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    return (
        metadata,
        faiss_index,
        search_texts,
        bm25,
        embed_model,
        reranker,
    )


# Gộp thứ hạng Dense và BM25 bằng RRF
def reciprocal_rank_fusion(dense_ids, bm25_ids):
    scores = {}

    for ranked_ids in (dense_ids, bm25_ids):
        for rank, doc_id in enumerate(ranked_ids, start=1):
            doc_id = int(doc_id)

            scores[doc_id] = (
                scores.get(doc_id, 0)
                + 1 / (RRF_K + rank)
            )

    return [
        doc_id
        for doc_id, _ in sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


# Truy xuất các chunk phù hợp nhất với câu hỏi
def retrieve(question, resources, top_k=TOP_K):
    (
        metadata,
        faiss_index,
        search_texts,
        bm25,
        embed_model,
        reranker,
    ) = resources

    question = question.strip()

    if not question:
        return []

    pool_size = min(
        SEARCH_POOL_SIZE,
        len(metadata),
    )

    # Dense Search
    query_vector = embed_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    _, dense_ids = faiss_index.search(
        query_vector,
        pool_size,
    )

    dense_ids = [
        doc_id
        for doc_id in dense_ids[0]
        if doc_id >= 0
    ]

    # BM25 Search
    bm25_scores = bm25.get_scores(
        tokenize(question)
    )

    bm25_ids = np.argsort(
        bm25_scores
    )[::-1][:pool_size]

    # RRF
    candidate_ids = reciprocal_rank_fusion(
        dense_ids,
        bm25_ids,
    )[:RERANK_CANDIDATES]

    candidate_ids = add_amendment_candidates(
        candidate_ids,
        metadata,
    )

    if not candidate_ids:
        return []

    # CrossEncoder rerank
    pairs = [
        (
            question,
            search_texts[doc_id],
        )
        for doc_id in candidate_ids
    ]

    scores = reranker.predict(pairs)

    final_scores = {
        doc_id: float(score)
        for doc_id, score in zip(
            candidate_ids,
            scores,
        )
    }

    candidate_ids.sort(
        key=lambda doc_id: final_scores[doc_id],
        reverse=True,
    )

    # Relevance gate
    top_score = final_scores[candidate_ids[0]]

    exact_matches = [
        doc_id
        for doc_id in candidate_ids
        if has_exact_match(
            question,
            search_texts[doc_id],
        )
    ]

    if (
        top_score < RELEVANCE_THRESHOLD
        and not exact_matches
    ):
        return []

    # Ưu tiên exact match cho truy vấn ngắn
    if exact_matches:
        candidate_ids.sort(
            key=lambda doc_id: (
                doc_id not in exact_matches,
                -final_scores[doc_id],
            )
        )

    results = []

    for doc_id in candidate_ids[:top_k]:
        item = metadata[doc_id].copy()
        item["score"] = final_scores[doc_id]
        results.append(item)

    return results

# In kết quả retrieval ra terminal
def print_results(results):
    if not results:
        print("Không tìm thấy kết quả phù hợp.")
        return

    for rank, item in enumerate(results, start=1):
        content = (
            item.get("content")
            or item.get("embedding_text")
            or ""
        )

        print("\n" + "=" * 80)
        print("TOP:", rank)
        print("Score:", f"{item.get('score', 0):.4f}")
        print(
            "Văn bản:",
            item.get("document_number") or "Không xác định",
        )
        print(
            "Điều:",
            item.get("article") or "Không xác định",
        )
        print(
            "Trang:",
            item.get("pages") or "Không xác định",
        )
        print("Chunk ID:", item.get("chunk_id", ""))
        print("Nội dung:")
        print(content)


# Chạy thử retriever trên terminal

def main():
    try:
        resources = load_retriever_resources()

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
                break

            if not question:
                print("Câu hỏi không được để trống.\n")
                continue

            results = retrieve(
                question,
                resources,
            )

            print_results(results)

        except KeyboardInterrupt:
            break

        except Exception as error:
            print("Lỗi khi truy xuất:", error)


if __name__ == "__main__":
    main()
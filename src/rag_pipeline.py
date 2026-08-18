import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from citations import build_context, prepare_sources
from retriever import load_retriever_resources, retrieve


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# Khởi tạo Gemini và tài nguyên retrieval

def init_rag_system():
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL")

    if not api_key or not model_name:
        raise ValueError(
            "Thiếu biến môi trường GEMINI_API_KEY hoặc GEMINI_MODEL."
        )

    client = genai.Client(
        api_key=api_key
    )

    resources = load_retriever_resources()

    return client, model_name, resources


# Truy xuất tài liệu và tạo câu trả lời từ Gemini

def ask(
    question,
    client,
    model_name,
    resources,
    top_k=5,
):
    question = question.strip()

    if not question:
        return {
            "answer": "Vui lòng nhập câu hỏi.",
            "sources": [],
        }

    results = retrieve(
        question,
        resources,
        top_k=top_k,
    )

    sources = prepare_sources(results)

    if not sources:
        return {
            "answer": (
                "Không tìm thấy thông tin "
                "trong hệ thống tài liệu quy chế."
            ),
            "sources": [],
        }

    context = build_context(sources)
    source_count = len(sources)

    prompt = f"""Bạn là trợ lý giải đáp quy chế đào tạo của Trường Đại học Khoa học.

Chỉ trả lời dựa trên TÀI LIỆU THAM KHẢO bên dưới.

Quy tắc:
1. Không bịa đặt hoặc bổ sung thông tin ngoài tài liệu.
2. Nếu văn bản sửa đổi đã có hiệu lực và sửa đúng nội dung đang được hỏi, ưu tiên quy định trong văn bản sửa đổi.
3. Không xem văn bản sửa đổi là thay thế toàn bộ văn bản gốc; chỉ áp dụng trong phạm vi sửa đổi được ghi trong nguồn.
4. Nếu nguồn cũ và nguồn sửa đổi khác nhau, câu trả lời chính phải theo quy định hiện hành.
5. Trích dẫn [Nguồn n] ngay sau thông tin được sử dụng.
6. Chỉ được dùng các trích dẫn từ [Nguồn 1] đến [Nguồn {source_count}].
7. Nếu tài liệu không đủ thông tin, trả lời: "Tài liệu hiện tại không đề cập chi tiết về vấn đề này."
8. Trả lời rõ ràng, ngắn gọn và dễ hiểu.

CÂU HỎI:
{question}

TÀI LIỆU THAM KHẢO:
{context}

TRẢ LỜI:"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )

        answer = response.text.strip()

    except Exception as error:
        logging.error(
            f"Lỗi khi gọi Gemini: {error}"
        )

        answer = (
            "Hệ thống AI đang gặp lỗi, "
            "vui lòng thử lại sau."
        )

    return {
        "answer": answer,
        "sources": sources,
    }
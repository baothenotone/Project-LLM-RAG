import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from retriever import retrieve


# =========================================================
# CẤU HÌNH
# =========================================================

# Đọc các biến môi trường từ file .env.
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

# Số lần thử lại khi Gemini gặp lỗi tạm thời.
MAX_RETRIES = 3

# Thời gian chờ cơ bản cho cơ chế exponential backoff.
BASE_WAIT_TIME = 2

# Lưu Gemini client để không phải khởi tạo lại ở mỗi câu hỏi.
_gemini_client = None


# =========================================================
# XỬ LÝ GEMINI CLIENT
# =========================================================

def get_gemini_client():
    """
    Tạo Gemini client một lần và tái sử dụng trong toàn bộ chương trình.
    """
    global _gemini_client

    if not GEMINI_API_KEY:
        raise ValueError(
            "Chưa cấu hình GEMINI_API_KEY trong file .env"
        )

    if not GEMINI_MODEL:
        raise ValueError(
            "Chưa cấu hình GEMINI_MODEL trong file .env"
        )

    if _gemini_client is None:
        _gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

    return _gemini_client


# =========================================================
# CHUẨN HÓA METADATA
# =========================================================

def get_chunk_content(item):
    """
    Lấy nội dung văn bản của một chunk.

    Hàm hỗ trợ nhiều tên trường để tương thích với metadata
    được tạo từ các phiên bản preprocessing khác nhau.
    """
    content = (
        item.get("content")
        or item.get("text")
        or item.get("chunk_text")
        or item.get("raw_text")
        or item.get("embedding_text")
        or ""
    )

    return str(content).strip()


def get_file_name(item):
    """
    Lấy tên file nguồn của chunk.
    """
    file_name = (
        item.get("source_file")
        or item.get("file_name")
        or item.get("document_name")
        or item.get("source")
        or "Không xác định"
    )

    return str(file_name)


def format_pages(item):
    """
    Chuẩn hóa thông tin số trang thành chuỗi để hiển thị.

    Các dạng metadata được hỗ trợ:
    - pages: [2, 3]
    - page: 2
    - page_start: 2, page_end: 3
    """
    pages = item.get("pages")

    if isinstance(pages, list):
        valid_pages = [
            str(page)
            for page in pages
            if page not in (None, "")
        ]

        if valid_pages:
            return ", ".join(valid_pages)

    if pages not in (None, ""):
        return str(pages)

    page = item.get("page")

    if page not in (None, ""):
        return str(page)

    page_start = item.get("page_start")
    page_end = item.get("page_end")

    if (
        page_start not in (None, "")
        and page_end not in (None, "")
    ):
        if page_start == page_end:
            return str(page_start)

        return f"{page_start}-{page_end}"

    if page_start not in (None, ""):
        return str(page_start)

    if page_end not in (None, ""):
        return str(page_end)

    return "Không xác định"


def get_score(item):
    """
    Lấy similarity score của kết quả retrieval.

    Nếu score không tồn tại hoặc không chuyển được sang số,
    hàm trả về 0.0.
    """
    score = item.get("score", 0.0)

    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


# =========================================================
# XÂY DỰNG CONTEXT
# =========================================================

def build_context(results):
    """
    Chuyển danh sách chunk đã truy xuất thành context
    để đưa vào Gemini.
    """
    context_parts = []

    for rank, item in enumerate(results, start=1):
        file_name = get_file_name(item)
        pages = format_pages(item)
        content = get_chunk_content(item)

        if not content:
            continue

        context_part = (
            f"[Nguồn {rank}]\n"
            f"Tài liệu: {file_name}\n"
            f"Trang: {pages}\n"
            f"Nội dung:\n{content}"
        )

        context_parts.append(context_part)

    return "\n\n".join(context_parts)


# =========================================================
# XÂY DỰNG PROMPT
# =========================================================

def build_prompt(question, context):
    """
    Tạo prompt yêu cầu Gemini chỉ trả lời
    dựa trên các tài liệu được retrieval.
    """
    return f"""
Bạn là trợ lý hỏi đáp hỗ trợ tra cứu quy chế đào tạo.

Hãy tuân thủ nghiêm ngặt các yêu cầu sau:

1. Chỉ sử dụng thông tin có trong phần TÀI LIỆU THAM KHẢO.

2. Không sử dụng kiến thức bên ngoài tài liệu.

3. Không suy đoán hoặc tự bổ sung những thông tin
   không được tài liệu cung cấp.

4. Chỉ đưa ra kết luận khi thông tin trong tài liệu
   trực tiếp hỗ trợ cho kết luận đó.

5. Khi sử dụng thông tin từ một nguồn, phải ghi trích dẫn
   theo định dạng [Nguồn n].

6. Đặt trích dẫn ngay sau câu hoặc đoạn thông tin
   được lấy từ nguồn tương ứng.

7. Có thể sử dụng nhiều nguồn khi cần tổng hợp thông tin.

8. Không liệt kê nguồn không được sử dụng trong câu trả lời.

9. Trả lời rõ ràng, tự nhiên và ngắn gọn bằng tiếng Việt.

10. Giữ nguyên các mốc thời gian, con số, tên đơn vị,
    điều khoản và thuật ngữ quan trọng trong tài liệu.

11. Nếu các tài liệu được cung cấp không chứa thông tin
    trực tiếp trả lời câu hỏi, hãy trả lời đúng câu sau:

"Không tìm thấy thông tin phù hợp trong tài liệu."

CÂU HỎI:
{question}

TÀI LIỆU THAM KHẢO:
{context}

CÂU TRẢ LỜI:
""".strip()


# =========================================================
# XỬ LÝ LỖI GEMINI
# =========================================================

def is_retryable_error(error):
    """
    Kiểm tra lỗi có phải là lỗi tạm thời
    và có thể thử lại hay không.
    """
    error_message = str(error).upper()

    status_code = getattr(
        error,
        "status_code",
        None,
    )

    if status_code is None:
        status_code = getattr(
            error,
            "code",
            None,
        )

    try:
        if status_code is not None:
            status_code = int(status_code)
    except (TypeError, ValueError):
        status_code = None

    retryable_status_codes = {
        429,
        500,
        502,
        503,
        504,
    }

    retryable_markers = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "RESOURCE_EXHAUSTED",
        "INTERNAL",
        "BAD_GATEWAY",
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
        "TIMEOUT",
        "TIMED OUT",
        "SERVICE UNAVAILABLE",
    )

    if status_code in retryable_status_codes:
        return True

    return any(
        marker in error_message
        for marker in retryable_markers
    )


# =========================================================
# GENERATION
# =========================================================

def generate_answer(question, context):
    """
    Gửi câu hỏi và context đến Gemini để tạo câu trả lời.
    """
    client = get_gemini_client()

    prompt = build_prompt(
        question=question,
        context=context,
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=512,
                ),
            )

            answer = getattr(
                response,
                "text",
                None,
            )

            if not answer:
                return "Gemini không trả về nội dung."

            return answer.strip()

        except Exception as error:
            if not is_retryable_error(error):
                raise

            is_last_attempt = (
                attempt == MAX_RETRIES - 1
            )

            if is_last_attempt:
                raise RuntimeError(
                    "Gemini tạm thời chưa sẵn sàng "
                    "sau nhiều lần thử. "
                    "Vui lòng thử lại sau."
                ) from error

            wait_time = (
                BASE_WAIT_TIME * (2 ** attempt)
            )

            print(
                "Gemini tạm thời chưa sẵn sàng, "
                f"thử lại sau {wait_time} giây..."
            )

            time.sleep(wait_time)

    return "Gemini không trả về nội dung."


# =========================================================
# RAG PIPELINE
# =========================================================

def ask_question(question, top_k=5):
    """
    Thực hiện toàn bộ quy trình RAG:

    Câu hỏi
    -> Retrieval
    -> Context
    -> Gemini
    -> Câu trả lời và nguồn
    """
    question = str(question).strip()

    if not question:
        raise ValueError(
            "Câu hỏi không được để trống."
        )

    if not isinstance(top_k, int):
        raise TypeError(
            "top_k phải là số nguyên."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k phải lớn hơn 0."
        )

    results = retrieve(
        question,
        top_k=top_k,
    )

    if results is None:
        results = []

    # Chỉ giữ lại các chunk có nội dung.
    valid_results = [
        item
        for item in results
        if isinstance(item, dict)
        and get_chunk_content(item)
    ]

    if not valid_results:
        return {
            "answer": (
                "Không tìm thấy thông tin phù hợp "
                "trong tài liệu."
            ),
            "sources": [],
        }

    context = build_context(
        valid_results
    )

    if not context:
        return {
            "answer": (
                "Không tìm thấy thông tin phù hợp "
                "trong tài liệu."
            ),
            "sources": [],
        }

    answer = generate_answer(
        question=question,
        context=context,
    )

    return {
        "answer": answer,
        "sources": valid_results,
    }


# =========================================================
# HIỂN THỊ KẾT QUẢ
# =========================================================

def print_sources(sources):
    """
    In các nguồn được Retriever trả về.
    """
    if not sources:
        return

    print("\nNguồn tham khảo:")

    for rank, item in enumerate(
        sources,
        start=1,
    ):
        file_name = get_file_name(item)
        pages = format_pages(item)
        score = get_score(item)

        print(
            f"{rank}. {file_name} - "
            f"trang {pages} - "
            f"score {score:.4f}"
        )


def print_retrieval_debug(sources):
    """
    In nội dung rút gọn của các kết quả retrieval.

    Hàm này dùng khi cần kiểm tra Retriever.
    Không được gọi mặc định trong main().
    """
    print("\n===== DEBUG RETRIEVAL =====")
    print(f"Số kết quả: {len(sources)}")

    for rank, item in enumerate(
        sources,
        start=1,
    ):
        content = get_chunk_content(item)
        preview = content[:300].replace(
            "\n",
            " ",
        )

        print(
            f"\nKết quả {rank}:"
        )
        print(
            f"File: {get_file_name(item)}"
        )
        print(
            f"Trang: {format_pages(item)}"
        )
        print(
            f"Score: {get_score(item):.4f}"
        )
        print(
            f"Nội dung: {preview}"
        )


# =========================================================
# CHƯƠNG TRÌNH TERMINAL
# =========================================================

def main():
    """
    Chạy hệ thống hỏi đáp trên terminal.
    """
    print("Hệ thống hỏi đáp đã sẵn sàng.")
    print("Nhập 'exit' để kết thúc.\n")

    while True:
        try:
            question = input(
                "Nhập câu hỏi: "
            ).strip()

        except (EOFError, KeyboardInterrupt):
            print("\nĐã kết thúc.")
            break

        if question.lower() in {
            "exit",
            "quit",
        }:
            print("Đã kết thúc.")
            break

        if not question:
            print(
                "Câu hỏi không được để trống.\n"
            )
            continue

        try:
            result = ask_question(
                question=question,
                top_k=5,
            )

            print("\nCâu trả lời:")
            print(result["answer"])

            if result["sources"]:
                print_sources(
                    result["sources"]
                )

            print()

        except Exception as error:
            print(
                f"Lỗi xử lý câu hỏi: {error}\n"
            )


if __name__ == "__main__":
    main()
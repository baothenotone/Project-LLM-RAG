"""
Mở rộng và viết lại câu hỏi (Query Rewriting / Expansion) bằng LLM.

Module này chỉ hỗ trợ bước Retrieval. Các truy vấn được tạo ra không
được dùng làm câu trả lời và không thay thế câu hỏi gốc của người dùng.
"""

from functools import lru_cache
from pathlib import Path
import json
import os
import unicodedata

from dotenv import load_dotenv
from google import genai
from google.genai import types


# =========================================================
# CẤU HÌNH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

ENABLE_QUERY_EXPANSION = (
    os.getenv("ENABLE_QUERY_EXPANSION", "true").strip().lower()
    not in {"0", "false", "no", "off"}
)

QUERY_EXPANSION_COUNT = 3


# =========================================================
# HÀM HỖ TRỢ
# =========================================================

def clean_query_text(text):
    """
    Chuẩn hóa Unicode và khoảng trắng nhưng vẫn giữ nguyên ý nghĩa.
    """
    text = unicodedata.normalize("NFC", str(text or ""))
    # Loại bỏ khoảng trắng thừa
    text = " ".join(text.split())
    return text.strip()


def query_comparison_key(text):
    """
    Tạo khóa dùng để loại các truy vấn trùng nhau.
    Khóa này chỉ dùng để so sánh. Câu hỏi thật không bị thay đổi.
    """
    text = clean_query_text(text).lower()
    # Chỉ giữ lại chữ và số để so sánh độ trùng lặp
    text = "".join(c if c.isalnum() else " " for c in text)
    return " ".join(text.split())


def unique_queries(queries, maximum=None):
    """
    Loại truy vấn rỗng và truy vấn trùng, giữ nguyên thứ tự ban đầu.
    """
    results = []
    seen = set()

    for query in queries:
        query = clean_query_text(query)
        key = query_comparison_key(query)

        if not key or key in seen:
            continue

        results.append(query)
        seen.add(key)

        if maximum is not None and len(results) >= maximum:
            break

    return tuple(results)


# =========================================================
# XÂY DỰNG PROMPT CHO LLM
# =========================================================

def build_expansion_prompt(question):
    """
    Tạo prompt yêu cầu Gemini đóng vai trò là Query Rewriter.
    LLM sẽ tự động xử lý mọi loại câu hỏi (định nghĩa, điều kiện, v.v.)
    và trả về các câu truy vấn mang văn phong của văn bản quy chế.
    """
    return f"""
Bạn là một chuyên gia tối ưu hóa truy vấn tìm kiếm cho hệ thống tra cứu văn bản quy chế đào tạo đại học.

Nhiệm vụ của bạn là chuyển đổi câu hỏi của người dùng thành đúng {QUERY_EXPANSION_COUNT} truy vấn tìm kiếm tối ưu nhất để khớp với cơ sở dữ liệu văn bản pháp lý.

Chiến lược viết lại:
1. Nếu là câu hỏi định nghĩa (Ví dụ: "Tín chỉ là gì?"), hãy viết lại thành câu khẳng định hoặc từ khóa cốt lõi (Ví dụ: "Định nghĩa tín chỉ", "Tín chỉ là").
2. Nếu là câu hỏi về điều kiện/quy trình (Ví dụ: "Bao nhiêu điểm thì bị đuổi học?"), hãy chuyển thành văn phong quy định (Ví dụ: "Quy định buộc thôi học", "Điều kiện cảnh báo học vụ").
3. Giữ nguyên các mốc thời gian, tên riêng, mã học phần nếu có.
4. KHÔNG trả lời câu hỏi. KHÔNG tự bịa thêm thông tin. Chỉ tạo câu truy vấn.

Chỉ trả về định dạng JSON chính xác như sau, không có markdown hay văn bản nào khác:
{{"queries": ["truy vấn 1", "truy vấn 2", "truy vấn 3"]}}

Câu hỏi của người dùng:
{question}
""".strip()


def parse_expansion_response(response_text):
    """
    Đọc danh sách truy vấn từ JSON do Gemini trả về.
    Có khả năng bóc tách JSON kể cả khi LLM sinh lỗi bọc trong Markdown.
    """
    response_text = str(response_text or "").strip()

    if not response_text:
        return ()

    candidates = [response_text]

    object_start = response_text.find("{")
    object_end = response_text.rfind("}")

    if 0 <= object_start < object_end:
        candidates.append(
            response_text[object_start:object_end + 1]
        )

    array_start = response_text.find("[")
    array_end = response_text.rfind("]")

    if 0 <= array_start < array_end:
        candidates.append(
            response_text[array_start:array_end + 1]
        )

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(data, dict):
            data = data.get("queries", [])

        if isinstance(data, list):
            return unique_queries(
                data,
                maximum=QUERY_EXPANSION_COUNT,
            )

    return ()


# =========================================================
# GEMINI QUERY EXPANSION
# =========================================================

@lru_cache(maxsize=1)
def get_query_expansion_client():
    """
    Tạo Gemini client một lần và tái sử dụng cho các câu hỏi sau.
    """
    if not GEMINI_API_KEY:
        raise ValueError(
            "Chưa cấu hình GEMINI_API_KEY để mở rộng câu hỏi."
        )

    if not GEMINI_MODEL:
        raise ValueError(
            "Chưa cấu hình GEMINI_MODEL để mở rộng câu hỏi."
        )

    return genai.Client(api_key=GEMINI_API_KEY)


@lru_cache(maxsize=256)
def expand_query(question):
    """
    Trả về câu hỏi gốc cùng các câu truy vấn được tối ưu bởi LLM.
    Khi Query Expansion bị tắt hoặc Gemini gặp lỗi, hàm trả về câu hỏi
    gốc để luồng Retrieval vẫn hoạt động bình thường.
    """
    question = clean_query_text(question)

    if not question:
        return ()

    base_queries = [question]

    if not ENABLE_QUERY_EXPANSION or not GEMINI_API_KEY or not GEMINI_MODEL:
        return unique_queries(base_queries)

    try:
        client = get_query_expansion_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=build_expansion_prompt(question),
            config=types.GenerateContentConfig(
                temperature=0.0, # Giữ ở 0.0 để kết quả sinh ra nhất quán
                max_output_tokens=256,
                response_mime_type="application/json",
            ),
        )

        expanded_queries = parse_expansion_response(
            getattr(response, "text", "")
        )

    except Exception as error:
        print(
            "Cảnh báo: không thể mở rộng câu hỏi. "
            "Hệ thống tạm dùng câu hỏi gốc."
        )
        print("Chi tiết:", error)
        expanded_queries = ()

    # Gộp câu hỏi gốc và các câu được sinh ra, sau đó loại bỏ trùng lặp
    return unique_queries(
        [*base_queries, *expanded_queries],
        maximum=QUERY_EXPANSION_COUNT + 1,
    )
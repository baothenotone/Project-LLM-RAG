import os
import time

from dotenv import load_dotenv
from google import genai

from retriever import retrieve


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")


# Tạo nội dung tham khảo từ các chunk truy xuất được.
def build_context(results):
    context_parts = []

    for rank, item in enumerate(results, start=1):
        file_name = item.get("file_name", "Không xác định")
        page = item.get("page", "Không xác định")
        text = item.get("text", "").strip()

        context_part = (
            f"[Nguồn {rank}]\n"
            f"Tài liệu: {file_name}\n"
            f"Trang: {page}\n"
            f"Nội dung: {text}"
        )
        context_parts.append(context_part)

    return "\n\n".join(context_parts)


# Tạo prompt để Gemini chỉ trả lời dựa trên tài liệu.
def build_prompt(question, context):
    return f"""
Bạn là trợ lý hỏi đáp về quy chế đào tạo.

Chỉ sử dụng thông tin trong phần TÀI LIỆU THAM KHẢO để trả lời.
Không tự bổ sung thông tin bên ngoài tài liệu.
Trả lời rõ ràng, ngắn gọn bằng tiếng Việt.
Nếu tài liệu không đủ thông tin, hãy nói:
"Không tìm thấy thông tin phù hợp trong tài liệu."

CÂU HỎI:
{question}

TÀI LIỆU THAM KHẢO:
{context}

CÂU TRẢ LỜI:
""".strip()


# Gọi Gemini và thử lại khi máy chủ đang quá tải.
def generate_answer(question, context):
    if not GEMINI_API_KEY:
        raise ValueError(
            "Chưa cấu hình GEMINI_API_KEY trong file gemini.env"
        )

    if not GEMINI_MODEL:
        raise ValueError(
            "Chưa cấu hình GEMINI_MODEL trong file gemini.env"
        )

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = build_prompt(question, context)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            if not response.text:
                return "Gemini không trả về nội dung."

            return response.text.strip()

        except Exception as error:
            error_message = str(error)

            is_overloaded = (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            )

            if not is_overloaded:
                raise

            if attempt == 2:
                raise RuntimeError(
                    "Gemini đang quá tải. Vui lòng thử lại sau."
                ) from error

            wait_time = 2 ** (attempt + 1)

            print(
                f"Gemini đang quá tải, thử lại sau "
                f"{wait_time} giây..."
            )

            time.sleep(wait_time)


# Thực hiện toàn bộ quy trình RAG cho một câu hỏi.
def ask_question(question, top_k=5):
    results = retrieve(question, top_k=top_k)

    if not results:
        return {
            "answer": "Không tìm thấy thông tin phù hợp trong tài liệu.",
            "sources": [],
        }

    context = build_context(results)
    answer = generate_answer(question, context)

    return {
        "answer": answer,
        "sources": results,
    }


def print_sources(sources):
    print("\nNguồn tham khảo:")

    for rank, item in enumerate(sources, start=1):
        print(
            f"{rank}. {item.get('file_name', '')} - "
            f"trang {item.get('page', '')} - "
            f"score {item.get('score', 0):.4f}"
        )


def main():
    print("Hệ thống hỏi đáp đã sẵn sàng.")
    print("Nhập 'exit' để kết thúc.\n")

    while True:
        question = input("Nhập câu hỏi: ").strip()

        if question.lower() == "exit":
            print("Đã kết thúc.")
            break

        if not question:
            print("Câu hỏi không được để trống.\n")
            continue

        try:
            result = ask_question(question, top_k=5)

            print("\nCâu trả lời:")
            print(result["answer"])

            if result["sources"]:
                print_sources(result["sources"])

            print()

        except Exception as error:
            print(f"Lỗi xử lý câu hỏi: {error}\n")


if __name__ == "__main__":
    main()
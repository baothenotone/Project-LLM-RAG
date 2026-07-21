import os

from dotenv import load_dotenv
from google import genai

from retriever import (
    load_embedding_model,
    load_vector_data,
    retrieve_top_k,
)

# Đọc các biến cấu hình trong file .env.
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

if not GEMINI_API_KEY:
    raise ValueError("Chưa cấu hình GEMINI_API_KEY trong file .env")
if not GEMINI_MODEL:
    raise ValueError("Chưa cấu hình GEMINI_MODEL trong file .env")


# Tạo kết nối đến Gemini API một lần khi chương trình bắt đầu.
client = genai.Client(api_key=GEMINI_API_KEY)


def build_context(results):
    """Ghép các chunk tìm được thành ngữ cảnh gửi cho LLM."""
    context_parts = []

    for position, result in enumerate(results, start=1):
        text = result.get("text", "").strip()

        if not text:
            continue

        context_parts.append(
            f"[Nguồn {position}]\n{text}"
        )

    return "\n\n".join(context_parts)


def generate_answer(query, context):
    """Gửi câu hỏi và ngữ cảnh đến Gemini để tạo câu trả lời."""
    prompt = f"""
Bạn là trợ lý tra cứu quy chế đào tạo.

Hãy trả lời câu hỏi chỉ dựa trên tài liệu được cung cấp.
Không tự thêm thông tin bên ngoài tài liệu.
Ghi nguồn theo dạng [Nguồn 1], [Nguồn 2].
Nếu tài liệu không đủ thông tin, hãy nói rằng chưa tìm thấy câu trả lời.

TÀI LIỆU:
{context}

CÂU HỎI:
{query}

CÂU TRẢ LỜI:
""".strip()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return response.text


def answer_question(query, model, embeddings, metadata):
    """Chạy toàn bộ quy trình truy xuất và sinh câu trả lời."""
    # Bước 1: tìm 3 chunk gần câu hỏi nhất.
    results = retrieve_top_k(
        query=query,
        model=model,
        embeddings=embeddings,
        metadata=metadata,
        top_k=3,
    )

    if not results:
        return "Không tìm thấy nội dung liên quan.", []

    # Bước 2: ghép nội dung các chunk thành context.
    context = build_context(results)

    if not context:
        return "Các chunk tìm được không có nội dung.", results

    # Bước 3: gửi context và câu hỏi cho Gemini.
    answer = generate_answer(
        query=query,
        context=context,
    )

    return answer, results


def display_sources(results):
    """Hiển thị tài liệu và trang của các chunk đã truy xuất."""
    print("\nNGUỒN THAM KHẢO")

    for position, result in enumerate(results, start=1):
        file_name = result.get(
            "file_name",
            result.get("source", "Không rõ"),
        )

        page = result.get(
            "page",
            result.get("page_number", "Không rõ"),
        )

        score = result.get("score", 0)

        print(
            f"[Nguồn {position}] "
            f"{file_name} - Trang {page} "
            f"- Score: {score:.4f}"
        )


def main():
    # Model embedding và vector store chỉ cần tải một lần.
    model = load_embedding_model()
    embeddings, metadata = load_vector_data()

    print("\nHệ thống hỏi đáp đã sẵn sàng.")
    print("Nhập 'exit' để kết thúc.")

    while True:
        query = input("\nNhập câu hỏi: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Đã kết thúc chương trình.")
            break

        if not query:
            print("Vui lòng nhập câu hỏi.")
            continue

        try:
            answer, sources = answer_question(
                query=query,
                model=model,
                embeddings=embeddings,
                metadata=metadata,
            )

            print("\nCÂU TRẢ LỜI")
            print(answer)

            if sources:
                display_sources(sources)

        except Exception as error:
            print(f"Lỗi xử lý câu hỏi: {error}")

if __name__ == "__main__":
    main()


    
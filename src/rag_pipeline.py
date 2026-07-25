import os
<<<<<<< Updated upstream

=======
import time
>>>>>>> Stashed changes
from dotenv import load_dotenv
from google import genai

<<<<<<< Updated upstream
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
=======
from citations import build_context, print_sources
from retriever import HybridRetriever

class RAGPipeline:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL")
        
        if not self.api_key or not self.model_name:
            raise ValueError("Thiếu GEMINI_API_KEY hoặc GEMINI_MODEL trong file .env")
            
        self.client = genai.Client(api_key=self.api_key)
        self.retriever = HybridRetriever()

    def _build_prompt(self, question, context):
        return f"""Bạn là trợ lý hỏi đáp hỗ trợ tra cứu quy chế đào tạo.
Chỉ sử dụng thông tin có trong phần TÀI LIỆU THAM KHẢO. Trả lời rõ ràng, tự nhiên bằng tiếng Việt.
Trích dẫn nguồn theo định dạng [Nguồn n] ngay sau thông tin được lấy.
Nếu không có thông tin, hãy nói: "Không tìm thấy thông tin phù hợp trong tài liệu."

CÂU HỎI: {question}

TÀI LIỆU THAM KHẢO:
{context}
>>>>>>> Stashed changes

CÂU TRẢ LỜI:"""

<<<<<<< Updated upstream
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


    
=======
    def generate_answer(self, question, context):
        prompt = self._build_prompt(question, context)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=512)
            )
            return response.text.strip() if response.text else "Gemini không trả về nội dung."
        except Exception as e:
            return f"Lỗi gọi Gemini API: {e}"

    def ask(self, question, top_k=5):
        if not question.strip():
            return {"answer": "Câu hỏi rỗng.", "sources": []}
            
        results = self.retriever.retrieve(question, top_k=top_k)
        if not results:
            return {"answer": "Không tìm thấy thông tin phù hợp trong tài liệu.", "sources": []}
            
        context = build_context(results)
        answer = self.generate_answer(question, context)
        
        return {"answer": answer, "sources": results}

if __name__ == "__main__":
    print("Khởi động RAG Pipeline...")
    pipeline = RAGPipeline()
    print("Sẵn sàng! Nhập 'exit' để thoát.\n")
    
    while True:
        q = input("Nhập câu hỏi: ")
        if q.lower() in ['exit', 'quit']: break
        
        res = pipeline.ask(q)
        print("\nCâu trả lời:\n", res["answer"])
        print_sources(res["sources"])
        print("-" * 50)
>>>>>>> Stashed changes

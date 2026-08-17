import logging
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from retriever import load_retriever_resources, retrieve

logging.basicConfig(level=logging.INFO, format="%(message)s")
load_dotenv()

def init_rag_system():
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL")
    
    if not api_key or not model_name:
        raise ValueError("Lỗi: Thiếu biến môi trường GEMINI_API_KEY hoặc GEMINI_MODEL.")
        
    client = genai.Client(api_key=api_key)
    resources = load_retriever_resources()
    
    return client, model_name, resources

def build_context(results):
    context_blocks = []
    
    for index, record in enumerate(results, start=1):
        source_file = record.get("source_file", "Không rõ")
        pages = record.get("pages", [])
        page_str = ", ".join(map(str, pages)) if pages else "Không rõ"
        content = record.get("embedding_text", "")
        
        block = f"[Nguồn {index}]\nTài liệu: {source_file}\nTrang: {page_str}\nNội dung:\n{content}"
        context_blocks.append(block)
        
    return "\n\n".join(context_blocks)

def ask(question, client, model_name, resources, top_k=5):
    if not question.strip():
        return {"answer": "Vui lòng nhập câu hỏi.", "sources": []}
        
    # Truy xuất tài liệu
    results = retrieve(question, resources, top_k=top_k)
    
    if len(results) == 0:
        return {"answer": "Không tìm thấy thông tin trong hệ thống tài liệu quy chế.", "sources": []}
        
    context = build_context(results)
    
    # Khởi tạo Prompt cho Gemini
    prompt = f"""Bạn là trợ lý giải đáp thắc mắc về quy chế đào tạo đại học.
Nhiệm vụ: Dựa vào thông tin ở phần TÀI LIỆU THAM KHẢO, hãy trả lời câu hỏi của người dùng.
Quy tắc:
1. KHÔNG bịaa đặt. Chỉ dùng thông tin được cung cấp.
2. Viết câu trả lời rõ ràng, dễ hiểu.
3. PHẢI trích dẫn nguồn ở định dạng [Nguồn n] ngay phía sau câu trả lời (VD: ...được quy định theo khoản 2 [Nguồn 1]).
4. Nếu tài liệu không chứa đủ thông tin để trả lời, hãy nói: "Tài liệu hiện tại không đề cập chi tiết về vấn đề này."

CÂU HỎI CỦA NGƯỜI DÙNG: {question}

TÀI LIỆU THAM KHẢO:
{context}

TRẢ LỜI:"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1, 
                max_output_tokens=1024
            )
        )
        answer = response.text.strip()
    except Exception as e:
        logging.error(f"Lỗi khi gọi Gemini: {e}")
        answer = "Hệ thống AI đang bận hoặc gặp lỗi, vui lòng thử lại sau."
        
    return {"answer": answer, "sources": results}
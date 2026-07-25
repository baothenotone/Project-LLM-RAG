import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

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

CÂU TRẢ LỜI:"""

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
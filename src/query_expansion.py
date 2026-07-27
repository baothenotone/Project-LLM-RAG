import json
import os
import unicodedata
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

class QueryExpander:
    def __init__(self):
        self.root = Path(__file__).resolve().parent.parent
        load_dotenv(self.root / ".env")
        
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL")
        self.enable = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() in ["true", "1", "yes", "on"]
        self.count = 3
        
        if self.enable and self.api_key and self.model_name:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def _clean_text(self, text):
        text = unicodedata.normalize("NFC", str(text or ""))
        return " ".join(text.split()).strip()

    def expand(self, question):
        question = self._clean_text(question)
        if not question:
            return ()
            
        base_queries = [question]
        
        if not self.client:
            return tuple(base_queries)

        prompt = f"""Bạn là chuyên gia tối ưu hóa truy vấn tìm kiếm cho hệ thống tra cứu văn bản quy chế.
Nhiệm vụ: Chuyển đổi câu hỏi thành {self.count} truy vấn tìm kiếm ngắn gọn, tối ưu.
- Nếu là định nghĩa (VD: "Tín chỉ là gì?"), viết lại thành: "Định nghĩa tín chỉ", "Tín chỉ là".
- Nếu là thủ tục/quy định (VD: "Bao nhiêu điểm thì rớt?"), viết thành: "Quy định buộc thôi học", "Điều kiện cảnh báo học vụ".
- KHÔNG trả lời. Chỉ trả về JSON theo mẫu chính xác sau:
{{"queries": ["truy vấn 1", "truy vấn 2", "truy vấn 3"]}}

Câu hỏi của người dùng: {question}"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                    response_mime_type="application/json",
                ),
            )
            data = json.loads(response.text)
            expanded = data.get("queries", [])
            
            # Gộp và loại bỏ trùng lặp
            results = []
            seen = set()
            for q in base_queries + expanded:
                cq = self._clean_text(q)
                if cq.lower() not in seen and cq:
                    results.append(cq)
                    seen.add(cq.lower())
            return tuple(results[:self.count + 1])
            
        except Exception as e:
            print(f"Cảnh báo: Lỗi mở rộng câu hỏi ({e}). Dùng câu gốc.")
            return tuple(base_queries)
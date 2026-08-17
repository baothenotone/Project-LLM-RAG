import json
import logging
import os
import unicodedata
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

def expand_query(question, count=3):
    # 1. Làm sạch câu hỏi gốc
    if not question:
        return []
        
    cleaned_question = unicodedata.normalize("NFC", str(question))
    cleaned_question = " ".join(cleaned_question.split()).strip()
    
    if cleaned_question == "": 
        return []
        
    base_queries = [cleaned_question]
    
    # 2. Kiểm tra cấu hình
    enable_expansion = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() in ["true", "1", "yes", "on"]
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL")
    
    if not enable_expansion or not api_key or not model_name:
        return base_queries

    # 3. Gọi Gemini để sinh thêm câu hỏi
    prompt = f"""Bạn là chuyên gia tra cứu văn bản pháp luật.
Nhiệm vụ: Chuyển câu hỏi sau thành {count} câu truy vấn tìm kiếm độc lập, ngắn gọn.
Ví dụ:
- "Tín chỉ là gì?" -> "Định nghĩa tín chỉ", "Khái niệm tín chỉ"
- "Bao nhiêu điểm thì rớt?" -> "Quy định buộc thôi học", "Điều kiện cảnh báo học vụ"
Chỉ trả về JSON theo mẫu: {{"queries": ["câu 1", "câu 2", "câu 3"]}}

Câu hỏi của người dùng: {cleaned_question}"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0, 
                response_mime_type="application/json"
            )
        )
        
        response_data = json.loads(response.text)
        expanded_queries = response_data.get("queries", [])
        
        # 4. Gộp và loại bỏ các câu hỏi trùng lặp
        final_results = []
        seen_queries = set()
        
        for q in base_queries + expanded_queries:
            clean_q = " ".join(str(q).split()).strip()
            lower_q = clean_q.lower()
            
            if clean_q != "" and lower_q not in seen_queries:
                final_results.append(clean_q)
                seen_queries.add(lower_q)
                
        return final_results[:count + 1]
        
    except Exception as e:
        logging.warning(f"Tính năng Query Expansion gặp lỗi: {e}. Sẽ chỉ dùng câu hỏi gốc.")
        return base_queries
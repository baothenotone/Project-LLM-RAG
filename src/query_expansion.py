import json
import logging
import os
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv(
    Path(__file__).resolve().parent.parent / ".env"
)


# Mở rộng câu hỏi thành nhiều truy vấn tìm kiếm
def expand_query(question, count=3):
    if not question:
        return []

    cleaned_question = unicodedata.normalize(
        "NFC",
        str(question),
    )

    cleaned_question = " ".join(
        cleaned_question.split()
    ).strip()

    if not cleaned_question:
        return []

    base_queries = [cleaned_question]

    enable_expansion = os.getenv(
        "ENABLE_QUERY_EXPANSION",
        "true",
    ).lower() in {
        "true",
        "1",
        "yes",
        "on",
    }

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    model_name = os.getenv(
        "GEMINI_MODEL"
    )

    if (
        not enable_expansion
        or not api_key
        or not model_name
    ):
        return base_queries

    prompt = f"""Bạn là chuyên gia tra cứu văn bản pháp luật.

Chuyển câu hỏi sau thành {count} câu truy vấn tìm kiếm độc lập, ngắn gọn.

Ví dụ:
- "Tín chỉ là gì?" -> "Định nghĩa tín chỉ", "Khái niệm tín chỉ"
- "Bao nhiêu điểm thì rớt?" -> "Quy định buộc thôi học", "Điều kiện cảnh báo học vụ"

Chỉ trả về JSON:
{{"queries": ["câu 1", "câu 2", "câu 3"]}}

Câu hỏi: {cleaned_question}"""

    try:
        client = genai.Client(
            api_key=api_key
        )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )

        data = json.loads(
            response.text
        )

        expanded_queries = data.get(
            "queries",
            [],
        )

        results = []
        seen = set()

        for query in (
            base_queries
            + expanded_queries
        ):
            query = " ".join(
                str(query).split()
            ).strip()

            key = query.lower()

            if query and key not in seen:
                results.append(query)
                seen.add(key)

        return results[: count + 1]

    except Exception as error:
        logging.warning(
            f"Query Expansion gặp lỗi: {error}"
        )

        return base_queries
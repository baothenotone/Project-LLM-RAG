import json
import re
from pathlib import Path


INPUT_FILE = Path("data/processed/pages.json")
CLEANED_FILE = Path("data/processed/cleaned_pages.json")
CHUNKS_FILE = Path("data/chunks/chunks.json")

CHUNK_SIZE = 180
OVERLAP = 30
MIN_TEXT_LENGTH = 30


def read_json(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def clean_text(text):
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    cleaned_lines = []

    for index, line in enumerate(lines):
        line = line.strip()

        if line == "":
            continue

        # Một số PDF có số trang đứng riêng ở dòng đầu, bỏ để tránh nhiễu khi truy xuất.
        if index == 0 and line.isdigit():
            continue

        # Bỏ các dòng/dãy chấm dài thường xuất hiện trong biểu mẫu PDF.
        line = re.sub(r"\.{5,}", " ", line)
        line = re.sub(r"\s+", " ", line)

        cleaned_lines.append(line)

    cleaned_text = " ".join(cleaned_lines)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def clean_all_pages(pages):
    cleaned_pages = []

    for page in pages:
        text = clean_text(page.get("text", ""))

        if len(text) < MIN_TEXT_LENGTH:
            continue

        cleaned_page = {
            "doc_id": page.get("doc_id", ""),
            "file_name": page.get("file_name", ""),
            "page": page.get("page", ""),
            "text": text,
            "source": page.get("source", ""),
        }

        cleaned_pages.append(cleaned_page)

    return cleaned_pages


def split_text_to_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

        if end >= len(words):
            break

        # Lùi lại một đoạn nhỏ để chunk sau vẫn giữ được ngữ cảnh của chunk trước.
        start = end - overlap

    return chunks


def create_chunks(cleaned_pages):
    chunks = []

    for page in cleaned_pages:
        page_chunks = split_text_to_chunks(page["text"])

        for chunk_number, chunk_text in enumerate(page_chunks, start=1):
            chunk_id = f"{page['doc_id']}_p{page['page']}_c{chunk_number}"

            chunk = {
                "chunk_id": chunk_id,
                "doc_id": page["doc_id"],
                "file_name": page["file_name"],
                "page": page["page"],
                "chunk_index": chunk_number,
                "text": chunk_text,
                "source": page["source"],
            }

            chunks.append(chunk)

    return chunks


def main():
    print("Bắt đầu Week 3: làm sạch text và chia chunk...")

    pages = read_json(INPUT_FILE)
    cleaned_pages = clean_all_pages(pages)
    chunks = create_chunks(cleaned_pages)

    write_json(cleaned_pages, CLEANED_FILE)
    write_json(chunks, CHUNKS_FILE)

    print(f"Số trang ban đầu: {len(pages)}")
    print(f"Số trang sau khi làm sạch: {len(cleaned_pages)}")
    print(f"Số chunk đã tạo: {len(chunks)}")
    print(f"Đã lưu file: {CLEANED_FILE}")
    print(f"Đã lưu file: {CHUNKS_FILE}")


if __name__ == "__main__":
    main()

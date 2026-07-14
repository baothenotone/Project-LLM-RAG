from pathlib import Path
import json
import re

#Đọc dữ liệu pages.json
def load_pages() -> list[dict]:
    input_file_path = "data/processed/pages.json"
    input_path = Path(input_file_path)

    with open(input_path, "r", encoding="utf-8") as file:
        pages = json.load(file)
    return pages

#Làm sạch 
def clean_text(text: str) -> str:

    text = text.replace("\t"," ")
    text = re.sub(r"\n+", "\n",text)
    text = re.sub(r"[ ]+", " ", text)

    text = text.strip()
    return text

#Chia thành nhiều chunk
def split_text_into_chunk(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks = []

    if not text:
        return chunks
    star_index = 0
    while star_index < len(text):
        end_index = star_index + chunk_size 

        chunk_text = text[star_index:end_index].strip()

        if chunk_text != "":
            chunks.append(chunk_text)

        star_index = end_index - overlap 
    return chunks
#Tạo danh sách chunk 
def build_chunks(pages: list[dict], chunk_size: int=1000, overlap: int=200) -> list[dict]:
    all_chunks = []

    for page in pages:
        doc_id = page["doc_id"]
        file_name = page["file_name"]
        page_number = page["page"]
        source = page["source"]

        #Làm sạch text của từng trang
        cleaned_text = clean_text(page["text"])

        #Chia text của tran thành nhiều chunk
        chunks_in_page = split_text_into_chunk(text=cleaned_text, chunk_size=chunk_size, overlap=overlap)

        for chunk_index, chunk_text in enumerate(chunks_in_page):

            chunk_number = chunk_index + 1
            chunk_id = f"{doc_id}_PAGE{page_number:03d}_CHUNK{chunk_number:03d}"

            chunk_data = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "file_name": file_name,
                "page": page_number,
                "chunk_index": chunk_number,
                "text": chunk_text,
                "source": source
            }
            all_chunks.append(chunk_data)
    return all_chunks
def save_chunks(chunks: list[dict]):
    output_file_path = "data/chunks/chunks.json"
    output_path = Path(output_file_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)
def main():
    pages = load_pages()

    chunks = build_chunks(pages=pages, chunk_size=1000, overlap=200)

    save_chunks(chunks)
    print(f"Tổng số trang đầu vào: {len(pages)}")
    print(f"Tổng số chunk đã tạo: {len(chunks)}")

if __name__ == "__main__":
    main()
    
from pathlib import Path
import json
import fitz     # PyMuPDF: dùng để đọc nội dung file PDF


# Đọc một file PDF
def read_pdf(pdf_path: Path, doc_id: str) -> list[dict]:
    pages = []

    pdf_file = fitz.open(pdf_path)

    for page_number in range(len(pdf_file)):
        page = pdf_file[page_number]

        real_page_number = page_number + 1
        page_text = page.get_text("text")

        page_data = {
            "doc_id": doc_id,
            "file_name": pdf_path.name,
            "page": real_page_number,
            "text": page_text.strip(),
            "source": str(pdf_path)
        }

        pages.append(page_data)

    pdf_file.close()

    return pages

# Đọc tất cả PDF trong thư mục
def load_pdfs() -> list[dict]:

    all_pages = []

    pdf_files = sorted(Path("data/raw").rglob("*.pdf"))

    for index, pdf_path in enumerate(pdf_files):
        #Dạng DOC001,..
        doc_number = index + 1
    
        doc_id = f"DOC{doc_number:03d}"

        pages_in_pdf = read_pdf(pdf_path, doc_id)
        all_pages.extend(pages_in_pdf)

    return all_pages


# Lưu thành file JSON
def save_json(pages: list[dict]):
    output_file_path = "data/processed/pages.json"
    output_path = Path(output_file_path)

    #Tạo thư mục data/processed nếu chưa có
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(pages, file, ensure_ascii=False, indent=2)

# Chương trình chính
def main():
    pages = load_pdfs()

    save_json(pages)

    print("Đã đọc", len(pages), "trang")

if __name__ == "__main__":
    main()
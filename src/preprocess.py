import gc
import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.chunking import HybridChunker
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Định nghĩa các thư mục
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"

# Tạo thư mục nếu chưa có
RAW_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    if not text:
        return ""
    # Xóa các ký tự ẩn
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00ad", "").replace("\u200b", "").replace("\x00", "")
    # Đưa về một chuẩn xuống dòng
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Xóa khoảng trắng thừa
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_legal_metadata(content, headings):
    searchable_text = "\n".join(headings + [content])
    metadata = {
        "chapter": None,
        "section": None,
        "article": None,
        "clause": None,
        "point": None
    }
    
    # Tìm Chương
    match_chapter = re.search(r"\bChương\s+[IVXLCDM]+\b", searchable_text, flags=re.IGNORECASE)
    if match_chapter:
        metadata["chapter"] = match_chapter.group(0).strip()
        
    # Tìm Mục
    match_section = re.search(r"\bMục\s+\d+\b", searchable_text, flags=re.IGNORECASE)
    if match_section:
        metadata["section"] = match_section.group(0).strip()
        
    # Tìm Điều
    match_article = re.search(r"\bĐiều\s+\d+[a-zA-Z]?\b", searchable_text, flags=re.IGNORECASE)
    if match_article:
        metadata["article"] = match_article.group(0).strip()
        
    # Tìm Khoản (VD: "1. ")
    match_clause = re.search(r"(?m)^\s*(\d+)\.\s+\S+", content)
    if match_clause:
        metadata["clause"] = f"Khoản {match_clause.group(1)}"
        
    # Tìm Điểm (VD: "a) ")
    match_point = re.search(r"(?m)^\s*([a-zđ])\)\s+\S+", content, flags=re.IGNORECASE)
    if match_point:
        metadata["point"] = f"Điểm {match_point.group(1).lower()}"
        
    return metadata

def init_docling():
    pipeline_options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    # Dùng chế độ ACCURATE để xuất bảng thành định dạng Markdown (giúp LLM đọc tốt hơn)
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE 
    pipeline_options.table_structure_options.do_cell_matching = True
    
    pipeline_options.layout_batch_size = 1
    pipeline_options.table_batch_size = 1
    pipeline_options.ocr_batch_size = 1
    pipeline_options.queue_max_size = 4
    pipeline_options.accelerator_options = AcceleratorOptions(num_threads=2, device=AcceleratorDevice.CPU)
    
    pdf_option = PdfFormatOption(pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend)
    converter = DocumentConverter(allowed_formats=[InputFormat.PDF], format_options={InputFormat.PDF: pdf_option})
    
    # Khởi tạo mô hình cắt chunk
    raw_tokenizer = AutoTokenizer.from_pretrained("bkai-foundation-models/vietnamese-bi-encoder")
    tokenizer = HuggingFaceTokenizer(tokenizer=raw_tokenizer, max_tokens=256)
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    
    return converter, chunker

def process_pdf(pdf_file, converter, chunker):
    # Đọc PDF
    result = converter.convert(source=str(pdf_file))
    document = result.document

    # Lưu lại file Markdown để xem (nếu cần)
    markdown_path = EXTRACTED_DIR / f"{pdf_file.stem}.md"
    markdown_path.write_text(document.export_to_markdown(), encoding="utf-8")

    records = []
    seen_chunks = set()
    
    # Cắt nhỏ file PDF thành các đoạn (chunks)
    chunk_iterator = chunker.chunk(dl_doc=document)
    
    for original_index, chunk in enumerate(chunk_iterator):
        content = clean_text(getattr(chunk, "text", ""))
        
        # 1. Trích xuất trang (pages)
        pages_set = set()
        meta = getattr(chunk, "meta", None)
        if meta and hasattr(meta, "doc_items"):
            for item in meta.doc_items or []:
                for prov in getattr(item, "prov", []) or []:
                    page_no = getattr(prov, "page_no", None)
                    if isinstance(page_no, int):
                        pages_set.add(page_no)
        pages = sorted(list(pages_set))
        
        # 2. Trích xuất tiêu đề (headings)
        headings = []
        if meta and hasattr(meta, "headings"):
            for heading in meta.headings or []:
                cleaned_heading = clean_text(str(heading))
                if cleaned_heading:
                    headings.append(cleaned_heading)
                    
        # 3. Trích xuất nhãn (labels)
        labels = []
        if meta and hasattr(meta, "doc_items"):
            for item in meta.doc_items or []:
                label = getattr(item, "label", None)
                if label is not None:
                    label_str = str(label).split(".")[-1].lower()
                    if label_str not in labels:
                        labels.append(label_str)
        
        # --- BỘ LỌC CHUNK ---
        if not content:
            continue
            
        # Bỏ qua header/footer
        if labels and set(labels).issubset({"page_header", "page_footer"}):
            continue
            
        # Bỏ qua chunk chỉ chứa số trang
        if re.fullmatch(r"(?i)\s*(?:trang\s*)?[-–—]?\s*\d+\s*[-–—]?\s*", content):
            continue
            
        # Bỏ qua chunk rác, quá ngắn
        letter_count = sum(c.isalpha() for c in content)
        contains_article = re.search(r"\bĐiều\s+\d+[a-zA-Z]?\b", content, re.IGNORECASE)
        if letter_count < 8 or (len(content) < 35 and not contains_article):
            continue
            
        # Lọc trùng lặp
        duplicate_key = re.sub(r"\s+", " ", content.lower()).strip()
        if duplicate_key in seen_chunks:
            continue
        seen_chunks.add(duplicate_key)
        # --- KẾT THÚC BỘ LỌC ---

        # Lấy nội dung để embedding (chứa thêm ngữ cảnh nếu có)
        if hasattr(chunker, "contextualize"):
            embedding_text = clean_text(chunker.contextualize(chunk=chunk))
        else:
            embedding_text = content
            
        # Tìm số Quyết định/Thông tư
        document_number = None
        source_text = f"{pdf_file.name}\n{embedding_text[:3000]}"
        doc_patterns = [r"\b\d+/\d{4}/TT-[A-ZĐ-]+\b", r"\b\d+/QĐ-[A-ZĐ-]+\b", r"\b\d+/[A-ZĐ]+-[A-ZĐ-]+\b"]
        for pattern in doc_patterns:
            match = re.search(pattern, source_text, flags=re.IGNORECASE)
            if match:
                document_number = match.group(0).upper()
                break

        # Tạo ID cho chunk
        raw_value = f"{pdf_file.name}|{pages}|{duplicate_key}"
        digest = hashlib.sha1(raw_value.encode("utf-8")).hexdigest()[:14]
        chunk_id = f"{pdf_file.stem}_{digest}"
        
        legal_metadata = extract_legal_metadata(content, headings)

        record = {
            "chunk_id": chunk_id,
            "content": content,
            "embedding_text": embedding_text,
            "source_file": pdf_file.name,
            "document_title": headings[0] if headings else pdf_file.stem,
            "document_number": document_number,
            "pages": pages,
        }
        # Gộp các metadata pháp lý vào record
        record.update(legal_metadata)
        records.append(record)

    return records

def run_preprocess():
    pdf_files = sorted(RAW_DIR.rglob("*.pdf"))
    if not pdf_files:
        logging.warning("Không tìm thấy file PDF nào trong thư mục data/raw/")
        return

    converter, chunker = init_docling()
    all_records = []

    for pdf_file in pdf_files:
        try:
            logging.info(f"Đang phân tích file: {pdf_file.name}")
            records = process_pdf(pdf_file, converter, chunker)
            all_records.extend(records)
        except Exception as e:
            logging.error(f"Lỗi khi xử lý {pdf_file.name}: {e}")
        finally:
            gc.collect()

    output_file = CHUNKS_DIR / "chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
        
    logging.info(f"Hoàn tất! Đã lưu {len(all_records)} chunks vào {output_file}")

if __name__ == "__main__":
    run_preprocess()
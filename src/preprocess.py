from __future__ import annotations

import gc
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from transformers import AutoTokenizer


# ==========================================================
# 1. CẤU HÌNH CHUNG
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

CHUNKS_FILE = CHUNKS_DIR / "chunks.json"
REPORT_FILE = REPORTS_DIR / "preprocess_report.json"
PREVIEW_FILE = REPORTS_DIR / "chunks_preview.txt"

# Dùng cùng tokenizer với embedding model để tránh tạo chunk dài hơn
# giới hạn mà model embedding thực sự có thể đọc.
EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Model trên có max_seq_length = 128. Đặt 120 để chừa một ít chỗ cho
# tiêu đề được HybridChunker thêm vào phần văn bản có ngữ cảnh.
MAX_CHUNK_TOKENS = 120

# Chạy thử một file trước. Khi kết quả ổn, đổi thành None để xử lý tất cả.
MAX_FILES: int | None = 1

# PDF HUSC chủ yếu là PDF text nên mặc định chưa bật OCR để chạy nhanh hơn.
# Nếu file bị scan hoặc copy ra toàn ký tự lỗi, đổi ENABLE_OCR thành True.
ENABLE_OCR = False
FORCE_FULL_PAGE_OCR = False

MIN_CHUNK_CHARS = 35
PREVIEW_CHUNK_COUNT = 30


# ==========================================================
# 2. CÁC HÀM LÀM SẠCH VĂN BẢN
# ==========================================================


def clean_text(text: str) -> str:
    """Làm sạch những lỗi định dạng phổ biến nhưng không tự sửa nội dung.

    Ta chỉ chuẩn hóa Unicode, khoảng trắng và ký tự vô hình. Những lỗi OCR
    như "Gido duc" không nên tự thay bằng regex vì tài liệu quy chế cần giữ
    nguyên nội dung pháp lý.
    """

    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)

    # Một số PDF chèn các ký tự này để điều khiển hiển thị. Chúng không có
    # ý nghĩa khi tìm kiếm và đôi khi làm câu bị tách sai.
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\u200b", "")  # zero-width space
    text = text.replace("\ufeff", "")  # byte-order mark
    text = text.replace("\x00", "")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalized_key(text: str) -> str:
    """Tạo phiên bản đơn giản của text để phát hiện chunk trùng lặp."""

    text = clean_text(text).lower()
    return re.sub(r"\s+", " ", text).strip()


def has_suspicious_characters(text: str) -> bool:
    """Đánh dấu các đoạn có dấu hiệu trích xuất sai để kiểm tra thủ công."""

    return "\ufffd" in text or "���" in text


# ==========================================================
# 3. ĐỌC METADATA TỪ CHUNK CỦA DOCLING
# ==========================================================


def get_page_numbers(chunk: Any) -> list[int]:
    """Lấy số trang từ provenance của các document item trong chunk."""

    pages: set[int] = set()
    meta = getattr(chunk, "meta", None)

    for item in getattr(meta, "doc_items", []) or []:
        for provenance in getattr(item, "prov", []) or []:
            page_number = getattr(provenance, "page_no", None)
            if isinstance(page_number, int):
                pages.add(page_number)

    return sorted(pages)


def get_headings(chunk: Any) -> list[str]:
    """Lấy chuỗi tiêu đề cha mà Docling đã gắn cho chunk."""

    meta = getattr(chunk, "meta", None)
    headings = getattr(meta, "headings", []) or []

    cleaned_headings: list[str] = []
    for heading in headings:
        value = clean_text(str(heading))
        if value:
            cleaned_headings.append(value)

    return cleaned_headings


def get_item_labels(chunk: Any) -> list[str]:
    """Lấy loại nội dung như text, table, page_header hoặc page_footer."""

    labels: list[str] = []
    meta = getattr(chunk, "meta", None)

    for item in getattr(meta, "doc_items", []) or []:
        label = getattr(item, "label", None)
        if label is None:
            continue

        # Với Enum, str(label) có thể trả về "DocItemLabel.TEXT".
        # Chỉ giữ phần cuối để metadata gọn và dễ đọc.
        value = str(label).split(".")[-1].lower()
        if value not in labels:
            labels.append(value)

    return labels


def contextualize_chunk(chunker: HybridChunker, chunk: Any) -> str:
    """Tạo text dùng cho embedding, có thêm tiêu đề và context.

    Docling hiện dùng contextualize(). Một số phiên bản cũ dùng serialize(),
    nên hàm này có fallback để project ít bị lỗi khi đổi môi trường.
    """

    if hasattr(chunker, "contextualize"):
        return clean_text(chunker.contextualize(chunk=chunk))

    if hasattr(chunker, "serialize"):
        return clean_text(chunker.serialize(chunk))

    return clean_text(getattr(chunk, "text", ""))


# ==========================================================
# 4. NHẬN DIỆN CẤU TRÚC VĂN BẢN QUY CHẾ
# ==========================================================


def first_match(pattern: str, text: str) -> str | None:
    """Trả về kết quả regex đầu tiên hoặc None nếu không tìm thấy."""

    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(0).strip() if match else None


def extract_legal_metadata(
    content: str,
    headings: list[str],
) -> dict[str, str | None]:
    """Nhận diện Chương, Mục, Điều, Khoản và Điểm khi thông tin xuất hiện.

    Đây là metadata hỗ trợ trích dẫn. Nó không thay thế nội dung gốc và cũng
    không đoán thông tin khi văn bản không ghi rõ.
    """

    searchable_text = "\n".join([*headings, content])

    chapter = first_match(r"\bChương\s+[IVXLCDM]+\b", searchable_text)
    section = first_match(r"\bMục\s+\d+\b", searchable_text)
    article = first_match(r"\bĐiều\s+\d+[a-zA-Z]?\b", searchable_text)

    # Khoản thường bắt đầu bằng "1.", "2."... Chỉ xét đầu dòng để tránh
    # nhầm các con số nằm trong nội dung hoặc ngày tháng.
    clause_match = re.search(
        r"(?m)^\s*(\d+)\.\s+\S+",
        content,
    )
    clause = f"Khoản {clause_match.group(1)}" if clause_match else None

    point_match = re.search(
        r"(?m)^\s*([a-zđ])\)\s+\S+",
        content,
        flags=re.IGNORECASE,
    )
    point = f"Điểm {point_match.group(1).lower()}" if point_match else None

    return {
        "chapter": chapter,
        "section": section,
        "article": article,
        "clause": clause,
        "point": point,
    }


def extract_document_number(text: str, filename: str) -> str | None:
    """Tìm số hiệu văn bản, ví dụ 673/QĐ-ĐHKH hoặc 08/2021/TT-BGDĐT."""

    source = f"{filename}\n{text[:3000]}"

    patterns = [
        r"\b\d+/\d{4}/TT-[A-ZĐ-]+\b",
        r"\b\d+/QĐ-[A-ZĐ-]+\b",
        r"\b\d+/[A-ZĐ]+-[A-ZĐ-]+\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if match:
            return match.group(0).upper()

    return None


# ==========================================================
# 5. KIỂM TRA VÀ TẠO RECORD CHUNK
# ==========================================================


def should_keep_chunk(content: str, labels: list[str]) -> bool:
    """Loại phần rỗng, số trang và header/footer đứng riêng một mình."""

    if not content:
        return False

    ignored_labels = {"page_header", "page_footer"}
    if labels and set(labels).issubset(ignored_labels):
        return False

    # Các chunk chỉ chứa số trang như "12", "- 12 -" hoặc "Trang 12".
    if re.fullmatch(
        r"(?i)\s*(?:trang\s*)?[-–—]?\s*\d+\s*[-–—]?\s*",
        content,
    ):
        return False

    letter_count = sum(character.isalpha() for character in content)
    if letter_count < 8:
        return False

    # Vẫn giữ tiêu đề Điều ngắn vì nó có ý nghĩa pháp lý.
    contains_article = bool(
        re.search(r"\bĐiều\s+\d+[a-zA-Z]?\b", content, re.IGNORECASE)
    )

    if len(content) < MIN_CHUNK_CHARS and not contains_article:
        return False

    return True


def make_chunk_id(source_file: str, pages: list[int], content: str) -> str:
    """Tạo ID ổn định để metadata và vector luôn liên kết đúng chunk."""

    raw_value = f"{source_file}|{pages}|{normalized_key(content)}"
    digest = hashlib.sha1(raw_value.encode("utf-8")).hexdigest()[:14]
    return f"{Path(source_file).stem}_{digest}"


def build_chunk_records(
    document: Any,
    source_file: Path,
    chunker: HybridChunker,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Chuyển DoclingDocument thành các record sẵn sàng cho bước ingest."""

    records: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()

    skipped_short = 0
    skipped_duplicate = 0

    chunk_iterator = chunker.chunk(dl_doc=document)

    for original_index, chunk in enumerate(chunk_iterator):
        content = clean_text(getattr(chunk, "text", ""))
        embedding_text = contextualize_chunk(chunker, chunk)
        pages = get_page_numbers(chunk)
        headings = get_headings(chunk)
        labels = get_item_labels(chunk)

        if not should_keep_chunk(content, labels):
            skipped_short += 1
            continue

        # Không để header/footer hoặc đoạn lặp lại xuất hiện nhiều lần trong
        # cùng một tài liệu vì chúng dễ làm nhiễu kết quả retrieval.
        duplicate_key = normalized_key(content)
        if duplicate_key in seen_chunks:
            skipped_duplicate += 1
            continue
        seen_chunks.add(duplicate_key)

        legal_metadata = extract_legal_metadata(content, headings)

        record = {
            "chunk_id": make_chunk_id(source_file.name, pages, content),
            "content": content,
            "embedding_text": embedding_text,
            "source_file": source_file.name,
            "source_path": str(source_file.relative_to(PROJECT_ROOT)),
            "document_title": headings[0] if headings else source_file.stem,
            "document_number": extract_document_number(
                embedding_text,
                source_file.name,
            ),
            "original_chunk_index": original_index,
            "headings": headings,
            "labels": labels,
            "pages": pages,
            "page_start": pages[0] if pages else None,
            "page_end": pages[-1] if pages else None,
            **legal_metadata,
            "character_count": len(content),
            "word_count": len(content.split()),
            "has_suspicious_characters": has_suspicious_characters(content),
        }

        records.append(record)

    statistics = {
        "kept_chunks": len(records),
        "skipped_short_or_noise": skipped_short,
        "skipped_duplicate": skipped_duplicate,
    }

    return records, statistics


# ==========================================================
# 6. KHỞI TẠO DOCLING VÀ CHUNKER
# ==========================================================


def create_converter() -> DocumentConverter:
    """Khởi tạo DocumentConverter một lần cho toàn bộ thư mục PDF."""

    pipeline_options = PdfPipelineOptions(
        do_ocr=ENABLE_OCR,
        do_table_structure=True,
    )

    # Chế độ ACCURATE chạy chậm hơn FAST nhưng phù hợp hơn với sổ tay học vụ
    # có nhiều bảng. Với tài liệu chỉ có chữ, chênh lệch không đáng kể.
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    if ENABLE_OCR and FORCE_FULL_PAGE_OCR:
        pipeline_options.ocr_options.force_full_page_ocr = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )


def create_chunker() -> HybridChunker:
    """Tạo HybridChunker dùng đúng tokenizer của embedding model."""

    raw_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

    tokenizer = HuggingFaceTokenizer(
        tokenizer=raw_tokenizer,
        max_tokens=MAX_CHUNK_TOKENS,
    )

    return HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
    )


# ==========================================================
# 7. XỬ LÝ TỪNG FILE PDF
# ==========================================================


def process_pdf(
    pdf_file: Path,
    converter: DocumentConverter,
    chunker: HybridChunker,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Chuyển một PDF thành Markdown kiểm tra và danh sách chunk."""

    print(f"\nĐang xử lý: {pdf_file.name}")

    result = converter.convert(source=str(pdf_file))
    document = result.document

    # Markdown chỉ dùng để mở ra kiểm tra bằng mắt. Ta không đọc Markdown
    # trở lại để chunk vì làm vậy sẽ mất provenance như số trang.
    markdown_file = EXTRACTED_DIR / f"{pdf_file.stem}.md"
    markdown_file.write_text(
        document.export_to_markdown(),
        encoding="utf-8",
    )

    records, statistics = build_chunk_records(
        document=document,
        source_file=pdf_file,
        chunker=chunker,
    )

    missing_page_count = sum(
        1 for record in records if not record["pages"]
    )
    suspicious_count = sum(
        1 for record in records if record["has_suspicious_characters"]
    )

    report = {
        "source_file": pdf_file.name,
        "status": "success",
        "pdf_page_count": len(getattr(document, "pages", {}) or {}),
        "markdown_file": str(markdown_file.relative_to(PROJECT_ROOT)),
        **statistics,
        "chunks_without_page_number": missing_page_count,
        "suspicious_chunks": suspicious_count,
    }

    print(f"  Giữ lại: {len(records)} chunk")
    print(f"  Thiếu số trang: {missing_page_count}")
    print(f"  Cần kiểm tra ký tự: {suspicious_count}")

    return records, report


# ==========================================================
# 8. XUẤT FILE KIỂM TRA
# ==========================================================


def write_preview(records: list[dict[str, Any]]) -> None:
    """Xuất một số chunk đầu ra file text để kiểm tra nhanh."""

    sections: list[str] = []

    for index, record in enumerate(records[:PREVIEW_CHUNK_COUNT], start=1):
        sections.append(
            "\n".join(
                [
                    "=" * 80,
                    f"CHUNK {index}: {record['chunk_id']}",
                    f"Nguồn: {record['source_file']}",
                    f"Trang: {record['pages']}",
                    f"Headings: {record['headings']}",
                    f"Điều/Khoản: {record['article']} / {record['clause']}",
                    "-" * 80,
                    record["embedding_text"],
                ]
            )
        )

    PREVIEW_FILE.write_text("\n\n".join(sections), encoding="utf-8")


# ==========================================================
# 9. HÀM CHẠY CHÍNH
# ==========================================================


def main() -> None:
    """Tiền xử lý các PDF trong data/raw và tạo chunks.json."""

    for folder in [RAW_DIR, EXTRACTED_DIR, CHUNKS_DIR, REPORTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DIR.rglob("*.pdf"))

    if MAX_FILES is not None:
        pdf_files = pdf_files[:MAX_FILES]

    if not pdf_files:
        raise FileNotFoundError(
            f"Không tìm thấy file PDF nào trong thư mục: {RAW_DIR}"
        )

    print(f"Tìm thấy {len(pdf_files)} file PDF cần xử lý.")
    print(f"Embedding tokenizer: {EMBEDDING_MODEL}")
    print(f"Giới hạn mỗi chunk: {MAX_CHUNK_TOKENS} tokens")

    converter = create_converter()
    chunker = create_chunker()

    all_records: list[dict[str, Any]] = []
    file_reports: list[dict[str, Any]] = []

    for pdf_file in pdf_files:
        try:
            records, report = process_pdf(pdf_file, converter, chunker)
            all_records.extend(records)
            file_reports.append(report)

        except Exception as error:
            print(f"  Lỗi khi xử lý {pdf_file.name}: {error}")
            file_reports.append(
                {
                    "source_file": pdf_file.name,
                    "status": "failed",
                    "error": str(error),
                }
            )

        finally:
            # Docling dùng khá nhiều bộ nhớ khi phân tích PDF. Thu gom sau mỗi
            # file giúp máy cấu hình vừa phải ổn định hơn khi chạy nhiều file.
            gc.collect()

    CHUNKS_FILE.write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "embedding_model": EMBEDDING_MODEL,
        "max_chunk_tokens": MAX_CHUNK_TOKENS,
        "ocr_enabled": ENABLE_OCR,
        "processed_file_count": len(pdf_files),
        "successful_file_count": sum(
            report["status"] == "success" for report in file_reports
        ),
        "failed_file_count": sum(
            report["status"] == "failed" for report in file_reports
        ),
        "total_chunks": len(all_records),
        "files": file_reports,
    }

    REPORT_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_preview(all_records)

    print("\nHoàn thành preprocess.")
    print(f"Chunks: {CHUNKS_FILE}")
    print(f"Report: {REPORT_FILE}")
    print(f"Preview: {PREVIEW_FILE}")


if __name__ == "__main__":
    main()
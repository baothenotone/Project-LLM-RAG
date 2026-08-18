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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"

TOKENIZER_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
MAX_TOKENS = 256

for folder in (RAW_DIR, EXTRACTED_DIR, CHUNKS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# Làm sạch và chuẩn hóa văn bản
def clean_text(text):
    if not text:
        return ""

    text = unicodedata.normalize("NFC", str(text))
    text = (
        text.replace("\u00ad", "")
        .replace("\u200b", "")
        .replace("\x00", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# Chuyển ngày tháng sang định dạng YYYY-MM-DD
def to_iso_date(match):
    if not match:
        return None

    day, month, year = map(int, match.groups())
    return f"{year:04d}-{month:02d}-{day:02d}"

# Lấy metadata chung của toàn bộ văn bản
def extract_document_metadata(markdown):
    header = markdown[:5000]

    number_match = re.search(
        r"(?im)^SỐ\s*:\s*([0-9]+/QĐ-[A-ZĐ]+)",
        header,
    )

    title_match = re.search(
        r"(?im)^##\s+QUYẾT ĐỊNH\s*$\s*([^\n]+)",
        header,
    )

    date_match = re.search(
        r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
        header,
        re.IGNORECASE,
    )

    title = clean_text(title_match.group(1)) if title_match else None
    issued_date = to_iso_date(date_match)

    effective_match = re.search(
        r"có hiệu lực kể từ ngày\s+(\d{1,2})/(\d{1,2})/(\d{4})",
        markdown,
        re.IGNORECASE,
    )

    effective_date = to_iso_date(effective_match)

    if not effective_date and re.search(
        r"có hiệu lực kể từ ngày ký",
        markdown,
        re.IGNORECASE,
    ):
        effective_date = issued_date

    is_amendment = bool(
        title and title.lower().startswith("sửa đổi")
    )

    amends_document = None
    amends_article = None
    amends_clause = None
    amends_point = None

    if is_amendment:
        match = re.search(
            r"ban hành theo Quyết định số\s+([0-9]+/QĐ-[A-ZĐ]+)",
            title,
            re.IGNORECASE,
        )

        if match:
            amends_document = match.group(1).upper()

        match = re.search(
            r"Sửa đổi\s+Điểm\s+([a-zđ])\s*,\s*"
            r"Khoản\s+(\d+)\s*,\s*"
            r"Điều\s+(\d+)",
            markdown,
            re.IGNORECASE,
        )

        if match:
            amends_point = f"Điểm {match.group(1).lower()}"
            amends_clause = f"Khoản {match.group(2)}"
            amends_article = f"Điều {match.group(3)}"

    return {
        "document_title": title,
        "document_number": (
            number_match.group(1).upper()
            if number_match
            else None
        ),
        "issued_date": issued_date,
        "effective_date": effective_date,
        "document_type": (
            "quyet_dinh_sua_doi"
            if is_amendment
            else "quyet_dinh"
        ),
        "amends_document": amends_document,
        "amends_article": amends_article,
        "amends_clause": amends_clause,
        "amends_point": amends_point,
    }

# Chuẩn hóa heading để so sánh
def normalize_heading(text):
    text = clean_text(text)
    return re.sub(r"\s+", " ", text).lower()

# Xây dựng quan hệ Chương - Mục - Điều
def build_heading_context(markdown):
    context = {}

    chapter = None
    section = None
    article = None

    headings = re.findall(
        r"(?m)^##\s+(.+?)\s*$",
        markdown,
    )

    for raw_heading in headings:
        heading = clean_text(raw_heading)

        match = re.match(
            r"(?i)^CHƯƠNG\s+([IVXLCDM]+)\b",
            heading,
        )

        if match:
            chapter = f"Chương {match.group(1).upper()}"
            section = None
            article = None

        match = re.match(
            r"(?i)^Mục\s+(\d+)\b",
            heading,
        )

        if match:
            section = f"Mục {match.group(1)}"
            article = None

        match = re.match(
            r"(?i)^Điều\s+(\d+[A-Za-z]?)\b",
            heading,
        )

        if match:
            article = f"Điều {match.group(1)}"

        context[normalize_heading(heading)] = {
            "chapter": chapter,
            "section": section,
            "article": article,
        }

    return context

# Lấy metadata pháp lý tương ứng với chunk
def get_legal_context(headings, content, heading_context):
    metadata = {
        "chapter": None,
        "section": None,
        "article": None,
        "clause": None,
        "point": None,
    }

    for heading in reversed(headings):
        found = heading_context.get(
            normalize_heading(heading)
        )

        if found:
            metadata.update(found)
            break

    if not metadata["article"]:
        match = re.search(
            r"(?im)^\s*[-–—]?\s*Điều\s+(\d+[A-Za-z]?)\b",
            content,
        )

        if match:
            metadata["article"] = f"Điều {match.group(1)}"

    return metadata

# Khởi tạo Docling và HybridChunker
def init_docling():
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
    )

    pipeline_options.table_structure_options.mode = (
        TableFormerMode.ACCURATE
    )
    pipeline_options.table_structure_options.do_cell_matching = True

    pipeline_options.layout_batch_size = 1
    pipeline_options.table_batch_size = 1
    pipeline_options.ocr_batch_size = 1
    pipeline_options.queue_max_size = 4
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=2,
        device=AcceleratorDevice.CPU,
    )

    pdf_option = PdfFormatOption(
        pipeline_options=pipeline_options,
        backend=PyPdfiumDocumentBackend,
    )

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: pdf_option},
    )

    raw_tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME
    )

    tokenizer = HuggingFaceTokenizer(
        tokenizer=raw_tokenizer,
        max_tokens=MAX_TOKENS,
    )

    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
    )

    return converter, chunker

# Lấy trang, heading và loại nội dung của chunk
def extract_chunk_metadata(meta):
    pages = set()
    headings = []
    labels = []

    if not meta:
        return [], [], []

    if hasattr(meta, "headings"):
        for heading in meta.headings or []:
            heading = clean_text(heading)

            if heading:
                headings.append(heading)

    if hasattr(meta, "doc_items"):
        for item in meta.doc_items or []:
            label = getattr(item, "label", None)

            if label is not None:
                label = str(label).split(".")[-1].lower()

                if label not in labels:
                    labels.append(label)

            for prov in getattr(item, "prov", []) or []:
                page_no = getattr(prov, "page_no", None)

                if isinstance(page_no, int):
                    pages.add(page_no)

    return sorted(pages), headings, labels

# Kiểm tra chunk có hợp lệ hay không
def is_valid_chunk(content, labels):
    if not content:
        return False

    if labels and set(labels).issubset(
        {"page_header", "page_footer"}
    ):
        return False

    if re.fullmatch(
        r"(?i)\s*(?:trang\s*)?[-–—]?\s*\d+\s*[-–—]?\s*",
        content,
    ):
        return False

    letter_count = sum(
        char.isalpha()
        for char in content
    )

    contains_article = re.search(
        r"\bĐiều\s+\d+[A-Za-z]?\b",
        content,
        re.IGNORECASE,
    )

    if letter_count < 8:
        return False

    if len(content) < 35 and not contains_article:
        return False

    return True

# Chuyển một file PDF thành danh sách chunk
def process_pdf(pdf_file, converter, chunker):
    result = converter.convert(
        source=str(pdf_file)
    )

    document = result.document
    markdown = document.export_to_markdown()

    markdown_path = (
        EXTRACTED_DIR / f"{pdf_file.stem}.md"
    )
    markdown_path.write_text(
        markdown,
        encoding="utf-8",
    )

    document_metadata = extract_document_metadata(
        markdown
    )
    heading_context = build_heading_context(
        markdown
    )

    records = []
    seen_chunks = set()

    for chunk in chunker.chunk(dl_doc=document):
        content = clean_text(
            getattr(chunk, "text", "")
        )

        meta = getattr(chunk, "meta", None)
        pages, headings, labels = extract_chunk_metadata(meta)

        if not is_valid_chunk(content, labels):
            continue

        duplicate_key = re.sub(
            r"\s+",
            " ",
            content.lower(),
        ).strip()

        if duplicate_key in seen_chunks:
            continue

        seen_chunks.add(duplicate_key)

        embedding_text = clean_text(
            chunker.contextualize(chunk=chunk)
        )

        legal_metadata = get_legal_context(
            headings,
            content,
            heading_context,
        )

        raw_value = (
            f"{pdf_file.name}|"
            f"{pages}|"
            f"{duplicate_key}"
        )

        digest = hashlib.sha1(
            raw_value.encode("utf-8")
        ).hexdigest()[:14]

        record = {
            "chunk_id": f"{pdf_file.stem}_{digest}",
            "content": content,
            "embedding_text": embedding_text,
            "source_file": pdf_file.name,
            "heading": headings[-1] if headings else None,
            "headings": headings,
            "pages": pages,
            **document_metadata,
            **legal_metadata,
        }

        records.append(record)

    return records

# Xử lý toàn bộ PDF và lưu chunks.json
def run_preprocess():
    pdf_files = sorted(
        RAW_DIR.rglob("*.pdf")
    )

    if not pdf_files:
        logging.warning(
            "Không tìm thấy file PDF trong data/raw/"
        )
        return

    converter, chunker = init_docling()
    all_records = []

    for pdf_file in pdf_files:
        try:
            logging.info(
                f"Đang xử lý: {pdf_file.name}"
            )

            records = process_pdf(
                pdf_file,
                converter,
                chunker,
            )

            all_records.extend(records)

            logging.info(
                f"  -> {len(records)} chunks"
            )

        except Exception as error:
            logging.error(
                f"Lỗi khi xử lý {pdf_file.name}: {error}"
            )

        finally:
            gc.collect()

    output_file = CHUNKS_DIR / "chunks.json"

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_records,
            file,
            ensure_ascii=False,
            indent=2,
        )

    logging.info(
        f"Hoàn tất: {len(all_records)} chunks -> {output_file}"
    )

if __name__ == "__main__":
    run_preprocess()
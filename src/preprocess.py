from __future__ import annotations

import gc
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.chunking import HybridChunker
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

class LegalDocumentPreprocessor:
    def __init__(self):
        # 1. CẤU HÌNH THƯ MỤC
        self.project_root = Path(__file__).resolve().parent.parent
        self.raw_dir = self.project_root / "data" / "raw"
        self.extracted_dir = self.project_root / "data" / "extracted"
        self.chunks_dir = self.project_root / "data" / "chunks"
        self.reports_dir = self.project_root / "data" / "reports"

        self.chunks_file = self.chunks_dir / "chunks.json"
        self.report_file = self.reports_dir / "preprocess_report.json"
        self.preview_file = self.reports_dir / "chunks_preview.txt"

        # 2. CẤU HÌNH MODEL & CHUNKING
        self.embedding_model = "bkai-foundation-models/vietnamese-bi-encoder"
        self.max_chunk_tokens = 256 
        
        self.enable_ocr = False
        self.force_full_page_ocr = False
        self.min_chunk_chars = 35
        self.preview_chunk_count = 30
        self.max_files = None

        self._setup_directories()
        self.converter = self._create_converter()
        self.chunker = self._create_chunker()

    def _setup_directories(self):
        for folder in [self.raw_dir, self.extracted_dir, self.chunks_dir, self.reports_dir]:
            folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text)
        text = text.replace("\u00ad", "").replace("\u200b", "").replace("\ufeff", "").replace("\x00", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def normalized_key(text: str) -> str:
        text = LegalDocumentPreprocessor.clean_text(text).lower()
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def has_suspicious_characters(text: str) -> bool:
        return "\ufffd" in text or "" in text

    def _get_page_numbers(self, chunk: Any) -> list[int]:
        pages: set[int] = set()
        meta = getattr(chunk, "meta", None)
        for item in getattr(meta, "doc_items", []) or []:
            for provenance in getattr(item, "prov", []) or []:
                page_number = getattr(provenance, "page_no", None)
                if isinstance(page_number, int):
                    pages.add(page_number)
        return sorted(pages)

    def _get_headings(self, chunk: Any) -> list[str]:
        meta = getattr(chunk, "meta", None)
        headings = getattr(meta, "headings", []) or []
        cleaned_headings = []
        for heading in headings:
            value = self.clean_text(str(heading))
            if value:
                cleaned_headings.append(value)
        return cleaned_headings

    def _get_item_labels(self, chunk: Any) -> list[str]:
        labels: list[str] = []
        meta = getattr(chunk, "meta", None)
        for item in getattr(meta, "doc_items", []) or []:
            label = getattr(item, "label", None)
            if label is None:
                continue
            value = str(label).split(".")[-1].lower()
            if value not in labels:
                labels.append(value)
        return labels

    def _contextualize_chunk(self, chunk: Any) -> str:
        if hasattr(self.chunker, "contextualize"):
            return self.clean_text(self.chunker.contextualize(chunk=chunk))
        if hasattr(self.chunker, "serialize"):
            return self.clean_text(self.chunker.serialize(chunk))
        return self.clean_text(getattr(chunk, "text", ""))

    def _extract_legal_metadata(self, content: str, headings: list[str]) -> dict[str, str | None]:
        searchable_text = "\n".join([*headings, content])
        
        def first_match(pattern: str, text: str) -> str | None:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            return match.group(0).strip() if match else None

        chapter = first_match(r"\bChương\s+[IVXLCDM]+\b", searchable_text)
        section = first_match(r"\bMục\s+\d+\b", searchable_text)
        article = first_match(r"\bĐiều\s+\d+[a-zA-Z]?\b", searchable_text)

        clause_match = re.search(r"(?m)^\s*(\d+)\.\s+\S+", content)
        clause = f"Khoản {clause_match.group(1)}" if clause_match else None

        point_match = re.search(r"(?m)^\s*([a-zđ])\)\s+\S+", content, flags=re.IGNORECASE)
        point = f"Điểm {point_match.group(1).lower()}" if point_match else None

        return {
            "chapter": chapter,
            "section": section,
            "article": article,
            "clause": clause,
            "point": point,
        }

    def _extract_document_number(self, text: str, filename: str) -> str | None:
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

    def _should_keep_chunk(self, content: str, labels: list[str]) -> bool:
        if not content:
            return False

        ignored_labels = {"page_header", "page_footer"}
        if labels and set(labels).issubset(ignored_labels):
            return False

        if re.fullmatch(r"(?i)\s*(?:trang\s*)?[-–—]?\s*\d+\s*[-–—]?\s*", content):
            return False

        letter_count = sum(character.isalpha() for character in content)
        if letter_count < 8:
            return False

        contains_article = bool(re.search(r"\bĐiều\s+\d+[a-zA-Z]?\b", content, re.IGNORECASE))
        if len(content) < self.min_chunk_chars and not contains_article:
            return False

        return True

    def _make_chunk_id(self, source_file: str, pages: list[int], content: str) -> str:
        raw_value = f"{source_file}|{pages}|{self.normalized_key(content)}"
        digest = hashlib.sha1(raw_value.encode("utf-8")).hexdigest()[:14]
        return f"{Path(source_file).stem}_{digest}"

    def _create_converter(self) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions(do_ocr=self.enable_ocr, do_table_structure=True)
        pipeline_options.table_structure_options.mode = TableFormerMode.FAST
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.layout_batch_size = 1
        pipeline_options.table_batch_size = 1
        pipeline_options.ocr_batch_size = 1
        pipeline_options.queue_max_size = 4
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False
        pipeline_options.generate_parsed_pages = False
        pipeline_options.accelerator_options = AcceleratorOptions(num_threads=2, device=AcceleratorDevice.CPU)

        if self.enable_ocr and self.force_full_page_ocr:
            pipeline_options.ocr_options.force_full_page_ocr = True

        pdf_option = PdfFormatOption(pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend)
        return DocumentConverter(allowed_formats=[InputFormat.PDF], format_options={InputFormat.PDF: pdf_option})

    def _create_chunker(self) -> HybridChunker:
        raw_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        tokenizer = HuggingFaceTokenizer(tokenizer=raw_tokenizer, max_tokens=self.max_chunk_tokens)
        return HybridChunker(tokenizer=tokenizer, merge_peers=True)

    def process_pdf(self, pdf_file: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        print(f"\nĐang xử lý: {pdf_file.name}")

        result = self.converter.convert(source=str(pdf_file))
        document = result.document

        markdown_file = self.extracted_dir / f"{pdf_file.stem}.md"
        markdown_file.write_text(document.export_to_markdown(), encoding="utf-8")

        records: list[dict[str, Any]] = []
        seen_chunks: set[str] = set()
        skipped_short, skipped_duplicate = 0, 0

        chunk_iterator = self.chunker.chunk(dl_doc=document)

        for original_index, chunk in enumerate(chunk_iterator):
            content = self.clean_text(getattr(chunk, "text", ""))
            embedding_text = self._contextualize_chunk(chunk)
            pages = self._get_page_numbers(chunk)
            headings = self._get_headings(chunk)
            labels = self._get_item_labels(chunk)

            if not self._should_keep_chunk(content, labels):
                skipped_short += 1
                continue

            duplicate_key = self.normalized_key(content)
            if duplicate_key in seen_chunks:
                skipped_duplicate += 1
                continue
            seen_chunks.add(duplicate_key)

            legal_metadata = self._extract_legal_metadata(content, headings)

            record = {
                "chunk_id": self._make_chunk_id(pdf_file.name, pages, content),
                "content": content,
                "embedding_text": embedding_text,
                "source_file": pdf_file.name,
                "source_path": str(pdf_file.relative_to(self.project_root)),
                "document_title": headings[0] if headings else pdf_file.stem,
                "document_number": self._extract_document_number(embedding_text, pdf_file.name),
                "original_chunk_index": original_index,
                "headings": headings,
                "labels": labels,
                "pages": pages,
                "page_start": pages[0] if pages else None,
                "page_end": pages[-1] if pages else None,
                **legal_metadata,
                "character_count": len(content),
                "word_count": len(content.split()),
                "has_suspicious_characters": self.has_suspicious_characters(content),
            }
            records.append(record)

        missing_page_count = sum(1 for r in records if not r["pages"])
        suspicious_count = sum(1 for r in records if r["has_suspicious_characters"])

        report = {
            "source_file": pdf_file.name,
            "status": "success",
            "pdf_page_count": len(getattr(document, "pages", {}) or {}),
            "markdown_file": str(markdown_file.relative_to(self.project_root)),
            "kept_chunks": len(records),
            "skipped_short_or_noise": skipped_short,
            "skipped_duplicate": skipped_duplicate,
            "chunks_without_page_number": missing_page_count,
            "suspicious_chunks": suspicious_count,
        }

        print(f"  Giữ lại: {len(records)} chunk")
        print(f"  Thiếu số trang: {missing_page_count}")
        print(f"  Cần kiểm tra ký tự: {suspicious_count}")

        return records, report

    def write_preview(self, records: list[dict[str, Any]]):
        sections = []
        for index, record in enumerate(records[:self.preview_chunk_count], start=1):
            sections.append(
                "\n".join([
                    "=" * 80,
                    f"CHUNK {index}: {record['chunk_id']}",
                    f"Nguồn: {record['source_file']}",
                    f"Trang: {record['pages']}",
                    f"Headings: {record['headings']}",
                    f"Điều/Khoản: {record['article']} / {record['clause']}",
                    "-" * 80,
                    record["embedding_text"],
                ])
            )
        self.preview_file.write_text("\n\n".join(sections), encoding="utf-8")

    def run(self):
        pdf_files = sorted(self.raw_dir.rglob("*.pdf"))
        if self.max_files is not None:
            pdf_files = pdf_files[:self.max_files]

        if not pdf_files:
            raise FileNotFoundError(f"Không tìm thấy file PDF nào trong: {self.raw_dir}")

        print(f"Tìm thấy {len(pdf_files)} file PDF cần xử lý.")
        print(f"Embedding tokenizer: {self.embedding_model}")
        print(f"Giới hạn mỗi chunk: {self.max_chunk_tokens} tokens")

        all_records = []
        file_reports = []

        for pdf_file in pdf_files:
            try:
                records, report = self.process_pdf(pdf_file)
                all_records.extend(records)
                file_reports.append(report)
            except Exception as error:
                print(f"  Lỗi khi xử lý {pdf_file.name}: {error}")
                file_reports.append({"source_file": pdf_file.name, "status": "failed", "error": str(error)})
            finally:
                gc.collect()

        self.chunks_file.write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = {
            "embedding_model": self.embedding_model,
            "max_chunk_tokens": self.max_chunk_tokens,
            "ocr_enabled": self.enable_ocr,
            "processed_file_count": len(pdf_files),
            "successful_file_count": sum(r["status"] == "success" for r in file_reports),
            "failed_file_count": sum(r["status"] == "failed" for r in file_reports),
            "total_chunks": len(all_records),
            "files": file_reports,
        }

        self.report_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self.write_preview(all_records)

        print("\nHoàn thành preprocess.")
        print(f"Chunks: {self.chunks_file}")
        print(f"Report: {self.report_file}")
        print(f"Preview: {self.preview_file}")

if __name__ == "__main__":
    processor = LegalDocumentPreprocessor()
    processor.run()
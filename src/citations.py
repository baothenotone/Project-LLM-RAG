# Lấy nội dung của chunk
def get_chunk_content(item):
    return str(
        item.get("content")
        or item.get("text")
        or item.get("embedding_text")
        or ""
    ).strip()


# Lấy tên file PDF gốc
def get_file_name(item):
    file_name = (
        item.get("source_file")
        or item.get("file_name")
        or item.get("document_name")
        or "Không xác định"
    )

    return str(file_name).replace("\\", "/").rsplit("/", 1)[-1]


# Chuẩn hóa số trang để hiển thị
def format_pages(item):
    pages = item.get("pages")

    if isinstance(pages, (list, tuple, set)):
        pages = [
            str(page)
            for page in pages
            if page not in (None, "")
        ]

        return ", ".join(dict.fromkeys(pages)) or "Không xác định"

    return str(
        pages
        or item.get("page")
        or "Không xác định"
    )


# Định dạng phạm vi văn bản sửa đổi
def format_amendment_scope(item):
    return ", ".join(
        value
        for value in [
            item.get("amends_point"),
            item.get("amends_clause"),
            item.get("amends_article"),
        ]
        if value
    )


# Loại các kết quả không có nội dung
def prepare_sources(results):
    return [
        item
        for item in (results or [])
        if get_chunk_content(item)
    ]


# Tạo context có metadata pháp lý cho Gemini
def build_context(sources):
    contexts = []

    for index, item in enumerate(sources, start=1):
        lines = [
            f"[Nguồn {index}]",
            f"Tài liệu: {get_file_name(item)}",
            f"Số văn bản: {item.get('document_number') or 'Không xác định'}",
            f"Ngày hiệu lực: {item.get('effective_date') or 'Không xác định'}",
            f"Điều: {item.get('article') or 'Không xác định'}",
            f"Trang: {format_pages(item)}",
        ]

        if item.get("document_type") == "quyet_dinh_sua_doi":
            scope = format_amendment_scope(item)

            lines.extend([
                "Loại: Văn bản sửa đổi",
                f"Sửa văn bản: {item.get('amends_document') or 'Không xác định'}",
            ])

            if scope:
                lines.append(f"Phạm vi sửa đổi: {scope}")

        lines.append(
            f"Nội dung:\n{get_chunk_content(item)}"
        )

        contexts.append("\n".join(lines))

    return "\n\n".join(contexts)


# Định dạng một nguồn để hiển thị
def format_source(item, source_number):
    document = (
        item.get("document_number")
        or get_file_name(item)
    )

    parts = [f"**{source_number}. {document}**"]

    for key in ("point", "clause", "article"):
        if item.get(key):
            parts.append(item[key])

    if item.get("document_type") == "quyet_dinh_sua_doi":
        scope = format_amendment_scope(item)

        if scope:
            parts.append(f"Sửa: {scope}")

    parts.append(f"Trang {format_pages(item)}")

    return " - ".join(parts)
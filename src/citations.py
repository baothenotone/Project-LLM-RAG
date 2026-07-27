"""
Các hàm chuẩn hóa metadata và định dạng trích dẫn nguồn cho RAG.

File này không tìm kiếm tài liệu và cũng không sinh câu trả lời.
Nó nhận các chunk do Retriever trả về, sau đó:

1. Lấy nội dung, tên file, số trang và score.
2. Đánh số chunk theo dạng [Nguồn 1], [Nguồn 2], ...
3. Tạo context có nguồn để gửi cho Gemini.
4. In danh sách nguồn tham khảo trên Terminal.
"""


def get_chunk_content(item):
    """
    Lấy nội dung văn bản của một chunk.

    Nhiều tên trường được hỗ trợ để tương thích với metadata
    sinh ra từ các phiên bản preprocessing khác nhau.
    """
    if not isinstance(item, dict):
        return ""

    content = (
        item.get("content")
        or item.get("text")
        or item.get("chunk_text")
        or item.get("raw_text")
        or item.get("embedding_text")
        or ""
    )

    return str(content).strip()


def get_file_name(item):
    """
    Lấy tên file PDF gốc của chunk.

    Nếu metadata chỉ có đường dẫn trong trường source,
    hàm sẽ loại bỏ phần thư mục và chỉ giữ tên file.
    """
    if not isinstance(item, dict):
        return "Không xác định"

    file_name = (
        item.get("source_file")
        or item.get("file_name")
        or item.get("document_name")
        or item.get("source")
        or "Không xác định"
    )

    file_name = str(file_name).strip()

    if not file_name:
        return "Không xác định"

    return file_name.replace("\\", "/").rsplit("/", 1)[-1]


def format_pages(item):
    """
    Chuẩn hóa số trang thành chuỗi để hiển thị.

    Các dạng metadata được hỗ trợ:
    - pages: [2, 3]
    - page: 2
    - page_start: 2, page_end: 3
    """
    if not isinstance(item, dict):
        return "Không xác định"

    pages = item.get("pages")

    if isinstance(pages, (list, tuple, set)):
        valid_pages = []

        for page in pages:
            if page in (None, ""):
                continue

            page_text = str(page)

            if page_text not in valid_pages:
                valid_pages.append(page_text)

        if valid_pages:
            return ", ".join(valid_pages)

    if pages not in (None, ""):
        return str(pages)

    page = item.get("page")

    if page not in (None, ""):
        return str(page)

    page_start = item.get("page_start")
    page_end = item.get("page_end")

    if (
        page_start not in (None, "")
        and page_end not in (None, "")
    ):
        if page_start == page_end:
            return str(page_start)

        return f"{page_start}-{page_end}"

    if page_start not in (None, ""):
        return str(page_start)

    if page_end not in (None, ""):
        return str(page_end)

    return "Không xác định"


def get_score(item):
    """
    Lấy score cuối cùng của một kết quả retrieval.

    Nếu score bị thiếu hoặc không chuyển được sang float,
    hàm trả về 0.0.
    """
    if not isinstance(item, dict):
        return 0.0

    score = item.get("score", 0.0)

    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def build_context(results):
    """
    Tạo context có đánh số nguồn để gửi cho Gemini.

    Mỗi chunk hợp lệ có dạng:

    [Nguồn 1]
    Tài liệu: ten_file.pdf
    Trang: 6, 7
    Nội dung:
    ...
    """
    context_parts = []
    source_number = 1

    for item in results or []:
        content = get_chunk_content(item)

        if not content:
            continue

        file_name = get_file_name(item)
        pages = format_pages(item)

        context_part = (
            f"[Nguồn {source_number}]\n"
            f"Tài liệu: {file_name}\n"
            f"Trang: {pages}\n"
            f"Nội dung:\n{content}"
        )

        context_parts.append(context_part)
        source_number += 1

    return "\n\n".join(context_parts)


def format_source(item, source_number):
    """
    Định dạng một nguồn thành một dòng để hiển thị.
    """
    file_name = get_file_name(item)
    pages = format_pages(item)
    score = get_score(item)

    return (
        f"{source_number}. {file_name} - "
        f"trang {pages} - "
        f"score {score:.4f}"
    )


def print_sources(sources):
    """
    In danh sách nguồn do Retriever trả về trên Terminal.
    """
    if not sources:
        return

    print("\nNguồn tham khảo:")

    for source_number, item in enumerate(
        sources,
        start=1,
    ):
        print(
            format_source(
                item,
                source_number,
            )
        )
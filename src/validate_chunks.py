from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "chunks.json"
MAX_CHUNK_TOKENS = 120

KNOWN_BAD_TEXT = {
    "s tay học vụ": "Thiếu chữ 'ổ' trong 'sổ tay học vụ'",
    "Đại hoc Khoa học": "Thiếu dấu trong 'Đại học Khoa học'",
    "chi tính những học phần": "Thiếu dấu trong từ 'chỉ'",
    "xin nghi ốm": "Thiếu dấu trong từ 'nghỉ'",
    "giữ vũng kỷ cương": "Sai từ 'vững'",
    "thuộc\nế Đại học Huế": "Dòng bị chèn ký tự 'ế'",
}


def load_chunks() -> list[dict[str, Any]]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {CHUNKS_FILE}")

    data = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("chunks.json phải chứa một danh sách record")

    return data


def pages_are_contiguous(pages: list[int]) -> bool:
    return all(right - left <= 1 for left, right in zip(pages, pages[1:]))


def looks_like_glued_table(record: dict[str, Any]) -> bool:
    if "table" not in record.get("labels", []):
        return False

    content = record.get("content", "")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return True

    glued_boundaries = len(
        re.findall(
            r"(?<=[a-zà-ỹ])(?=[A-ZÀ-Ỵ])|"
            r"(?<=\d)(?=[A-Za-zÀ-ỹ])|"
            r"(?<=[A-Za-zÀ-ỹ])(?=\d)",
            content,
        )
    )

    return (
        len(content) > 120
        and len(lines) <= 2
        and max(len(line) for line in lines) >= 120
        and glued_boundaries >= 5
    )


def main() -> None:
    chunks = load_chunks()
    errors: list[str] = []
    warning_counter: Counter[str] = Counter()

    for record in chunks:
        chunk_id = record.get("chunk_id", "<không có id>")
        content = record.get("content", "")
        pages = record.get("pages", []) or []
        issues = set(record.get("quality_issues", []) or [])
        retrieval_enabled = bool(record.get("retrieval_enabled"))
        token_count = int(record.get("token_count", 0))

        for issue in issues:
            warning_counter[issue] += 1

        if not pages:
            errors.append(f"{chunk_id}: thiếu số trang")
        elif not pages_are_contiguous(pages):
            errors.append(f"{chunk_id}: trang không liên tục {pages}")

        if token_count > MAX_CHUNK_TOKENS:
            errors.append(
                f"{chunk_id}: {token_count} token, vượt {MAX_CHUNK_TOKENS}"
            )

        if retrieval_enabled and {
            "short_fragment",
            "unstructured_table",
            "possible_margin_line_number",
        }.intersection(issues):
            errors.append(
                f"{chunk_id}: vẫn bật retrieval dù có cảnh báo {sorted(issues)}"
            )

        if retrieval_enabled and looks_like_glued_table(record):
            errors.append(f"{chunk_id}: bảng dính chữ vẫn được bật retrieval")

        for bad_text, explanation in KNOWN_BAD_TEXT.items():
            if bad_text in content:
                errors.append(f"{chunk_id}: {explanation}")

    print(f"Tổng số chunk: {len(chunks)}")
    print(
        "Số chunk bật retrieval:",
        sum(bool(item.get("retrieval_enabled")) for item in chunks),
    )
    print("Cảnh báo:", dict(warning_counter))

    if errors:
        print(f"\nChưa đạt: tìm thấy {len(errors)} lỗi")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nĐạt kiểm tra cơ bản. Có thể chuyển sang kiểm thử retrieval.")


if __name__ == "__main__":
    main()
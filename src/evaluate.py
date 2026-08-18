import json
import time
import unicodedata
from pathlib import Path

from rag_pipeline import ask, init_rag_system


TEST_CASES = [
    {
        "question": "Tín chỉ là gì?",
        "answerable": True,
        "document": "673/QĐ-ĐHKH",
        "article": "Điều 6",
        "keywords": [
            "đơn vị quy chuẩn",
            "khối lượng học tập",
        ],
    },
    {
        "question": "Sinh viên bị buộc thôi học trong những trường hợp nào?",
        "answerable": True,
        "document": "673/QĐ-ĐHKH",
        "article": "Điều 20",
        "keywords": [
            "3 lần cảnh báo",
            "thời gian tối đa",
            "thi hộ",
        ],
    },
    {
        "question": "Sinh viên thi hộ lần đầu bị xử lý như thế nào?",
        "answerable": True,
        "document": "673/QĐ-ĐHKH",
        "article": "Điều 44",
        "keywords": [
            "đình chỉ học tập",
            "01 năm",
        ],
    },
    {
        "question": "Thi tự luận học phần 3 tín chỉ kéo dài bao lâu?",
        "answerable": True,
        "document": "673/QĐ-ĐHKH",
        "article": "Điều 32",
        "keywords": [
            "120 phút",
        ],
    },
    {
        "question": "Một ngày có bao nhiêu tiết học?",
        "answerable": True,
        "document": "1453/QĐ-ĐHKH",
        "article": None,
        "keywords": [
            "13 tiết",
        ],
    },
    {
        "question": "Tiết 13 học từ mấy giờ đến mấy giờ?",
        "answerable": True,
        "document": "1453/QĐ-ĐHKH",
        "article": None,
        "keywords": [
            "19 giờ 50",
            "20 giờ 40",
        ],
    },
    {
        "question": "Trường có bán đồ ăn trưa cho sinh viên không?",
        "answerable": False,
    },
    {
        "question": "Messi ghi bao nhiêu bàn thắng?",
        "answerable": False,
    },
    {
        "question": "xếp loại xuất sắc cần bao nhiêu điểm",
        "answerable": True,
        "document": "673/QĐ-ĐHKH",
        "article": None,
        "keywords": [
            "3.6",
            "4.0",
        ],
    },
    {
        "question": "Tiết 6",
        "answerable": True,
        "document": "1453/QĐ-ĐHKH",
        "article": None,
        "keywords": [
            "13 giờ 00",
            "13 giờ 50",
        ],
    },
]


# Chuẩn hóa văn bản để so sánh từ khóa

def normalize(text):
    text = unicodedata.normalize(
        "NFC",
        str(text),
    ).lower()

    return " ".join(
        text.split()
    )


# Tìm thứ hạng của nguồn đúng trong kết quả

def find_expected_rank(sources, test_case):
    document = test_case.get("document")
    article = test_case.get("article")

    for rank, source in enumerate(
        sources,
        start=1,
    ):
        if (
            document
            and source.get("document_number") != document
        ):
            continue

        if (
            article
            and source.get("article") != article
        ):
            continue

        return rank

    return None


# Kiểm tra câu trả lời có chứa các ý chính

def check_answer(answer, keywords):
    answer = normalize(answer)

    return all(
        normalize(keyword) in answer
        for keyword in keywords
    )


# Đánh giá retrieval, answer, OOD và thời gian

def run_evaluation():
    client, model_name, resources = init_rag_system()

    report_dir = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "reports"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file = (
        report_dir / "evaluation_report.json"
    )

    details = []

    retrieval_hits = 0
    reciprocal_rank_sum = 0.0
    answer_hits = 0
    ood_hits = 0

    in_domain_count = 0
    ood_count = 0
    total_time = 0.0

    for index, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):
        start = time.time()

        result = ask(
            test_case["question"],
            client,
            model_name,
            resources,
            top_k=3,
        )

        elapsed = time.time() - start
        total_time += elapsed

        answer = result["answer"]
        sources = result["sources"]

        if test_case["answerable"]:
            in_domain_count += 1

            rank = find_expected_rank(
                sources,
                test_case,
            )

            retrieval_ok = rank is not None
            answer_ok = check_answer(
                answer,
                test_case.get("keywords", []),
            )

            if retrieval_ok:
                retrieval_hits += 1
                reciprocal_rank_sum += 1 / rank

            if answer_ok:
                answer_hits += 1

            passed = retrieval_ok and answer_ok

        else:
            ood_count += 1
            rank = None
            retrieval_ok = len(sources) == 0
            answer_ok = retrieval_ok
            passed = retrieval_ok

            if passed:
                ood_hits += 1

        top_score = (
            round(sources[0]["score"], 4)
            if sources
            else None
        )

        details.append(
            {
                "test_id": index,
                "question": test_case["question"],
                "passed": passed,
                "source_rank": rank,
                "retrieval_ok": retrieval_ok,
                "answer_ok": answer_ok,
                "top_score": top_score,
                "answer": answer,
                "time_seconds": round(elapsed, 2),
            }
        )

        status = "PASS" if passed else "FAIL"

        print(
            f"[{index}/{len(TEST_CASES)}] "
            f"{status} - {test_case['question']}"
        )
        print(f"  Top score: {top_score}")
        print(f"  Time: {elapsed:.2f}s")

    summary = {
        "retrieval_hit_at_3": round(
            retrieval_hits / in_domain_count,
            4,
        ),
        "mrr": round(
            reciprocal_rank_sum / in_domain_count,
            4,
        ),
        "answer_accuracy": round(
            answer_hits / in_domain_count,
            4,
        ),
        "ood_accuracy": round(
            ood_hits / ood_count,
            4,
        ),
        "average_time_seconds": round(
            total_time / len(TEST_CASES),
            2,
        ),
    }

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "summary": summary,
                "details": details,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nKết quả:")

    for key, value in summary.items():
        print(f"- {key}: {value}")

    print(
        f"\nĐã lưu báo cáo: {report_file}"
    )


if __name__ == "__main__":
    run_evaluation()
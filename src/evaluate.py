import json
import logging
import time
from pathlib import Path
from rag_pipeline import init_rag_system, ask

# Danh sách các câu hỏi để test đa dạng các trường hợp
TEST_CASES = [
    # --- 1. Câu hỏi định nghĩa cơ bản ---
    {
        "question": "Tín chỉ là gì?", 
        "expect_found": True
    },
    
    # --- 2. Câu hỏi điều kiện, quy định phức tạp ---
    {
        "question": "Sinh viên bị buộc thôi học trong những trường hợp nào?", 
        "expect_found": True
    },
    {
        "question": "Điều kiện để được xét làm đồ án, khóa luận tốt nghiệp là gì?", 
        "expect_found": True
    },
    {
        "question": "Sinh viên thi hộ thì bị xử lý kỷ luật như thế nào?", 
        "expect_found": True
    },
    
    # --- 3. Câu hỏi về thời gian, con số cụ thể ---
    {
        "question": "Một tiết học kéo dài bao nhiêu phút?", 
        "expect_found": True
    },
    {
        "question": "Thời gian làm bài thi tự luận đối với học phần 3 tín chỉ là bao lâu?", 
        "expect_found": True
    },
    
    # --- 4. Câu hỏi ngoài luồng (Out of domain) ---
    # Kỳ vọng hệ thống phải trả lời là "Không tìm thấy thông tin"
    {
        "question": "Trường có bán đồ ăn trưa cho sinh viên không?", 
        "expect_found": False
    },
    {
        "question": "Lịch thi lại môn Triết học Mác Lênin khi nào có?", 
        "expect_found": False
    }
]

def run_detailed_evaluation():
    print("Đang khởi tạo hệ thống RAG để đánh giá chi tiết...")
    client, model_name, resources = init_rag_system()
    
    # Chuẩn bị thư mục và file lưu báo cáo
    report_dir = Path(__file__).resolve().parent.parent / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / "detailed_evaluation_report.json"
    
    evaluation_results = []
    success_count = 0
    total_time = 0
    
    print("\n" + "="*80)
    print("BẮT ĐẦU ĐÁNH GIÁ TỰ ĐỘNG HỆ THỐNG RAG")
    print("="*80)
    
    for index, test_case in enumerate(TEST_CASES, start=1):
        question = test_case["question"]
        expect_found = test_case["expect_found"]
        
        print(f"\n[Test {index}/{len(TEST_CASES)}] Câu hỏi: {question}")
        
        # Bắt đầu đo thời gian
        start_time = time.time()
        
        # Gọi RAG pipeline (Lấy top 3 tài liệu tốt nhất)
        result = ask(question, client, model_name, resources, top_k=3)
        
        # Kết thúc đo thời gian
        end_time = time.time()
        execution_time = end_time - start_time
        total_time += execution_time
        
        answer = result['answer']
        sources = result['sources']
        
        # Kiểm tra xem AI có từ chối trả lời (không tìm thấy) hay không
        not_found_keywords = ["không tìm thấy", "không đề cập", "tài liệu hiện tại không"]
        is_not_found_response = any(kw in answer.lower() for kw in not_found_keywords)
        
        # Thực tế hệ thống có lấy được nguồn để trả lời hay không
        actual_found = (len(sources) > 0) and not is_not_found_response
        
        # Chấm điểm (PASS / FAIL): Nếu kỳ vọng = thực tế thì là PASS
        is_pass = (actual_found == expect_found)
        if is_pass:
            success_count += 1
            status_icon = "✅ PASS"
        else:
            status_icon = "❌ FAIL"
            
        # Lấy thông tin tài liệu tốt nhất (Top 1) để in ra log
        top_source_info = "Không có"
        if len(sources) > 0:
            top_doc = sources[0]
            file_name = top_doc.get("source_file", "Không rõ")
            pages = ", ".join(map(str, top_doc.get("pages", [])))
            score = top_doc.get("score", 0)
            top_source_info = f"{file_name} (Trang {pages}) - Score: {score:.4f}"
        
        # In chi tiết ra terminal
        print(f"  Trạng thái    : {status_icon}")
        print(f"  Thời gian chạy: {execution_time:.2f} giây")
        print(f"  Kỳ vọng có KQ : {'Có' if expect_found else 'Không'}")
        print(f"  Thực tế có KQ : {'Có' if actual_found else 'Không'}")
        print(f"  Nguồn tốt nhất: {top_source_info}")
        
        # Rút gọn câu trả lời nếu quá dài để terminal dễ nhìn
        short_answer = answer[:150] + "..." if len(answer) > 150 else answer
        print(f"  Câu trả lời   : {short_answer}")
        
        # Lưu dữ liệu vào mảng để xuất JSON
        evaluation_results.append({
            "test_id": index,
            "question": question,
            "expected_found": expect_found,
            "actual_found": actual_found,
            "is_pass": is_pass,
            "execution_time_seconds": round(execution_time, 2),
            "retrieved_sources_count": len(sources),
            "top_source": top_source_info,
            "answer": answer
        })

    # In tổng kết
    hit_rate = (success_count / len(TEST_CASES)) * 100
    avg_time = total_time / len(TEST_CASES)
    
    print("\n" + "="*80)
    print("KẾT QUẢ ĐÁNH GIÁ TỔNG QUAN:")
    print(f"- Số test thành công (PASS) : {success_count}/{len(TEST_CASES)}")
    print(f"- Tỷ lệ chính xác (Accuracy): {hit_rate:.2f}%")
    print(f"- Thời gian phản hồi TB     : {avg_time:.2f} giây/câu hỏi")
    print("="*80 + "\n")
    
    # Lưu file báo cáo JSON
    report_data = {
        "summary": {
            "total_tests": len(TEST_CASES),
            "passed_tests": success_count,
            "accuracy_percent": round(hit_rate, 2),
            "average_time_seconds": round(avg_time, 2),
        },
        "detailed_results": evaluation_results
    }
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=4)
        
    print(f"Đã lưu báo cáo chi tiết đầy đủ câu trả lời tại: {report_file}")

if __name__ == "__main__":
    run_detailed_evaluation()
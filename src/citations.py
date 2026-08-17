"""
Các hàm chuẩn hóa metadata và định dạng trích dẫn nguồn cho RAG.
File này chuyên xử lý việc bóc tách tên file, số trang, và tạo context cho LLM.
"""

def get_chunk_content(item):
    """Lấy nội dung văn bản của một chunk."""
    # Kiểm tra xem item có phải là dictionary không, nếu không thì trả về chuỗi rỗng
    if not isinstance(item, dict):
        return ""
    
    # Ưu tiên tìm nội dung trong các trường dữ liệu theo thứ tự
    content = item.get("content")
    
    if not content:
        content = item.get("text")
        
    if not content:
        content = item.get("embedding_text")
        
    # Nếu không tìm thấy nội dung nào thì gán bằng chuỗi rỗng
    if content is None:
        content = ""
        
    # Chuyển đổi thành chuỗi và xóa các khoảng trắng thừa ở 2 đầu
    return str(content).strip()

def get_file_name(item):
    """Lấy tên file PDF gốc của chunk, tự động loại bỏ đường dẫn thư mục rườm rà."""
    if not isinstance(item, dict):
        return "Không xác định"
        
    # Cố gắng lấy tên file từ các key phổ biến
    file_name = item.get("source_file")
    
    if not file_name:
        file_name = item.get("file_name")
        
    if not file_name:
        file_name = item.get("document_name")
        
    # Nếu không tìm thấy key nào hợp lệ
    if not file_name:
        return "Không xác định"
        
    # Xóa khoảng trắng thừa
    file_name = str(file_name).strip()
    
    # Xử lý đường dẫn: Đổi dấu gạch chéo ngược (Windows) thành gạch chéo xuôi (Linux/Mac)
    file_name = file_name.replace("\\", "/")
    
    # Cắt chuỗi theo dấu gạch chéo và chỉ lấy phần tử cuối cùng (tức là tên file)
    # Ví dụ: "data/raw/quy_che.pdf" -> "quy_che.pdf"
    file_name = file_name.rsplit("/", 1)[-1]
    
    return file_name

def format_pages(item):
    """Chuẩn hóa số trang thành chuỗi để giao diện hiển thị đẹp mắt."""
    if not isinstance(item, dict):
        return "Không xác định"

    # Trường hợp 1: Metadata chứa danh sách các trang (ví dụ: [1, 2, 3])
    pages = item.get("pages")
    if isinstance(pages, (list, tuple, set)):
        valid_pages = []
        for page in pages:
            # Bỏ qua nếu trang bị rỗng
            if page in (None, ""):
                continue
                
            page_str = str(page)
            # Kiểm tra để tránh hiển thị trùng lặp số trang
            if page_str not in valid_pages:
                valid_pages.append(page_str)
                
        # Nếu có trang hợp lệ, nối chúng bằng dấu phẩy
        if len(valid_pages) > 0:
            return ", ".join(valid_pages)

    # Trường hợp 2: Metadata lưu thẳng 1 số trang vào biến 'pages' hoặc 'page'
    if pages not in (None, ""):
        return str(pages)

    page = item.get("page")
    if page not in (None, ""):
        return str(page)

    # Trường hợp 3: Metadata lưu trang bắt đầu và trang kết thúc
    page_start = item.get("page_start")
    page_end = item.get("page_end")
    
    if page_start not in (None, "") and page_end not in (None, ""):
        # Nếu trang đầu và cuối giống nhau thì chỉ cần in 1 số
        if page_start == page_end:
            return str(page_start)
        else:
            return f"{page_start}-{page_end}"
            
    # Nếu chỉ có trang bắt đầu hoặc trang kết thúc
    if page_start not in (None, ""): 
        return str(page_start)
        
    if page_end not in (None, ""): 
        return str(page_end)

    # Nếu tất cả các trường hợp trên đều thất bại
    return "Không xác định"

def build_context(results):
    """
    Tạo một chuỗi văn bản lớn (context) chứa tất cả các nguồn tài liệu, 
    có đánh số thứ tự để gửi cho AI (Gemini) đọc và tham khảo.
    """
    context_parts = []
    source_number = 1

    # Kiểm tra an toàn, nếu results rỗng thì trả về chuỗi rỗng
    if not results:
        return ""

    # Duyệt qua từng kết quả tìm kiếm được
    for item in results:
        content = get_chunk_content(item)
        
        # Nếu đoạn văn bản rỗng thì bỏ qua, không đưa vào context
        if not content:
            continue

        file_name = get_file_name(item)
        pages = format_pages(item)

        # Định dạng khối văn bản cho từng nguồn
        context_part = (
            f"[Nguồn {source_number}]\n"
            f"Tài liệu: {file_name}\n"
            f"Trang: {pages}\n"
            f"Nội dung:\n{content}"
        )
        
        # Thêm khối văn bản vào danh sách và tăng số đếm
        context_parts.append(context_part)
        source_number += 1

    # Nối tất cả các khối văn bản lại, cách nhau bởi 2 dấu xuống dòng
    return "\n\n".join(context_parts)

def format_source(item, source_number):
    """Định dạng một nguồn thành một dòng duy nhất để in ra Terminal."""
    file_name = get_file_name(item)
    pages = format_pages(item)
    return f"{source_number}. {file_name} - trang {pages}"

def print_sources(sources):
    """In danh sách nguồn do Retriever trả về ra màn hình Terminal (dùng để kiểm thử)."""
    if not sources:
        return
        
    print("\nNguồn tham khảo:")
    for index, item in enumerate(sources, start=1):
        print(format_source(item, index))
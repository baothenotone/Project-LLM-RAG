import streamlit as st
from rag_pipeline import init_rag_system, ask

# Chỉ import các hàm bóc tách dữ liệu cần thiết từ citations.py
from citations import get_file_name, format_pages

# Cấu hình hiển thị trang web tràn viền
st.set_page_config(page_title="RAG - Quy chế Đào tạo", layout="wide")

# Sử dụng cache để không phải load lại database mỗi khi giao diện cập nhật
@st.cache_resource
def load_system():
    return init_rag_system()

st.title("📚 Trợ lý Tra cứu Quy chế Đào tạo HUSC")

# Khởi tạo các tài nguyên (client AI, mô hình, dữ liệu vector)
client, model_name, resources = load_system()

# Lưu trữ lịch sử chat trong session_state của Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. Vẽ lại toàn bộ tin nhắn từ lịch sử chat lên màn hình
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Nếu tin nhắn của AI có đính kèm nguồn tài liệu, thì vẽ thêm phần nguồn
        if "sources" in msg and len(msg["sources"]) > 0:
            with st.expander("🔍 Xem nguồn tài liệu tham khảo"):
                for index, src in enumerate(msg["sources"], start=1):
                    file_name = get_file_name(src)
                    pages = format_pages(src)
                    st.write(f"**{index}. {file_name}** (Trang {pages})")

# 2. Xử lý khi người dùng nhập câu hỏi mới
user_question = st.chat_input("Nhập câu hỏi (Ví dụ: Học cải thiện điểm được quy định như thế nào?)...")

if user_question:
    # Lưu câu hỏi vào lịch sử và hiển thị lên màn hình
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 3. Quá trình AI suy nghĩ và trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm trong quy chế và tổng hợp thông tin..."):
            # Gọi hệ thống RAG để tìm câu trả lời
            result = ask(user_question, client, model_name, resources)
            answer = result["answer"]
            sources = result["sources"]
            
        # In câu trả lời ra màn hình
        st.markdown(answer)
        
        # In danh sách nguồn (nếu có)
        if len(sources) > 0:
            with st.expander("🔍 Xem nguồn tài liệu tham khảo"):
                for index, src in enumerate(sources, start=1):
                    file_name = get_file_name(src)
                    pages = format_pages(src)
                    st.write(f"**{index}. {file_name}** (Trang {pages})")
                    
    # Lưu kết quả của AI vào lịch sử chat để lần sau vẽ lại
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer, 
        "sources": sources
    })
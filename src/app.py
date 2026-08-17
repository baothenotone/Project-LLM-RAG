import streamlit as st
from rag_pipeline import init_rag_system, ask

st.set_page_config(page_title="RAG - Quy chế Đào tạo", layout="wide")

# Cache hệ thống để không phải tải lại các Model nặng mỗi khi người dùng chat
@st.cache_resource
def load_system():
    return init_rag_system()

st.title("📚 Trợ lý Tra cứu Quy chế Đào tạo HUSC")
client, model_name, resources = load_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Vẽ lại các tin nhắn cũ
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Nếu AI trả lời thì vẽ luôn nguồn tham khảo
        if "sources" in msg and len(msg["sources"]) > 0:
            with st.expander("🔍 Xem nguồn tài liệu tham khảo"):
                for index, src in enumerate(msg["sources"], start=1):
                    file_name = src.get("source_file", "")
                    pages = ", ".join(map(str, src.get("pages", [])))
                    score = src.get("score", 0)
                    st.write(f"**{index}. {file_name}** (Trang {pages}) - Độ chính xác ước tính: {score:.4f}")

# Ô nhập chat
user_question = st.chat_input("Nhập câu hỏi (Ví dụ: Học cải thiện điểm được quy định như thế nào?)...")

if user_question:
    # 1. In câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 2. In câu trả lời của AI
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm trong quy chế và tổng hợp thông tin..."):
            result = ask(user_question, client, model_name, resources)
            answer = result["answer"]
            sources = result["sources"]
            
        st.markdown(answer)
        if len(sources) > 0:
            with st.expander("🔍 Xem nguồn tài liệu tham khảo"):
                for index, src in enumerate(sources, start=1):
                    file_name = src.get("source_file", "")
                    pages = ", ".join(map(str, src.get("pages", [])))
                    st.write(f"**{index}. {file_name}** (Trang {pages})")
                    
    # Lưu vào lịch sử
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer, 
        "sources": sources
    })
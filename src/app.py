import streamlit as st
from rag_pipeline import RAGPipeline
from citations import get_file_name, format_pages

st.set_page_config(page_title="Trợ lý Quy chế Đào tạo", layout="wide")

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

st.title("🤖 Trợ lý tra cứu Quy chế Đào tạo")
pipeline = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Nguồn tham khảo"):
                for i, src in enumerate(msg["sources"], 1):
                    st.write(f"**{i}. {get_file_name(src)}** (Trang {format_pages(src)}) - Score: {src.get('score', 0):.4f}")

# Khung nhập câu hỏi
if prompt := st.chat_input("Nhập câu hỏi của bạn (VD: Tín chỉ là gì?)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu tài liệu và suy nghĩ..."):
            result = pipeline.ask(prompt)
            answer = result["answer"]
            sources = result["sources"]
            
            st.markdown(answer)
            if sources:
                with st.expander("📚 Nguồn tham khảo"):
                    for i, src in enumerate(sources, 1):
                        st.write(f"**{i}. {get_file_name(src)}** (Trang {format_pages(src)})")
            
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
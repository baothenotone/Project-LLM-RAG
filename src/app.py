import re
from pathlib import Path

import streamlit as st

from citations import format_source
from rag_pipeline import ask, init_rag_system


st.set_page_config(
    page_title="Trợ lý Quy chế Đào tạo HUSC",
    page_icon="🎓",
    layout="wide",
)


# Đường dẫn logo
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGO_PATH = PROJECT_ROOT / "logo.png"


# Tải hệ thống RAG một lần
@st.cache_resource
def load_system():
    return init_rag_system()


# Lấy các nguồn thực sự được trích dẫn trong câu trả lời
def get_cited_sources(answer, sources):
    cited_numbers = []

    for citation in re.findall(
        r"\[Nguồn\s+([^\]]+)\]",
        answer,
        re.IGNORECASE,
    ):
        for number in re.findall(r"\d+", citation):
            number = int(number)

            if (
                1 <= number <= len(sources)
                and number not in cited_numbers
            ):
                cited_numbers.append(number)

    return [
        (number, sources[number - 1])
        for number in cited_numbers
    ]


# Hiển thị các nguồn đã được sử dụng
def show_sources(answer, sources):
    cited_sources = get_cited_sources(
        answer,
        sources,
    )

    if not cited_sources:
        return

    with st.expander(
        "🔍 Xem nguồn tài liệu tham khảo"
    ):
        for number, source in cited_sources:
            st.markdown(
                format_source(
                    source,
                    number,
                )
            )


# ==========================================================
# LOGO + TÊN TRƯỜNG
# ==========================================================

logo_col, title_col = st.columns(
    [1, 8],
    vertical_alignment="center",
)

with logo_col:
    if LOGO_PATH.exists():
        st.image(
            str(LOGO_PATH),
            width=100,
        )
    else:
        st.markdown("### 🎓 HUSC")


with title_col:
    st.markdown(
        "### TRƯỜNG ĐẠI HỌC KHOA HỌC, ĐẠI HỌC HUẾ"
    )

    st.title(
        "📚 Trợ lý Tra cứu Quy chế Đào tạo"
    )

    st.caption(
        "Hỗ trợ sinh viên tra cứu thông tin "
        "trong Quy chế đào tạo của Nhà trường."
    )


st.divider()


# ==========================================================
# KHỞI TẠO HỆ THỐNG RAG
# ==========================================================

client, model_name, resources = load_system()


# Khởi tạo lịch sử hội thoại
if "messages" not in st.session_state:
    st.session_state.messages = []


# Hiển thị lịch sử hội thoại
for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        if message["role"] == "assistant":
            show_sources(
                message["content"],
                message.get(
                    "sources",
                    [],
                ),
            )


# ==========================================================
# Ô NHẬP CÂU HỎI
# ==========================================================

user_question = st.chat_input(
    "Nhập câu hỏi về quy chế đào tạo..."
)


# ==========================================================
# XỬ LÝ CÂU HỎI
# ==========================================================

if user_question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user"):
        st.markdown(
            user_question
        )

    with st.chat_message("assistant"):
        with st.spinner(
            "Đang tìm kiếm trong quy chế..."
        ):
            result = ask(
                user_question,
                client,
                model_name,
                resources,
            )

            answer = result["answer"]
            sources = result["sources"]

        st.markdown(answer)

        show_sources(
            answer,
            sources,
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
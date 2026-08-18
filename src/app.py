import re

import streamlit as st

from citations import format_source
from rag_pipeline import ask, init_rag_system


st.set_page_config(
    page_title="RAG - Quy chế Đào tạo",
    layout="wide",
)


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


st.title(
    "📚 Trợ lý Tra cứu Quy chế Đào tạo HUSC"
)

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


user_question = st.chat_input(
    "Nhập câu hỏi về quy chế đào tạo..."
)


# Xử lý câu hỏi mới
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
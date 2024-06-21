import datetime
import json

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage, FunctionMessage
from streamlit.delta_generator import DeltaGenerator

from backend.chat_bot.json_decoder import CustomJSONDecoder
from backend.constants.streamlit_keys import CHAT_CURRENT_USER_SESSIONS, EL_SESSION_SELECTOR, \
    EL_UPLOAD_FILES_STATUS, USER_PRIVATE_FILES, EL_BUILD_KB_WITH_FILES, \
    EL_PERSONAL_KB_NAME, EL_PERSONAL_KB_DESCRIPTION, \
    USER_PERSONAL_KNOWLEDGE_BASES, AVAILABLE_RETRIEVAL_TOOLS, EL_PERSONAL_KB_NEEDS_REMOVE, \
    CHAT_KNOWLEDGE_TABLE, EL_UPLOAD_FILES
from backend.constants.variables import DIVIDER_HTML, USER_NAME, RETRIEVER_TOOLS
from backend.construct.build_chat_bot import build_chat_knowledge_table, initialize_session_manager
from backend.chat_bot.chat import refresh_sessions, on_session_change_submit, refresh_agent, \
    create_private_knowledge_base_as_tool, \
    remove_private_knowledge_bases, add_file, clear_files, clear_history, back_to_main, on_chat_submit


def render_session_manager():
    with st.expander("🤖 Session Management"):
        if CHAT_CURRENT_USER_SESSIONS not in st.session_state:
            refresh_sessions()
        st.markdown("Here you can update `session_id` and `system_prompt`")
        st.markdown("- Click empty row to add a new item")
        st.markdown("- If needs to delete an item, just click it and press `DEL` key")
        st.markdown("- Don't forget to submit your change.")

        st.data_editor(
            data=st.session_state[CHAT_CURRENT_USER_SESSIONS],
            num_rows="dynamic",
            key="session_editor",
            use_container_width=True,
        )
        st.button("⏫ Submit", on_click=on_session_change_submit, type="primary")


def render_session_selection():
    with st.expander("✅ Session Selection", expanded=True):
        st.selectbox(
            "Choose a `session` to chat",
            options=st.session_state[CHAT_CURRENT_USER_SESSIONS],
            index=None,
            key=EL_SESSION_SELECTOR,
            format_func=lambda x: x["session_id"],
            on_change=refresh_agent,
        )


def render_files_manager():
    with st.expander("📃 **Upload your personal files**", expanded=False):
        st.markdown("- Files will be parsed by [Unstructured API](https://unstructured.io/api-key).")
        st.markdown("- All files will be converted into vectors and stored in [MyScaleDB](https://myscale.com/).")
        st.file_uploader(label="⏫ **Upload files**", key=EL_UPLOAD_FILES, accept_multiple_files=True)
        # st.markdown("### Uploaded Files")
        st.dataframe(
            data=st.session_state[CHAT_KNOWLEDGE_TABLE].list_files(st.session_state[USER_NAME]),
            use_container_width=True,
        )
        st.session_state[EL_UPLOAD_FILES_STATUS] = st.empty()
        col_1, col_2 = st.columns(2)
        with col_1:
            st.button(label="Upload files", on_click=add_file)
        with col_2:
            st.button(label="Clear all files and tools", on_click=clear_files)


def _render_create_personal_knowledge_bases(div: DeltaGenerator):
    with div:
        st.markdown("- If you haven't upload your personal files, please upload them first.")
        st.markdown("- Select some **files** to build your `personal knowledge base`.")
        st.markdown("- Once the your `personal knowledge base` is built, "
                    "it will answer your questions using information from your personal **files**.")
        st.multiselect(
            label="⚡️Select some files to build a **personal knowledge base**",
            options=st.session_state[USER_PRIVATE_FILES],
            placeholder="You should upload some files first",
            key=EL_BUILD_KB_WITH_FILES,
            format_func=lambda x: x["file_name"],
        )
        st.text_input(
            label="⚡️Personal knowledge base name",
            value="get_relevant_documents",
            key=EL_PERSONAL_KB_NAME
        )
        st.text_input(
            label="⚡️Personal knowledge base description",
            value="Searches from some personal files.",
            key=EL_PERSONAL_KB_DESCRIPTION,
        )
        st.button(
            label="Build 🔧",
            on_click=create_private_knowledge_base_as_tool
        )


def _render_remove_personal_knowledge_bases(div: DeltaGenerator):
    with div:
        st.markdown("> Here is all your personal knowledge bases.")
        if USER_PERSONAL_KNOWLEDGE_BASES in st.session_state and len(st.session_state[USER_PERSONAL_KNOWLEDGE_BASES]) > 0:
            st.dataframe(st.session_state[USER_PERSONAL_KNOWLEDGE_BASES])
        else:
            st.warning("You don't have any personal knowledge bases, please create a new one.")
        st.multiselect(
            label="Choose a personal knowledge base to delete",
            placeholder="Choose a personal knowledge base to delete",
            options=st.session_state[USER_PERSONAL_KNOWLEDGE_BASES],
            format_func=lambda x: x["tool_name"],
            key=EL_PERSONAL_KB_NEEDS_REMOVE,
        )
        st.button("Delete", on_click=remove_private_knowledge_bases, type="primary")


def render_personal_tools_build():
    with st.expander("🔨 **Build your personal knowledge base**", expanded=True):
        create_new_kb, kb_manager = st.tabs(["Create personal knowledge base", "Personal knowledge base management"])
        _render_create_personal_knowledge_bases(create_new_kb)
        _render_remove_personal_knowledge_bases(kb_manager)


def render_knowledge_base_selector():
    with st.expander("🙋 **Select some knowledge bases to query**", expanded=True):
        st.markdown("- Knowledge bases come in two types: `public` and `private`.")
        st.markdown("- All users can access our `public` knowledge bases.")
        st.markdown("- Only you can access your `personal` knowledge bases.")
        options = st.session_state[RETRIEVER_TOOLS].keys()
        if AVAILABLE_RETRIEVAL_TOOLS in st.session_state:
            options = st.session_state[AVAILABLE_RETRIEVAL_TOOLS]
        st.multiselect(
            label="Select some knowledge base tool",
            placeholder="Please select some knowledge bases to query",
            options=options,
            default=["Wikipedia + Self Querying"],
            key="selected_tools",
            on_change=refresh_agent,
        )


def chat_page():
    # initialize resources
    build_chat_knowledge_table()
    initialize_session_manager()

    # render sidebar
    with st.sidebar:
        left, middle, right = st.columns([1, 1, 2])
        with left:
            st.button(label="↩️ Log Out", help="log out and back to main page", on_click=back_to_main)
        with right:
            st.markdown(f"👤 `{st.session_state[USER_NAME]}`")
        st.markdown(DIVIDER_HTML, unsafe_allow_html=True)
        render_session_manager()
        render_session_selection()
        render_files_manager()
        render_personal_tools_build()
        render_knowledge_base_selector()

    # render chat history
    if "agent" not in st.session_state:
        refresh_agent()
    for msg in st.session_state.agent.memory.chat_memory.messages:
        speaker = "user" if isinstance(msg, HumanMessage) else "assistant"
        if isinstance(msg, FunctionMessage):
            with st.chat_message(name="from knowledge base", avatar="📚"):
                st.write(
                    f"*{datetime.datetime.fromtimestamp(msg.additional_kwargs['timestamp']).isoformat()}*"
                )
                st.write("Retrieved from knowledge base:")
                try:
                    st.dataframe(
                        pd.DataFrame.from_records(
                            json.loads(msg.content, cls=CustomJSONDecoder)
                        ),
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(e)
                    st.write(msg.content)
        else:
            if len(msg.content) > 0:
                with st.chat_message(speaker):
                    print(type(msg), msg.dict())
                    st.write(
                        f"*{datetime.datetime.fromtimestamp(msg.additional_kwargs['timestamp']).isoformat()}*"
                    )
                    st.write(f"{msg.content}")
    st.session_state["next_round"] = st.empty()
    from streamlit import _bottom
    with _bottom:
        col1, col2 = st.columns([1, 16])
        with col1:
            st.button("🗑️", help="Clean chat history", on_click=clear_history, type="secondary")
        with col2:
            st.chat_input("Input Message", on_submit=on_chat_submit, key="chat_input")

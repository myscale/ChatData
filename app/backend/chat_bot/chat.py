import time

from os import environ
from time import sleep
import streamlit as st

from backend.constants.prompts import DEFAULT_SYSTEM_PROMPT
from backend.constants.streamlit_keys import CHAT_KNOWLEDGE_TABLE, CHAT_SESSION_MANAGER, \
    CHAT_CURRENT_USER_SESSIONS, EL_SESSION_SELECTOR, USER_PRIVATE_FILES, \
    EL_BUILD_KB_WITH_FILES, \
    EL_PERSONAL_KB_NAME, EL_PERSONAL_KB_DESCRIPTION, \
    USER_PERSONAL_KNOWLEDGE_BASES, AVAILABLE_RETRIEVAL_TOOLS, EL_PERSONAL_KB_NEEDS_REMOVE, \
    EL_UPLOAD_FILES_STATUS, EL_SELECTED_KBS, EL_UPLOAD_FILES
from backend.constants.variables import USER_INFO, USER_NAME, JUMP_QUERY_ASK, RETRIEVER_TOOLS
from backend.construct.build_agents import build_agents
from backend.chat_bot.session_manager import SessionManager
from backend.callbacks.arxiv_callbacks import ChatDataAgentCallBackHandler

from logger import logger

environ["OPENAI_API_BASE"] = st.secrets["OPENAI_API_BASE"]

TOOL_NAMES = {
    "langchain_retriever_tool": "Self-querying retriever",
    "vecsql_retriever_tool": "Vector SQL",
}


def on_chat_submit():
    with st.session_state.next_round.container():
        with st.chat_message("user"):
            st.write(st.session_state.chat_input)
        with st.chat_message("assistant"):
            container = st.container()
        st_callback = ChatDataAgentCallBackHandler(
            container, collapse_completed_thoughts=False
        )
        ret = st.session_state.agent(
            {"input": st.session_state.chat_input}, callbacks=[st_callback]
        )
        print(ret)


def clear_history():
    if "agent" in st.session_state:
        st.session_state.agent.memory.clear()


def back_to_main():
    if USER_INFO in st.session_state:
        del st.session_state[USER_INFO]
    if USER_NAME in st.session_state:
        del st.session_state[USER_NAME]
    if JUMP_QUERY_ASK in st.session_state:
        del st.session_state[JUMP_QUERY_ASK]
    if EL_SESSION_SELECTOR in st.session_state:
        del st.session_state[EL_SESSION_SELECTOR]
    if CHAT_CURRENT_USER_SESSIONS in st.session_state:
        del st.session_state[CHAT_CURRENT_USER_SESSIONS]


def refresh_sessions():
    chat_session_manager: SessionManager = st.session_state[CHAT_SESSION_MANAGER]
    current_user_name = st.session_state[USER_NAME]
    current_user_sessions = chat_session_manager.list_sessions(current_user_name)

    if not isinstance(current_user_sessions, dict) or not current_user_sessions:
        # generate a default session for current user.
        chat_session_manager.add_session(
            user_id=current_user_name,
            session_id=f"{current_user_name}?default",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
        st.session_state[CHAT_CURRENT_USER_SESSIONS] = chat_session_manager.list_sessions(current_user_name)
        current_user_sessions = st.session_state[CHAT_CURRENT_USER_SESSIONS]
    else:
        st.session_state[CHAT_CURRENT_USER_SESSIONS] = current_user_sessions

    # load current user files.
    st.session_state[USER_PRIVATE_FILES] = st.session_state[CHAT_KNOWLEDGE_TABLE].list_files(
        current_user_name
    )
    # load current user private knowledge bases.
    st.session_state[USER_PERSONAL_KNOWLEDGE_BASES] = \
        st.session_state[CHAT_KNOWLEDGE_TABLE].list_private_knowledge_bases(current_user_name)
    logger.info(f"current user name: {current_user_name}, "
                f"user private knowledge bases: {st.session_state[USER_PERSONAL_KNOWLEDGE_BASES]}, "
                f"user private files: {st.session_state[USER_PRIVATE_FILES]}")
    st.session_state[AVAILABLE_RETRIEVAL_TOOLS] = {
        # public retrieval tools
        **st.session_state[RETRIEVER_TOOLS],
        # private retrieval tools
        **st.session_state[CHAT_KNOWLEDGE_TABLE].as_retrieval_tools(current_user_name),
    }
    # print(f"sel_session is {st.session_state.sel_session}, current_user_sessions is {current_user_sessions}")
    print(f"current_user_sessions is {current_user_sessions}")
    st.session_state[EL_SESSION_SELECTOR] = current_user_sessions[0]


# process for session add and delete.
def on_session_change_submit():
    if "session_manager" in st.session_state and "session_editor" in st.session_state:
        try:
            for elem in st.session_state.session_editor["added_rows"]:
                if len(elem) > 0 and "system_prompt" in elem and "session_id" in elem:
                    if elem["session_id"] != "" and "?" not in elem["session_id"]:
                        st.session_state.session_manager.add_session(
                            user_id=st.session_state.user_name,
                            session_id=f"{st.session_state.user_name}?{elem['session_id']}",
                            system_prompt=elem["system_prompt"],
                        )
                    else:
                        st.toast("`session_id` shouldn't be neither empty nor contain char `?`.", icon="❌")
                        raise KeyError(
                            "`session_id` shouldn't be neither empty nor contain char `?`."
                        )
                else:
                    st.toast("`You should fill both `session_id` and `system_prompt` to add a column!", icon="❌")
                    raise KeyError(
                        "You should fill both `session_id` and `system_prompt` to add a column!"
                    )
            for elem in st.session_state.session_editor["deleted_rows"]:
                user_name = st.session_state[USER_NAME]
                session_id = st.session_state[CHAT_CURRENT_USER_SESSIONS][elem]['session_id']
                user_with_session_id = f"{user_name}?{session_id}"
                st.session_state.session_manager.remove_session(session_id=user_with_session_id)
                st.toast(f"session `{user_with_session_id}` removed.", icon="✅")

            refresh_sessions()
        except Exception as e:
            sleep(2)
            st.error(f"{type(e)}: {str(e)}")
        finally:
            st.session_state.session_editor["added_rows"] = []
            st.session_state.session_editor["deleted_rows"] = []
        refresh_agent()


def create_private_knowledge_base_as_tool():
    current_user_name = st.session_state[USER_NAME]

    if (
            EL_PERSONAL_KB_NAME in st.session_state
            and EL_PERSONAL_KB_DESCRIPTION in st.session_state
            and EL_BUILD_KB_WITH_FILES in st.session_state
            and len(st.session_state[EL_PERSONAL_KB_NAME]) > 0
            and len(st.session_state[EL_PERSONAL_KB_DESCRIPTION]) > 0
            and len(st.session_state[EL_BUILD_KB_WITH_FILES]) > 0
    ):
        st.session_state[CHAT_KNOWLEDGE_TABLE].create_private_knowledge_base(
            user_id=current_user_name,
            tool_name=st.session_state[EL_PERSONAL_KB_NAME],
            tool_description=st.session_state[EL_PERSONAL_KB_DESCRIPTION],
            files=[f["file_name"] for f in st.session_state[EL_BUILD_KB_WITH_FILES]],
        )
        refresh_sessions()
    else:
        st.session_state[EL_UPLOAD_FILES_STATUS].error(
            "You should fill all fields to build up a tool!"
        )
        sleep(2)


def remove_private_knowledge_bases():
    if EL_PERSONAL_KB_NEEDS_REMOVE in st.session_state and st.session_state[EL_PERSONAL_KB_NEEDS_REMOVE]:
        private_knowledge_bases_needs_remove = st.session_state[EL_PERSONAL_KB_NEEDS_REMOVE]
        private_knowledge_base_names = [item["tool_name"] for item in private_knowledge_bases_needs_remove]
        # remove these private knowledge bases.
        st.session_state[CHAT_KNOWLEDGE_TABLE].remove_private_knowledge_bases(
            user_id=st.session_state[USER_NAME],
            private_knowledge_bases=private_knowledge_base_names
        )
        refresh_sessions()
    else:
        st.session_state[EL_UPLOAD_FILES_STATUS].error(
            "You should specify at least one private knowledge base to delete!"
        )
        time.sleep(2)


def refresh_agent():
    with st.spinner("Initializing session..."):
        user_name = st.session_state[USER_NAME]
        session_id = st.session_state[EL_SESSION_SELECTOR]['session_id']
        user_with_session_id = f"{user_name}?{session_id}"

        if EL_SELECTED_KBS in st.session_state:
            selected_knowledge_bases = st.session_state[EL_SELECTED_KBS]
        else:
            selected_knowledge_bases = ["Wikipedia + Vector SQL"]

        if EL_SESSION_SELECTOR in st.session_state:
            system_prompt = st.session_state[EL_SESSION_SELECTOR]["system_prompt"]
        else:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        st.session_state["agent"] = build_agents(
            session_id=user_with_session_id,
            tool_names=selected_knowledge_bases,
            system_prompt=system_prompt
        )


def add_file():
    user_name = st.session_state[USER_NAME]
    if EL_UPLOAD_FILES not in st.session_state or len(st.session_state[EL_UPLOAD_FILES]) == 0:
        st.session_state[EL_UPLOAD_FILES_STATUS].error("Please upload files!", icon="⚠️")
        sleep(2)
        return
    try:
        st.session_state[EL_UPLOAD_FILES_STATUS].info("Uploading...")
        st.session_state[CHAT_KNOWLEDGE_TABLE].add_by_file(
            user_id=user_name,
            files=st.session_state[EL_UPLOAD_FILES]
        )
        refresh_sessions()
    except ValueError as e:
        st.session_state[EL_UPLOAD_FILES_STATUS].error("Failed to upload! " + str(e))
        sleep(2)


def clear_files():
    st.session_state[CHAT_KNOWLEDGE_TABLE].clear(user_id=st.session_state[USER_NAME])
    refresh_sessions()

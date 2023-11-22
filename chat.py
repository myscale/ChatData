import json
import pandas as pd
from os import environ
from time import sleep
import datetime
import streamlit as st
from lib.sessions import SessionManager
from lib.private_kb import PrivateKnowledgeBase
from langchain.schema import HumanMessage, FunctionMessage
from callbacks.arxiv_callbacks import ChatDataAgentCallBackHandler
from langchain.callbacks.streamlit.streamlit_callback_handler import (
    StreamlitCallbackHandler,
)
from lib.json_conv import CustomJSONDecoder

from lib.helper import (
    build_agents,
    MYSCALE_HOST,
    MYSCALE_PASSWORD,
    MYSCALE_PORT,
    MYSCALE_USER,
    DEFAULT_SYSTEM_PROMPT,
    UNSTRUCTURED_API,
)
from login import back_to_main

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
    if "user_info" in st.session_state:
        del st.session_state.user_info
    if "user_name" in st.session_state:
        del st.session_state.user_name
    if "jump_query_ask" in st.session_state:
        del st.session_state.jump_query_ask
    if "sel_sess" in st.session_state:
        del st.session_state.sel_sess
    if "current_sessions" in st.session_state:
        del st.session_state.current_sessions


def on_session_change_submit():
    if "session_manager" in st.session_state and "session_editor" in st.session_state:
        print(st.session_state.session_editor)
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
                        raise KeyError(
                            "`session_id` should NOT be neither empty nor contain question marks."
                        )
                else:
                    raise KeyError(
                        "You should fill both `session_id` and `system_prompt` to add a column!"
                    )
            for elem in st.session_state.session_editor["deleted_rows"]:
                st.session_state.session_manager.remove_session(
                    session_id=f"{st.session_state.user_name}?{st.session_state.current_sessions[elem]['session_id']}",
                )
            refresh_sessions()
        except Exception as e:
            sleep(2)
            st.error(f"{type(e)}: {str(e)}")
        finally:
            st.session_state.session_editor["added_rows"] = []
            st.session_state.session_editor["deleted_rows"] = []
        refresh_agent()


def build_session_manager():
    return SessionManager(
        st.session_state,
        host=MYSCALE_HOST,
        port=MYSCALE_PORT,
        username=MYSCALE_USER,
        password=MYSCALE_PASSWORD,
    )


def refresh_sessions():
    st.session_state[
        "current_sessions"
    ] = st.session_state.session_manager.list_sessions(st.session_state.user_name)
    if (
        type(st.session_state.current_sessions) is not dict
        and len(st.session_state.current_sessions) <= 0
    ):
        st.session_state.session_manager.add_session(
            st.session_state.user_name,
            f"{st.session_state.user_name}?default",
            DEFAULT_SYSTEM_PROMPT,
        )
        st.session_state[
            "current_sessions"
        ] = st.session_state.session_manager.list_sessions(st.session_state.user_name)
    st.session_state["user_files"] = st.session_state.private_kb.list_files(
        st.session_state.user_name
    )
    st.session_state["user_tools"] = st.session_state.private_kb.list_tools(
        st.session_state.user_name
    )
    st.session_state["tools_with_users"] = {
        **st.session_state.tools,
        **st.session_state.private_kb.as_tools(st.session_state.user_name),
    }
    try:
        dfl_indx = [x["session_id"] for x in st.session_state.current_sessions].index(
            "default"
            if "" not in st.session_state
            else st.session_state.sel_session["session_id"]
        )
    except ValueError:
        dfl_indx = 0
    st.session_state.sel_sess = st.session_state.current_sessions[dfl_indx]


def build_kb_as_tool():
    if (
        "b_tool_name" in st.session_state
        and "b_tool_desc" in st.session_state
        and "b_tool_files" in st.session_state
        and len(st.session_state.b_tool_name) > 0
        and len(st.session_state.b_tool_desc) > 0
        and len(st.session_state.b_tool_files) > 0
    ):
        st.session_state.private_kb.create_tool(
            st.session_state.user_name,
            st.session_state.b_tool_name,
            st.session_state.b_tool_desc,
            [f["file_name"] for f in st.session_state.b_tool_files],
        )
        refresh_sessions()
    else:
        st.session_state.tool_status.error(
            "You should fill all fields to build up a tool!"
        )
        sleep(2)


def remove_kb():
    if "r_tool_names" in st.session_state and len(st.session_state.r_tool_names) > 0:
        st.session_state.private_kb.remove_tools(
            st.session_state.user_name,
            [f["tool_name"] for f in st.session_state.r_tool_names],
        )
        refresh_sessions()
    else:
        st.session_state.tool_status.error(
            "You should specify at least one tool to delete!"
        )
        sleep(2)


def refresh_agent():
    with st.spinner("Initializing session..."):
        print(
            f"??? Changed to ",
            f"{st.session_state.user_name}?{st.session_state.sel_sess['session_id']}",
        )
        st.session_state["agent"] = build_agents(
            f"{st.session_state.user_name}?{st.session_state.sel_sess['session_id']}",
            ["LangChain Self Query Retriever For Wikipedia"]
            if "selected_tools" not in st.session_state
            else st.session_state.selected_tools,
            system_prompt=DEFAULT_SYSTEM_PROMPT
            if "sel_sess" not in st.session_state
            else st.session_state.sel_sess["system_prompt"],
        )


def add_file():
    if (
        "uploaded_files" not in st.session_state
        or len(st.session_state.uploaded_files) == 0
    ):
        st.session_state.tool_status.error("Please upload files!", icon="⚠️")
        sleep(2)
        return
    try:
        st.session_state.tool_status.info("Uploading...")
        st.session_state.private_kb.add_by_file(
            st.session_state.user_name, st.session_state.uploaded_files
        )
        refresh_sessions()
    except ValueError as e:
        st.session_state.tool_status.error("Failed to upload! " + str(e))
        sleep(2)


def clear_files():
    st.session_state.private_kb.clear(st.session_state.user_name)
    refresh_sessions()


def chat_page():
    if "sel_sess" not in st.session_state:
        st.session_state["sel_sess"] = {
            "session_id": "default",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
        }
    if "private_kb" not in st.session_state:
        st.session_state["private_kb"] = PrivateKnowledgeBase(
            host=MYSCALE_HOST,
            port=MYSCALE_PORT,
            username=MYSCALE_USER,
            password=MYSCALE_PASSWORD,
            embedding=st.session_state.embeddings["Wikipedia"],
            parser_api_key=UNSTRUCTURED_API,
        )
    if "session_manager" not in st.session_state:
        st.session_state["session_manager"] = build_session_manager()
    with st.sidebar:
        with st.expander("Session Management"):
            if "current_sessions" not in st.session_state:
                refresh_sessions()
            st.info(
                "Here you can set up your session! \n\nYou can **change your prompt** here!",
                icon="🤖",
            )
            st.info(
                (
                    "**Add columns by clicking the empty row**.\n"
                    "And **delete columns by selecting rows with a press on `DEL` Key**"
                ),
                icon="💡",
            )
            st.info(
                "Don't forget to **click `Submit Change` to save your change**!",
                icon="📒",
            )
            st.data_editor(
                st.session_state.current_sessions,
                num_rows="dynamic",
                key="session_editor",
                use_container_width=True,
            )
            st.button("Submit Change!", on_click=on_session_change_submit)
        with st.expander("Session Selection", expanded=True):
            st.info(
                "If no session is attach to your account, then we will add a default session to you!",
                icon="❤️",
            )
            try:
                dfl_indx = [
                    x["session_id"] for x in st.session_state.current_sessions
                ].index(
                    "default"
                    if "" not in st.session_state
                    else st.session_state.sel_session["session_id"]
                )
            except Exception as e:
                print("*** ", str(e))
                dfl_indx = 0
            st.selectbox(
                "Choose a session to chat:",
                options=st.session_state.current_sessions,
                index=dfl_indx,
                key="sel_sess",
                format_func=lambda x: x["session_id"],
                on_change=refresh_agent,
            )
            print(st.session_state.sel_sess)
        with st.expander("Tool Settings", expanded=True):
            st.info(
                "We provides you several knowledge base tools for you. We are building more tools!",
                icon="🔧",
            )
            st.session_state["tool_status"] = st.empty()
            tab_kb, tab_file = st.tabs(
                [
                    "Knowledge Bases",
                    "File Upload",
                ]
            )
            with tab_kb:
                st.markdown("#### Build You Own Knowledge")
                st.multiselect(
                    "Select Files to Build up",
                    st.session_state.user_files,
                    placeholder="You should upload files first",
                    key="b_tool_files",
                    format_func=lambda x: x["file_name"],
                )
                st.text_input("Tool Name", "get_relevant_documents", key="b_tool_name")
                st.text_input(
                    "Tool Description",
                    "Searches among user's private files and returns related documents",
                    key="b_tool_desc",
                )
                st.button("Build!", on_click=build_kb_as_tool)
                st.markdown("### Knowledge Base Selection")
                if (
                    "user_tools" in st.session_state
                    and len(st.session_state.user_tools) > 0
                ):
                    st.markdown("***User Created Knowledge Bases***")
                    st.dataframe(st.session_state.user_tools)
                st.multiselect(
                    "Select a Knowledge Base Tool",
                    st.session_state.tools.keys()
                    if "tools_with_users" not in st.session_state
                    else st.session_state.tools_with_users,
                    default=["Wikipedia + Self Querying"],
                    key="selected_tools",
                    on_change=refresh_agent,
                )
                st.markdown("### Delete Knowledge Base")
                st.multiselect(
                    "Choose Knowledge Base to Remove",
                    st.session_state.user_tools,
                    format_func=lambda x: x["tool_name"],
                    key="r_tool_names",
                )
                st.button("Delete", on_click=remove_kb)
            with tab_file:
                st.info(
                    (
                        "We adopted [Unstructured API](https://unstructured.io/api-key) "
                        "here and we only store the processed texts from your documents. "
                        "For privacy concerns, please refer to "
                        "[our policy issue](https://myscale.com/privacy/)."
                    ),
                    icon="📃",
                )
                st.file_uploader(
                    "Upload files", key="uploaded_files", accept_multiple_files=True
                )
                st.markdown("### Uploaded Files")
                st.dataframe(
                    st.session_state.private_kb.list_files(st.session_state.user_name),
                    use_container_width=True,
                )
                col_1, col_2 = st.columns(2)
                with col_1:
                    st.button("Add Files", on_click=add_file)
                with col_2:
                    st.button("Clear Files and All Tools", on_click=clear_files)

        st.button("Clear Chat History", on_click=clear_history)
        st.button("Logout", on_click=back_to_main)
    if "agent" not in st.session_state:
        refresh_agent()
    print("!!! ", st.session_state.agent.memory.chat_memory.session_id)
    for msg in st.session_state.agent.memory.chat_memory.messages:
        speaker = "user" if isinstance(msg, HumanMessage) else "assistant"
        if isinstance(msg, FunctionMessage):
            with st.chat_message("Knowledge Base", avatar="📖"):
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
                except:
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
    st.chat_input("Input Message", on_submit=on_chat_submit, key="chat_input")

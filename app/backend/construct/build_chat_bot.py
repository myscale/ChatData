from backend.chat_bot.private_knowledge_base import ChatBotKnowledgeTable
from backend.constants.streamlit_keys import CHAT_KNOWLEDGE_TABLE, CHAT_SESSION, CHAT_SESSION_MANAGER
import streamlit as st

from backend.constants.variables import GLOBAL_CONFIG, TABLE_EMBEDDINGS_MAPPING
from backend.constants.prompts import DEFAULT_SYSTEM_PROMPT
from backend.chat_bot.session_manager import SessionManager


def build_chat_knowledge_table():
    if CHAT_KNOWLEDGE_TABLE not in st.session_state:
        st.session_state[CHAT_KNOWLEDGE_TABLE] = ChatBotKnowledgeTable(
            host=GLOBAL_CONFIG.myscale_host,
            port=GLOBAL_CONFIG.myscale_port,
            username=GLOBAL_CONFIG.myscale_user,
            password=GLOBAL_CONFIG.myscale_password,
            # embedding=st.session_state[TABLE_EMBEDDINGS_MAPPING]["Wikipedia"],
            embedding=st.session_state[TABLE_EMBEDDINGS_MAPPING]["ArXiv Papers"],
            parser_api_key=GLOBAL_CONFIG.untrusted_api,
        )


def initialize_session_manager():
    if CHAT_SESSION not in st.session_state:
        st.session_state[CHAT_SESSION] = {
            "session_id": "default",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
        }
    if CHAT_SESSION_MANAGER not in st.session_state:
        st.session_state[CHAT_SESSION_MANAGER] = SessionManager(
            st.session_state,
            host=GLOBAL_CONFIG.myscale_host,
            port=GLOBAL_CONFIG.myscale_port,
            username=GLOBAL_CONFIG.myscale_user,
            password=GLOBAL_CONFIG.myscale_password,
        )

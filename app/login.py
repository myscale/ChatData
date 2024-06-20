import streamlit as st

from backend.constants.variables import JUMP_QUERY_ASK, USER_NAME, USER_INFO


def back_to_main():
    if USER_INFO in st.session_state:
        del st.session_state[USER_INFO]
    if USER_NAME in st.session_state:
        del st.session_state[USER_NAME]
    if JUMP_QUERY_ASK in st.session_state:
        del st.session_state[JUMP_QUERY_ASK]

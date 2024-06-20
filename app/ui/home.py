from logger import logger
import os

import streamlit as st
from auth0_component import login_button

from backend.constants.variables import JUMP_QUERY_ASK, USER_INFO


def render_home():
    render_home_header()
    st.divider()
    render_home_content()
    st.divider()
    render_home_footer()


def render_home_header():
    logger.info("render home header")
    st.subheader(
        "🤗 Welcom to [MyScale](https://myscale.com)'s "
        "[ChatData](https://github.com/myscale/ChatData)! 🤗 "
    )
    st.write("You can now chat with ArXiv and Wikipedia! 🌟\n")
    st.write("Built purely with streamlit 👑 , LangChain 🦜🔗 and love ❤️ for AI!")
    st.write(
        "Follow us on [Twitter](https://x.com/myscaledb) and "
        "[Discord](https://discord.gg/D2qpkqc4Jq)!"
    )
    st.write(
        "For more details, please refer to [our repository on GitHub](https://github.com/myscale/ChatData)!")


def render_home_content():
    logger.info("render home content")
    col1, col2 = st.columns(2, gap='large')
    with col1.container():
        st.write("Try out MyScale's Self-query and Vector SQL retrievers!")
        st.write("In this demo, you will be able to see how those retrievers "
                 "**digest** -> **translate** -> **retrieve** -> **answer** to your question!")
        st.session_state[JUMP_QUERY_ASK] = st.button("Have a try >")
    with col2.container():
        st.write("Now with the power of LangChain's Conversantional Agents, we are able to build "
                 "an RAG-enabled chatbot within one MyScale instance! ")
        st.write("Log in to Chat with RAG!")
        st.write(
            "Recommended to use the standalone version of Chat-Data, "
            "available [here](https://myscale-chatdata.hf.space/)."
        )
        login_button(os.environ["AUTH0_CLIENT_ID"], os.environ["AUTH0_DOMAIN"], "auth0")


def render_home_footer():
    logger.info("render home footer")
    st.write("- [Privacy Policy](https://myscale.com/privacy/)\n"
             "- [Terms of Sevice](https://myscale.com/terms/)")
    if st.session_state.auth0 is not None:
        st.session_state[USER_INFO] = dict(st.session_state.auth0)
        if 'email' in st.session_state[USER_INFO]:
            email = st.session_state[USER_INFO]["email"]
        else:
            email = f"{st.session_state[USER_INFO]['nickname']}@{st.session_state[USER_INFO]['sub']}"
        st.session_state["user_name"] = email
        del st.session_state.auth0
        st.rerun()
    if st.session_state.jump_query_ask:
        st.rerun()

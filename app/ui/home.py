import base64

from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.card import card
from streamlit_extras.colored_header import colored_header
from streamlit_extras.mention import mention
from streamlit_extras.tags import tagger_component

from logger import logger
import os

import streamlit as st
from auth0_component import login_button

from backend.constants.variables import JUMP_QUERY_ASK, USER_INFO, USER_NAME, DIVIDER_HTML, DIVIDER_THIN_HTML
from streamlit_extras.let_it_rain import rain


def render_home():
    render_home_header()
    # st.divider()
    # st.markdown(DIVIDER_THIN_HTML, unsafe_allow_html=True)
    add_vertical_space(5)
    render_home_content()
    # st.divider()
    st.markdown(DIVIDER_THIN_HTML, unsafe_allow_html=True)
    render_home_footer()


def render_home_header():
    logger.info("render home header")
    st.header("ChatData - Your Intelligent Assistant")
    st.markdown(DIVIDER_THIN_HTML, unsafe_allow_html=True)
    st.markdown("> [ChatData](https://github.com/myscale/ChatData) \
                     is developed by [MyScale](https://myscale.com/), \
                     it's an integration of [LangChain](https://www.langchain.com/) \
                     and [MyScaleDB](https://github.com/myscale/myscaledb)")

    tagger_component(
        "Keywords:",
        ["MyScaleDB", "LangChain", "VectorSearch", "ChatBot", "GPT", "arxiv", "wikipedia", "Personal Knowledge Base 📚"],
        color_name=["darkslateblue", "green", "orange", "darkslategrey", "red", "crimson", "darkcyan", "darkgrey"],
    )
    text, col1, col2, col3, _ = st.columns([1, 1, 1, 1, 4])
    with text:
        st.markdown("Related:")
    with col1.container():
        mention(
            label="streamlit",
            icon="streamlit",
            url="https://streamlit.io/",
            write=True
        )
    with col2.container():
        mention(
            label="langchain",
            icon="🦜🔗",
            url="https://www.langchain.com/",
            write=True
        )
    with col3.container():
        mention(
            label="streamlit-extras",
            icon="🪢",
            url="https://github.com/arnaudmiribel/streamlit-extras",
            write=True
        )


def _render_self_query_chain_content():
    col1, col2 = st.columns([1, 1], gap='large')
    with col1.container():
        st.image(image='./assets/home_page_background_1.png',
                 caption=None,
                 width=None,
                 use_column_width=True,
                 clamp=False,
                 channels="RGB",
                 output_format="PNG")
    with col2.container():
        st.header("VectorSearch & SelfQuery with Sources")
        st.info("In this sample, you will learn how **LangChain** integrates with **MyScaleDB**.")
        st.markdown("""This example demonstrates two methods for integrating MyScale into LangChain: [Vector SQL](https://api.python.langchain.com/en/latest/sql/langchain_experimental.sql.vector_sql.VectorSQLDatabaseChain.html) and [Self-querying retriever](https://python.langchain.com/v0.2/docs/integrations/retrievers/self_query/myscale_self_query/). For each method, you can choose one of the following options:

1. `Retrieve from MyScaleDB ➡️` - The LLM (GPT) converts user queries into SQL statements with vector search, executes these searches in MyScaleDB, and retrieves relevant content.
   
2. `Retrieve and answer with LLM ➡️` - After retrieving relevant content from MyScaleDB, the user query along with the retrieved content is sent to the LLM (GPT), which then provides a comprehensive answer.""")
        add_vertical_space(3)
        _, middle, _ = st.columns([2, 1, 2], gap='small')
        with middle.container():
            st.session_state[JUMP_QUERY_ASK] = st.button("Try sample", use_container_width=False, type="secondary")


def _render_chat_bot_content():
    col1, col2 = st.columns(2, gap='large')
    with col1.container():
        st.image(image='./assets/home_page_background_2.png',
                 caption=None,
                 width=None,
                 use_column_width=True,
                 clamp=False,
                 channels="RGB",
                 output_format="PNG")
    with col2.container():
        st.header("Chat Bot")
        st.info("Now you can try our chatbot, this chatbot is built with MyScale and LangChain.")
        st.markdown("- You need to go to [https://myscale-chatdata.hf.space/](https://myscale-chatdata.hf.space/) "
                    "to log in successfully, otherwise the auth service will not work.")
        st.markdown("- You can upload your own PDF files and build your own knowledge base. \
                     (This is just a sample application. Please do not upload important or confidential files.)")
        st.markdown("- A default session will be assigned as your initial chat session. \
                     You can create and switch to other sessions to jump between different chat conversations.")
        add_vertical_space(1)
        _, middle, _ = st.columns([1, 2, 1], gap='small')
        with middle.container():
            if USER_NAME not in st.session_state:
                login_button(clientId=os.environ["AUTH0_CLIENT_ID"],
                             domain=os.environ["AUTH0_DOMAIN"],
                             key="auth0")
                # if user_info:
                #     user_name = user_info.get("nickname", "default") + "_" + user_info.get("email", "null")
                #     st.session_state[USER_NAME] = user_name
                #     print(user_info)


def render_home_content():
    logger.info("render home content")
    _render_self_query_chain_content()
    add_vertical_space(3)
    _render_chat_bot_content()


def render_home_footer():
    logger.info("render home footer")
    st.write(
        "Please follow us on [Twitter](https://x.com/myscaledb) and [Discord](https://discord.gg/D2qpkqc4Jq)!"
    )
    st.write(
        "For more details, please refer to [our repository on GitHub](https://github.com/myscale/ChatData)!")
    st.write("Our [privacy policy](https://myscale.com/privacy/), [terms of service](https://myscale.com/terms/)")

    # st.write(
    #     "Recommended to use the standalone version of Chat-Data, "
    #     "available [here](https://myscale-chatdata.hf.space/)."
    # )

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

import os
import time

import streamlit as st

from backend.constants.streamlit_keys import DATA_INITIALIZE_NOT_STATED, DATA_INITIALIZE_COMPLETED, \
    DATA_INITIALIZE_STARTED
from backend.constants.variables import DATA_INITIALIZE_STATUS, JUMP_QUERY_ASK, CHAINS_RETRIEVERS_MAPPING, \
    TABLE_EMBEDDINGS_MAPPING, RETRIEVER_TOOLS, USER_NAME, GLOBAL_CONFIG, update_global_config
from backend.construct.build_all import build_chains_and_retrievers, load_embedding_models, update_retriever_tools
from backend.types.global_config import GlobalConfig
from logger import logger
from ui.chat_page import chat_page
from ui.home import render_home
from ui.retrievers import render_retrievers


# warnings.filterwarnings("ignore", category=UserWarning)

def prepare_environment():
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    # os.environ["LANGCHAIN_API_KEY"] = ""
    os.environ["OPENAI_API_BASE"] = st.secrets['OPENAI_API_BASE']
    os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
    os.environ["AUTH0_CLIENT_ID"] = st.secrets['AUTH0_CLIENT_ID']
    os.environ["AUTH0_DOMAIN"] = st.secrets['AUTH0_DOMAIN']

    update_global_config(GlobalConfig(
        openai_api_base=st.secrets['OPENAI_API_BASE'],
        openai_api_key=st.secrets['OPENAI_API_KEY'],
        auth0_client_id=st.secrets['AUTH0_CLIENT_ID'],
        auth0_domain=st.secrets['AUTH0_DOMAIN'],
        myscale_user=st.secrets['MYSCALE_USER'],
        myscale_password=st.secrets['MYSCALE_PASSWORD'],
        myscale_host=st.secrets['MYSCALE_HOST'],
        myscale_port=st.secrets['MYSCALE_PORT'],
        query_model="gpt-3.5-turbo-0125",
        chat_model="gpt-3.5-turbo-0125",
        untrusted_api=st.secrets['UNSTRUCTURED_API'],
        myscale_enable_https=st.secrets.get('MYSCALE_ENABLE_HTTPS', True),
    ))


# when refresh browser, all session keys will be cleaned.
def initialize_session_state():
    if DATA_INITIALIZE_STATUS not in st.session_state:
        st.session_state[DATA_INITIALIZE_STATUS] = DATA_INITIALIZE_NOT_STATED
        logger.info(f"Initialize session state key: {DATA_INITIALIZE_STATUS}")
    if JUMP_QUERY_ASK not in st.session_state:
        st.session_state[JUMP_QUERY_ASK] = False
        logger.info(f"Initialize session state key: {JUMP_QUERY_ASK}")


def initialize_chat_data():
    if st.session_state[DATA_INITIALIZE_STATUS] != DATA_INITIALIZE_COMPLETED:
        start_time = time.time()
        st.session_state[DATA_INITIALIZE_STATUS] = DATA_INITIALIZE_STARTED
        st.session_state[TABLE_EMBEDDINGS_MAPPING] = load_embedding_models()
        st.session_state[CHAINS_RETRIEVERS_MAPPING] = build_chains_and_retrievers()
        st.session_state[RETRIEVER_TOOLS] = update_retriever_tools()
        # mark data initialization finished.
        st.session_state[DATA_INITIALIZE_STATUS] = DATA_INITIALIZE_COMPLETED
        end_time = time.time()
        logger.info(f"ChatData initialized finished in {round(end_time - start_time, 3)} seconds, "
                    f"session state keys: {list(st.session_state.keys())}")


st.set_page_config(
    page_title="ChatData",
    page_icon="https://myscale.com/favicon.ico",
    initial_sidebar_state="expanded",
    layout="wide",
)

prepare_environment()
initialize_session_state()
initialize_chat_data()

if USER_NAME in st.session_state:
    chat_page()
else:
    if st.session_state[JUMP_QUERY_ASK]:
        render_retrievers()
    else:
        render_home()

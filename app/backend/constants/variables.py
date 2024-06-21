from backend.types.global_config import GlobalConfig

# ***** str variables ***** #
EMBEDDING_MODEL_PREFIX = "embedding_model"
CHAINS_RETRIEVERS_MAPPING = "sel_map_obj"
LANGCHAIN_RETRIEVER = "langchain_retriever"
VECTOR_SQL_RETRIEVER = "vecsql_retriever"
TABLE_EMBEDDINGS_MAPPING = "embeddings"
RETRIEVER_TOOLS = "tools"
DATA_INITIALIZE_STATUS = "data_initialized"
UI_INITIALIZED = "ui_initialized"
JUMP_QUERY_ASK = "jump_query_ask"
USER_NAME = "user_name"
USER_INFO = "user_info"

DIVIDER_HTML = """
    <div style="
        height: 4px;
        background: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);
        margin-top: 20px;
        margin-bottom: 20px;
    "></div>
"""

DIVIDER_THIN_HTML = """
    <div style="
        height: 2px;
        background: linear-gradient(to right, blue, darkslateblue, indigo, violet);
        margin-top: 20px;
        margin-bottom: 20px;
    "></div>
"""


class RetrieverButtons:
    vector_sql_query_from_db = "vector_sql_query_from_db"
    vector_sql_query_with_llm = "vector_sql_query_with_llm"
    self_query_from_db = "self_query_from_db"
    self_query_with_llm = "self_query_with_llm"


GLOBAL_CONFIG = GlobalConfig()


def update_global_config(new_config: GlobalConfig):
    global GLOBAL_CONFIG
    GLOBAL_CONFIG.openai_api_base = new_config.openai_api_base
    GLOBAL_CONFIG.openai_api_key = new_config.openai_api_key
    GLOBAL_CONFIG.auth0_client_id = new_config.auth0_client_id
    GLOBAL_CONFIG.auth0_domain = new_config.auth0_domain
    GLOBAL_CONFIG.myscale_user = new_config.myscale_user
    GLOBAL_CONFIG.myscale_password = new_config.myscale_password
    GLOBAL_CONFIG.myscale_host = new_config.myscale_host
    GLOBAL_CONFIG.myscale_port = new_config.myscale_port
    GLOBAL_CONFIG.query_model = new_config.query_model
    GLOBAL_CONFIG.chat_model = new_config.chat_model
    GLOBAL_CONFIG.untrusted_api = new_config.untrusted_api
    GLOBAL_CONFIG.mode = new_config.mode

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.myscale import MyScaleTranslator
from langchain.utilities.sql_database import SQLDatabase
from langchain.vectorstores import MyScaleSettings
from langchain_experimental.retrievers.vector_sql_database import VectorSQLDatabaseChainRetriever
from langchain_experimental.sql.vector_sql import VectorSQLDatabaseChain
from sqlalchemy import create_engine, MetaData

from backend.constants.myscale_tables import MYSCALE_TABLES
from backend.constants.prompts import MYSCALE_PROMPT
from backend.constants.variables import TABLE_EMBEDDINGS_MAPPING, GLOBAL_CONFIG
from backend.retrievers.vector_sql_output_parser import VectorSQLRetrieveOutputParser
from backend.vector_store.myscale_without_metadata import MyScaleWithoutMetadataJson
from logger import logger


@st.cache_resource
def build_self_query_retriever(table_name: str) -> SelfQueryRetriever:
    with st.spinner(f"Building VectorStore for MyScaleDB/{table_name} ..."):
        myscale_connection = {
            "host": GLOBAL_CONFIG.myscale_host,
            "port": GLOBAL_CONFIG.myscale_port,
            "username": GLOBAL_CONFIG.myscale_user,
            "password": GLOBAL_CONFIG.myscale_password,
        }
        myscale_settings = MyScaleSettings(
            **myscale_connection,
            database=MYSCALE_TABLES[table_name].database,
            table=MYSCALE_TABLES[table_name].table,
            column_map={
                "id": "id",
                "text": MYSCALE_TABLES[table_name].text_col_name,
                "vector": MYSCALE_TABLES[table_name].vector_col_name,
                # TODO refine MyScaleDB metadata in langchain.
                "metadata": MYSCALE_TABLES[table_name].metadata_col_name
            }
        )
        myscale_vector_store = MyScaleWithoutMetadataJson(
            embedding=st.session_state[TABLE_EMBEDDINGS_MAPPING][table_name],
            config=myscale_settings,
            must_have_cols=MYSCALE_TABLES[table_name].must_have_col_names
        )

    with st.spinner(f"Building SelfQueryRetriever for MyScaleDB/{table_name} ..."):
        retriever: SelfQueryRetriever = SelfQueryRetriever.from_llm(
            llm=ChatOpenAI(
                model_name=GLOBAL_CONFIG.query_model,
                base_url=GLOBAL_CONFIG.openai_api_base,
                api_key=GLOBAL_CONFIG.openai_api_key,
                temperature=0
            ),
            vectorstore=myscale_vector_store,
            document_contents=MYSCALE_TABLES[table_name].table_contents,
            metadata_field_info=MYSCALE_TABLES[table_name].metadata_col_attributes,
            use_original_query=False,
            structured_query_translator=MyScaleTranslator()
        )
    return retriever


@st.cache_resource
def build_vector_sql_db_chain_retriever(table_name: str) -> VectorSQLDatabaseChainRetriever:
    """Get a group of relative docs from MyScaleDB"""
    with st.spinner(f'Building Vector SQL Database Retriever for MyScaleDB/{table_name}...'):
        if GLOBAL_CONFIG.myscale_enable_https == False:
            engine = create_engine(
                f'clickhouse://{GLOBAL_CONFIG.myscale_user}:{GLOBAL_CONFIG.myscale_password}@'
                f'{GLOBAL_CONFIG.myscale_host}:{GLOBAL_CONFIG.myscale_port}'
                f'/{MYSCALE_TABLES[table_name].database}?protocol=http'
            )
        else:
            engine = create_engine(
                f'clickhouse://{GLOBAL_CONFIG.myscale_user}:{GLOBAL_CONFIG.myscale_password}@'
                f'{GLOBAL_CONFIG.myscale_host}:{GLOBAL_CONFIG.myscale_port}'
                f'/{MYSCALE_TABLES[table_name].database}?protocol=https'
            )
        metadata = MetaData(bind=engine)
        logger.info(f"{table_name} metadata is : {metadata}")
        prompt = PromptTemplate(
            input_variables=["input", "table_info", "top_k"],
            template=MYSCALE_PROMPT,
        )
        # Custom `out_put_parser` rewrite search SQL, make it's possible to query custom column.
        output_parser = VectorSQLRetrieveOutputParser.from_embeddings(
            model=st.session_state[TABLE_EMBEDDINGS_MAPPING][table_name],
            # rewrite columns needs be searched.
            must_have_columns=MYSCALE_TABLES[table_name].must_have_col_names
        )

        # `db_chain` will generate a SQL
        vector_sql_db_chain: VectorSQLDatabaseChain = VectorSQLDatabaseChain.from_llm(
            llm=ChatOpenAI(
                model_name=GLOBAL_CONFIG.query_model,
                base_url=GLOBAL_CONFIG.openai_api_base,
                api_key=GLOBAL_CONFIG.openai_api_key,
                temperature=0
            ),
            prompt=prompt,
            top_k=10,
            return_direct=True,
            db=SQLDatabase(
                engine,
                None,
                metadata,
                include_tables=[MYSCALE_TABLES[table_name].table],
                max_string_length=1024
            ),
            sql_cmd_parser=output_parser,  # TODO needs update `langchain`, fix return type.
            native_format=True
        )

        # `retriever` can search a group of documents with `db_chain`
        vector_sql_db_chain_retriever = VectorSQLDatabaseChainRetriever(
            sql_db_chain=vector_sql_db_chain,
            page_content_key=MYSCALE_TABLES[table_name].text_col_name
        )
    return vector_sql_db_chain_retriever

from logger import logger
from typing import Dict, Any, Union

import streamlit as st

from backend.constants.myscale_tables import MYSCALE_TABLES
from backend.constants.variables import CHAINS_RETRIEVERS_MAPPING
from backend.construct.build_chains import build_retrieval_qa_with_sources_chain
from backend.construct.build_retriever_tool import create_retriever_tool
from backend.construct.build_retrievers import build_self_query_retriever, build_vector_sql_db_chain_retriever
from backend.types.chains_and_retrievers import ChainsAndRetrievers, MetadataColumn

from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings, \
    SentenceTransformerEmbeddings


@st.cache_resource
def load_embedding_model_for_table(table_name: str) -> \
        Union[SentenceTransformerEmbeddings, HuggingFaceInstructEmbeddings]:
    with st.spinner(f"Loading embedding models for [{table_name}] ..."):
        embeddings = MYSCALE_TABLES[table_name].emb_model()
    return embeddings


@st.cache_resource
def load_embedding_models() -> Dict[str, Union[HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings]]:
    embedding_models = {}
    for table in MYSCALE_TABLES:
        embedding_models[table] = load_embedding_model_for_table(table)
    return embedding_models


@st.cache_resource
def update_retriever_tools():
    retrievers_tools = {}
    for table in MYSCALE_TABLES:
        logger.info(f"Updating retriever tools [<retriever>, <sql_retriever>] for table {table}")
        retrievers_tools.update(
            {
                f"{table} + Self Querying": create_retriever_tool(
                    st.session_state[CHAINS_RETRIEVERS_MAPPING][table]["retriever"],
                    *MYSCALE_TABLES[table].tool_desc
                ),
                f"{table} + Vector SQL": create_retriever_tool(
                    st.session_state[CHAINS_RETRIEVERS_MAPPING][table]["sql_retriever"],
                    *MYSCALE_TABLES[table].tool_desc
                ),
            })
    return retrievers_tools


@st.cache_resource
def build_chains_retriever_for_table(table_name: str) -> ChainsAndRetrievers:
    metadata_col_attributes = MYSCALE_TABLES[table_name].metadata_col_attributes

    self_query_retriever = build_self_query_retriever(table_name)
    self_query_chain = build_retrieval_qa_with_sources_chain(
        table_name=table_name,
        retriever=self_query_retriever,
        chain_name="Self Query Retriever"
    )

    vector_sql_retriever = build_vector_sql_db_chain_retriever(table_name)
    vector_sql_chain = build_retrieval_qa_with_sources_chain(
        table_name=table_name,
        retriever=vector_sql_retriever,
        chain_name="Vector SQL DB Retriever"
    )

    metadata_columns = [
        MetadataColumn(
            name=attribute.name,
            desc=attribute.description,
            type=attribute.type
        )
        for attribute in metadata_col_attributes
    ]
    return ChainsAndRetrievers(
        metadata_columns=metadata_columns,
        # for self query
        retriever=self_query_retriever,
        chain=self_query_chain,
        # for vector sql
        sql_retriever=vector_sql_retriever,
        sql_chain=vector_sql_chain
    )


@st.cache_resource
def build_chains_and_retrievers() -> Dict[str, Dict[str, Any]]:
    chains_and_retrievers = {}
    for table in MYSCALE_TABLES:
        logger.info(f"Building chains, retrievers for table {table}")
        chains_and_retrievers[table] = build_chains_retriever_for_table(table).to_dict()
    return chains_and_retrievers

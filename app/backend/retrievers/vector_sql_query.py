from typing import List

import pandas as pd
import streamlit as st
from langchain.schema import Document
from langchain_experimental.retrievers.vector_sql_database import VectorSQLDatabaseChainRetriever

from backend.chains.retrieval_qa_with_sources import CustomRetrievalQAWithSourcesChain
from backend.constants.myscale_tables import MYSCALE_TABLES
from backend.constants.variables import CHAINS_RETRIEVERS_MAPPING, DIVIDER_HTML, RetrieverButtons
from backend.callbacks.vector_sql_callbacks import VectorSQLSearchDBCallBackHandler, VectorSQLSearchLLMCallBackHandler
from ui.utils import display
from logger import logger


def process_sql_query(selected_table: str, query_type: str):
    place_holder = st.empty()
    logger.info(
        f"button-1: {st.session_state[RetrieverButtons.vector_sql_query_from_db]}, "
        f"button-2: {st.session_state[RetrieverButtons.vector_sql_query_with_llm]}, "
        f"table: {selected_table}, "
        f"content: {st.session_state.query_sql}"
    )
    with place_holder.expander('🪵 Query Log', expanded=True):
        try:
            if query_type == RetrieverButtons.vector_sql_query_from_db:
                callback = VectorSQLSearchDBCallBackHandler()
                vector_sql_retriever: VectorSQLDatabaseChainRetriever = \
                    st.session_state[CHAINS_RETRIEVERS_MAPPING][selected_table]["sql_retriever"]
                relevant_docs: List[Document] = vector_sql_retriever.get_relevant_documents(
                    query=st.session_state.query_sql,
                    callbacks=[callback]
                )

                callback.progress_bar.progress(
                    value=1.0,
                    text="[Question -> LLM -> SQL Statement -> MyScaleDB -> Results] Done! ✅"
                )

                st.markdown(f"### Vector Search Results from `{selected_table}` \n"
                            f"> Here we get documents from MyScaleDB with given sql statement \n\n")
                display(
                    pd.DataFrame(
                        [{**d.metadata, 'abstract': d.page_content} for d in relevant_docs]
                    )
                )
            elif query_type == RetrieverButtons.vector_sql_query_with_llm:
                callback = VectorSQLSearchLLMCallBackHandler(table=selected_table)
                vector_sql_chain: CustomRetrievalQAWithSourcesChain = \
                    st.session_state[CHAINS_RETRIEVERS_MAPPING][selected_table]["sql_chain"]
                chain_results = vector_sql_chain(
                    inputs=st.session_state.query_sql,
                    callbacks=[callback]
                )

                callback.progress_bar.progress(
                    value=1.0,
                    text="[Question -> LLM -> SQL Statement -> MyScaleDB -> "
                         "(Question,Results) -> LLM -> Results] Done! ✅"
                )

                documents_reference: List[Document] = chain_results["source_documents"]
                st.markdown(f"### Vector Search Results from `{selected_table}` \n"
                            f"> Here we get documents from MyScaleDB with given sql statement \n\n")
                display(
                    pd.DataFrame(
                        [{**d.metadata, 'abstract': d.page_content} for d in documents_reference]
                    )
                )
                st.markdown(
                    f"### Answer from LLM \n"
                    f"> The response of the LLM when given the vector search results. \n\n"
                )
                st.write(chain_results['answer'])
                st.markdown(
                    f"### References from `{selected_table}`\n"
                    f"> Here shows that which documents used by LLM \n\n"
                )
                if len(chain_results['sources']) == 0:
                    st.write("No documents is used by LLM.")
                else:
                    display(
                        dataframe=pd.DataFrame(
                            [{**d.metadata, 'abstract': d.page_content} for d in chain_results['sources']]
                        ),
                        columns_=['ref_id'] + MYSCALE_TABLES[selected_table].must_have_col_names,
                        index='ref_id'
                    )
            else:
                raise NotImplementedError(f"Unsupported query type: {query_type}")
            st.markdown(DIVIDER_HTML, unsafe_allow_html=True)
        except Exception as e:
            st.write('Oops 😵 Something bad happened...')
            raise e


from typing import List

import pandas as pd
import streamlit as st
from langchain.retrievers import SelfQueryRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from backend.chains.retrieval_qa_with_sources import CustomRetrievalQAWithSourcesChain
from backend.constants.myscale_tables import MYSCALE_TABLES
from backend.constants.variables import CHAINS_RETRIEVERS_MAPPING, DIVIDER_HTML, RetrieverButtons
from backend.callbacks.self_query_callbacks import ChatDataSelfAskCallBackHandler, CustomSelfQueryRetrieverCallBackHandler
from ui.utils import display
from logger import logger


def process_self_query(selected_table, query_type):
    place_holder = st.empty()
    logger.info(
        f"button-1: {RetrieverButtons.self_query_from_db}, "
        f"button-2: {RetrieverButtons.self_query_with_llm}, "
        f"content: {st.session_state.query_self}"
    )
    with place_holder.expander('🪵 Chat Log', expanded=True):
        try:
            if query_type == RetrieverButtons.self_query_from_db:
                callback = CustomSelfQueryRetrieverCallBackHandler()
                retriever: SelfQueryRetriever = \
                    st.session_state[CHAINS_RETRIEVERS_MAPPING][selected_table]["retriever"]
                config: RunnableConfig = {"callbacks": [callback]}

                relevant_docs = retriever.invoke(
                    input=st.session_state.query_self,
                    config=config
                )

                callback.progress_bar.progress(
                    value=1.0, text="[Question -> LLM -> Query filter -> MyScaleDB -> Results] Done!✅")

                st.markdown(f"### Self Query Results from `{selected_table}` \n"
                            f"> Here we get documents from MyScaleDB by `SelfQueryRetriever` \n\n")
                display(
                    dataframe=pd.DataFrame(
                        [{**d.metadata, 'abstract': d.page_content} for d in relevant_docs]
                    ),
                    columns_=MYSCALE_TABLES[selected_table].must_have_col_names
                )
            elif query_type == RetrieverButtons.self_query_with_llm:
                # callback = CustomSelfQueryRetrieverCallBackHandler()
                callback = ChatDataSelfAskCallBackHandler()
                chain: CustomRetrievalQAWithSourcesChain = \
                    st.session_state[CHAINS_RETRIEVERS_MAPPING][selected_table]["chain"]
                chain_results = chain(st.session_state.query_self, callbacks=[callback])
                callback.progress_bar.progress(
                    value=1.0,
                    text="[Question -> LLM -> Query filter -> MyScaleDB -> Related Results -> LLM -> LLM Answer] Done!✅"
                )

                documents_reference: List[Document] = chain_results["source_documents"]
                st.markdown(f"### SelfQueryRetriever Results from `{selected_table}` \n"
                            f"> Here we get documents from MyScaleDB by `SelfQueryRetriever` \n\n")
                display(
                    pd.DataFrame(
                        [{**d.metadata, 'abstract': d.page_content} for d in documents_reference]
                    )
                )
                st.markdown(
                    f"### Answer from LLM \n"
                    f"> The response of the LLM when given the `SelfQueryRetriever` results. \n\n"
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
            st.markdown(DIVIDER_HTML, unsafe_allow_html=True)
        except Exception as e:
            st.write('Oops 😵 Something bad happened...')
            raise e

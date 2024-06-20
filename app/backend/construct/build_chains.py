from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseRetriever
import streamlit as st

from backend.chains.retrieval_qa_with_sources import CustomRetrievalQAWithSourcesChain
from backend.chains.stuff_documents import CustomStuffDocumentChain
from backend.constants.myscale_tables import MYSCALE_TABLES
from backend.constants.prompts import COMBINE_PROMPT
from backend.constants.variables import GLOBAL_CONFIG


def build_retrieval_qa_with_sources_chain(
        table_name: str,
        retriever: BaseRetriever,
        chain_name: str = "<chain_name>"
) -> CustomRetrievalQAWithSourcesChain:
    with st.spinner(f'Building QA source chain named `{chain_name}` for MyScaleDB/{table_name} ...'):
        # Assign ref_id for documents
        custom_stuff_document_chain = CustomStuffDocumentChain(
            llm_chain=LLMChain(
                prompt=COMBINE_PROMPT,
                llm=ChatOpenAI(
                    model_name=GLOBAL_CONFIG.chat_model,
                    openai_api_key=GLOBAL_CONFIG.openai_api_key,
                    temperature=0.6
                ),
            ),
            document_prompt=MYSCALE_TABLES[table_name].doc_prompt,
            document_variable_name="summaries",
        )
        chain = CustomRetrievalQAWithSourcesChain(
            retriever=retriever,
            combine_documents_chain=custom_stuff_document_chain,
            return_source_documents=True,
            max_tokens_limit=12000,
        )
    return chain

from prompts.arxiv_prompt import combine_prompt_template, _myscale_prompt
from callbacks.arxiv_callbacks import ChatDataSelfSearchCallBackHandler, \
    ChatDataSelfAskCallBackHandler, ChatDataSQLSearchCallBackHandler, \
    ChatDataSQLAskCallBackHandler
from chains.arxiv_chains import ArXivQAwithSourcesChain, ArXivStuffDocumentChain
from chains.arxiv_chains import VectorSQLRetrieveCustomOutputParser
from langchain_experimental.sql.vector_sql import VectorSQLDatabaseChain
from langchain_experimental.retrievers.vector_sql_database import VectorSQLDatabaseChainRetriever
from langchain.utilities.sql_database import SQLDatabase
from langchain.chains import LLMChain
from sqlalchemy import create_engine, MetaData
from langchain.prompts import PromptTemplate, ChatPromptTemplate, \
    SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
import re
import pandas as pd
from os import environ
import streamlit as st
import datetime
from helper import build_all, sel_map, display
environ['OPENAI_API_BASE'] = st.secrets['OPENAI_API_BASE']

st.set_page_config(page_title="ChatData")

st.header("ChatData")

if 'retriever' not in st.session_state:
    st.session_state["sel_map_obj"] = build_all()

sel = st.selectbox('Choose the knowledge base you want to ask with:',
                   options=['ArXiv Papers', 'Wikipedia'])
sel_map[sel]['hint']()
tab_sql, tab_self_query = st.tabs(['Vector SQL', 'Self-Query Retrievers'])
with tab_sql:
    sel_map[sel]['hint_sql']()
    st.text_input("Ask a question:", key='query_sql')
    cols = st.columns([1, 1, 7])
    cols[0].button("Query", key='search_sql')
    cols[1].button("Ask", key='ask_sql')
    plc_hldr = st.empty()
    if st.session_state.search_sql:
        plc_hldr = st.empty()
        print(st.session_state.query_sql)
        with plc_hldr.expander('Query Log', expanded=True):
            callback = ChatDataSQLSearchCallBackHandler()
            try:
                docs = st.session_state.sel_map_obj[sel]["sql_retriever"].get_relevant_documents(
                    st.session_state.query_sql, callbacks=[callback])
                callback.progress_bar.progress(value=1.0, text="Done!")
                docs = pd.DataFrame(
                    [{**d.metadata, 'abstract': d.page_content} for d in docs])
                display(docs)
            except Exception as e:
                st.write('Oops 😵 Something bad happened...')
                raise e

    if st.session_state.ask_sql:
        plc_hldr = st.empty()
        print(st.session_state.query_sql)
        with plc_hldr.expander('Chat Log', expanded=True):
            callback = ChatDataSQLAskCallBackHandler()
            try:
                ret = st.session_state.sel_map_obj[sel]["sql_chain"](
                    st.session_state.query_sql, callbacks=[callback])
                callback.progress_bar.progress(value=1.0, text="Done!")
                st.markdown(
                    f"### Answer from LLM\n{ret['answer']}\n### References")
                docs = ret['sources']
                docs = pd.DataFrame(
                    [{**d.metadata, 'abstract': d.page_content} for d in docs])
                display(
                    docs, ['ref_id'] + sel_map[sel]["must_have_cols"], index='ref_id')
            except Exception as e:
                st.write('Oops 😵 Something bad happened...')
                raise e


with tab_self_query:
    st.info("You can retrieve papers with button `Query` or ask questions based on retrieved papers with button `Ask`.", icon='💡')
    st.dataframe(st.session_state.sel_map_obj[sel]["metadata_columns"])
    st.text_input("Ask a question:", key='query_self')
    cols = st.columns([1, 1, 7])
    cols[0].button("Query", key='search_self')
    cols[1].button("Ask", key='ask_self')
    plc_hldr = st.empty()
    if st.session_state.search_self:
        plc_hldr = st.empty()
        print(st.session_state.query_self)
        with plc_hldr.expander('Query Log', expanded=True):
            call_back = None
            callback = ChatDataSelfSearchCallBackHandler()
            try:
                docs = st.session_state.sel_map_obj[sel]["retriever"].get_relevant_documents(
                    st.session_state.query_self, callbacks=[callback])
                print(docs)
                callback.progress_bar.progress(value=1.0, text="Done!")
                docs = pd.DataFrame(
                    [{**d.metadata, 'abstract': d.page_content} for d in docs])
                display(docs, sel_map[sel]["must_have_cols"])
            except Exception as e:
                st.write('Oops 😵 Something bad happened...')
                raise e

    if st.session_state.ask_self:
        plc_hldr = st.empty()
        print(st.session_state.query_self)
        with plc_hldr.expander('Chat Log', expanded=True):
            call_back = None
            callback = ChatDataSelfAskCallBackHandler()
            try:
                ret = st.session_state.sel_map_obj[sel]["chain"](
                    st.session_state.query_self, callbacks=[callback])
                callback.progress_bar.progress(value=1.0, text="Done!")
                st.markdown(
                    f"### Answer from LLM\n{ret['answer']}\n### References")
                docs = ret['sources']
                docs = pd.DataFrame(
                    [{**d.metadata, 'abstract': d.page_content} for d in docs])
                display(
                    docs, ['ref_id'] + sel_map[sel]["must_have_cols"], index='ref_id')
            except Exception as e:
                st.write('Oops 😵 Something bad happened...')
                raise e

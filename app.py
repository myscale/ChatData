import re
import pandas as pd
from os import environ
import streamlit as st
import datetime
environ['TOKENIZERS_PARALLELISM'] = 'true'
environ['OPENAI_API_BASE'] = st.secrets['OPENAI_API_BASE']

from langchain.vectorstores import MyScale, MyScaleSettings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import PromptTemplate, ChatPromptTemplate, \
    SystemMessagePromptTemplate, HumanMessagePromptTemplate
from sqlalchemy import create_engine, MetaData
from langchain.chains import LLMChain

from langchain_experimental.utilities.sql_database import SQLDatabase
from langchain_experimental.sql.base import SQLDatabaseChain
from langchain_experimental.sql.parser import VectorSQLRetrieveAllOutputParser
from langchain_experimental.retrievers.sql_database import SQLDatabaseChainRetriever

from chains.arxiv_chains import ArXivQAwithSourcesChain, ArXivStuffDocumentChain
from callbacks.arxiv_callbacks import ChatDataSelfSearchCallBackHandler, \
    ChatDataSelfAskCallBackHandler, ChatDataSQLSearchCallBackHandler, \
    ChatDataSQLAskCallBackHandler
from prompts.arxiv_prompt import combine_prompt_template, _myscale_prompt

st.set_page_config(page_title="ChatData")

st.header("ChatData")

columns = ['ref_id', 'title', 'id', 'categories', 'abstract', 'authors', 'pubdate']


def try_eval(x):
    try:
        return eval(x, {'datetime': datetime})
    except:
        return x


def display(dataframe, columns=None, index=None):
    if len(dataframe) > 0:
        if index:
            dataframe.set_index(index)
        if columns:
            st.dataframe(dataframe[columns])
        else:
            st.dataframe(dataframe)
    else:
        st.write("Sorry 😵 we didn't find any articles related to your query.\n\nMaybe the LLM is too naughty that does not follow our instruction... \n\nPlease try again and use verbs that may match the datatype.", unsafe_allow_html=True)


@st.cache_resource
def build_retriever():
    with st.spinner("Loading Model..."):
        embeddings = HuggingFaceInstructEmbeddings(
            model_name='hkunlp/instructor-xl',
            embed_instruction="Represent the question for retrieving supporting scientific papers: ")

    with st.spinner("Connecting DB..."):
        myscale_connection = {
            "host": st.secrets['MYSCALE_HOST'],
            "port": st.secrets['MYSCALE_PORT'],
            "username": st.secrets['MYSCALE_USER'],
            "password": st.secrets['MYSCALE_PASSWORD'],
        }

        config = MyScaleSettings(**myscale_connection, table='ChatArXiv',
                                 column_map={
                                     "id": "id",
                                     "text": "abstract",
                                     "vector": "vector",
                                     "metadata": "metadata"
                                 })
        doc_search = MyScale(embeddings, config)

    with st.spinner("Building Self Query Retriever..."):
        metadata_field_info = [
            AttributeInfo(
                name="pubdate",
                description="The year the paper is published",
                type="timestamp",
            ),
            AttributeInfo(
                name="authors",
                description="List of author names",
                type="list[string]",
            ),
            AttributeInfo(
                name="title",
                description="Title of the paper",
                type="string",
            ),
            AttributeInfo(
                name="categories",
                description="arxiv categories to this paper",
                type="list[string]"
            ),
            AttributeInfo(
                name="length(categories)",
                description="length of arxiv categories to this paper",
                type="int"
            ),
        ]
        retriever = SelfQueryRetriever.from_llm(
            OpenAI(openai_api_key=st.secrets['OPENAI_API_KEY'], temperature=0),
            doc_search, "Scientific papers indexes with abstracts. All in English.", metadata_field_info,
            use_original_query=False)


    document_with_metadata_prompt = PromptTemplate(
        input_variables=["page_content", "id", "title", "ref_id",
                        "authors", "pubdate", "categories"],
        template="Title for PDF #{ref_id}: {title}\n\tAbstract: {page_content}\n\tAuthors: {authors}\n\tDate of Publication: {pubdate}\n\tCategories: {categories}\nSOURCE: {id}")

    COMBINE_PROMPT = ChatPromptTemplate.from_strings(
        string_messages=[(SystemMessagePromptTemplate, combine_prompt_template),
                        (HumanMessagePromptTemplate, '{question}')])
    OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

    with st.spinner('Building QA Chain with Self-query...'):
        chain = ArXivQAwithSourcesChain(
            retriever=retriever,
            combine_documents_chain=ArXivStuffDocumentChain(
                llm_chain=LLMChain(
                    prompt=COMBINE_PROMPT,
                    llm=ChatOpenAI(model_name='gpt-3.5-turbo-16k',
                                openai_api_key=OPENAI_API_KEY, temperature=0.6),
                ),
                document_prompt=document_with_metadata_prompt,
                document_variable_name="summaries",

            ),
            return_source_documents=True,
            max_tokens_limit=12000,
        )

    with st.spinner('Building Vector SQL Database Retriever'):
        MYSCALE_USER = st.secrets['MYSCALE_USER']
        MYSCALE_PASSWORD = st.secrets['MYSCALE_PASSWORD']
        MYSCALE_HOST = st.secrets['MYSCALE_HOST']
        MYSCALE_PORT = st.secrets['MYSCALE_PORT']
        engine = create_engine(
            f'clickhouse://{MYSCALE_USER}:{MYSCALE_PASSWORD}@{MYSCALE_HOST}:{MYSCALE_PORT}/default?protocol=https')
        metadata = MetaData(bind=engine)
        PROMPT = PromptTemplate(
            input_variables=["input", "table_info", "top_k"],
            template=_myscale_prompt,
        )

        output_parser = VectorSQLRetrieveAllOutputParser.from_embeddings(
            model=embeddings)
        sql_query_chain = SQLDatabaseChain.from_llm(
            llm=OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
            prompt=PROMPT,
            top_k=10,
            return_direct=True,
            db=SQLDatabase(engine, None, metadata, max_string_length=1024),
            sql_cmd_parser=output_parser,
            native_format=True
        )
        sql_retriever = SQLDatabaseChainRetriever(
            sql_db_chain=sql_query_chain, page_content_key="abstract")

    with st.spinner('Building QA Chain with Vector SQL...'):
        sql_chain = ArXivQAwithSourcesChain(
            retriever=sql_retriever,
            combine_documents_chain=ArXivStuffDocumentChain(
                llm_chain=LLMChain(
                    prompt=COMBINE_PROMPT,
                    llm=ChatOpenAI(model_name='gpt-3.5-turbo-16k',
                                openai_api_key=OPENAI_API_KEY, temperature=0.6),
                ),
                document_prompt=document_with_metadata_prompt,
                document_variable_name="summaries",

            ),
            return_source_documents=True,
            max_tokens_limit=12000,
        )

    return [{'name': m.name, 'desc': m.description, 'type': m.type} for m in metadata_field_info], retriever, chain, sql_retriever, sql_chain


if 'retriever' not in st.session_state:
    st.session_state['metadata_columns'], \
        st.session_state['retriever'], \
        st.session_state['chain'], \
        st.session_state['sql_retriever'], \
        st.session_state['sql_chain'] = build_retriever()

st.info("We provides you metadata columns below for query. Please choose a natural expression to describe filters on those columns.\n\n"
        "For example: \n\n"
        "*If you want to search papers with complex filters*:\n\n"
        "- What is a Bayesian network? Please use articles published later than Feb 2018 and with more than 2 categories and whose title like `computer` and must have `cs.CV` in its category.\n\n"
        "*If you want to ask questions based on papers in database*:\n\n"
        "- What is PageRank?\n"
        "- Did Geoffrey Hinton wrote paper about Capsule Neural Networks?\n"
        "- Introduce some applications of GANs published around 2019.\n"
        "- 请根据 2019 年左右的文章介绍一下 GAN 的应用都有哪些\n"
        "- Veuillez présenter les applications du GAN sur la base des articles autour de 2019 ?")
tab_sql, tab_self_query = st.tabs(['Vector SQL', 'Self-Query Retrievers'])
with tab_sql:
    st.info("You can retrieve papers with button `Query` or ask questions based on retrieved papers with button `Ask`.", icon='💡')
    st.markdown('''```sql
CREATE TABLE default.ChatArXiv (
    `abstract` String, 
    `id` String, 
    `vector` Array(Float32), 
    `metadata` Object('JSON'), 
    `pubdate` DateTime,
    `title` String,
    `categories` Array(String),
    `authors` Array(String), 
    `comment` String,
    `primary_category` String,
    VECTOR INDEX vec_idx vector TYPE MSTG('metric_type=Cosine'), 
    CONSTRAINT vec_len CHECK length(vector) = 768) 
ENGINE = ReplacingMergeTree ORDER BY id
```''')

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
                docs = st.session_state.sql_retriever.get_relevant_documents(
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
                ret = st.session_state.sql_chain(
                    st.session_state.query_sql, callbacks=[callback])
                callback.progress_bar.progress(value=1.0, text="Done!")
                st.markdown(
                    f"### Answer from LLM\n{ret['answer']}\n### References")
                docs = ret['sources']
                docs = pd.DataFrame([{**d.metadata, 'abstract': d.page_content} for d in docs])
                display(docs, columns, index='ref_id')
            except Exception as e:
                st.write('Oops 😵 Something bad happened...')
                raise e


with tab_self_query:
    st.info("You can retrieve papers with button `Query` or ask questions based on retrieved papers with button `Ask`.", icon='💡')
    st.dataframe(st.session_state.metadata_columns)
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
                docs = st.session_state.retriever.get_relevant_documents(
                    st.session_state.query_self, callbacks=[callback])
                callback.progress_bar.progress(value=1.0, text="Done!")
                docs = pd.DataFrame(
                    [{**d.metadata, 'abstract': d.page_content} for d in docs])

                display(docs, columns)
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
                ret = st.session_state.chain(
                    st.session_state.query_self, callbacks=[callback])
                callback.progress_bar.progress(value=1.0, text="Done!")
                st.markdown(
                    f"### Answer from LLM\n{ret['answer']}\n### References")
                docs = ret['sources']
                docs = pd.DataFrame([{**d.metadata, 'abstract': d.page_content} for d in docs])
                display(docs, columns, index='ref_id')
            except Exception as e:
                st.write('Oops 😵 Something bad happened...')
                raise e

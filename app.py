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
from langchain.chains.query_constructor.base import AttributeInfo, VirtualColumnName
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.myscale import MyScaleTranslator
from langchain.embeddings import HuggingFaceInstructEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import MyScaleSettings
from chains.arxiv_chains import MyScaleWithoutMetadataJson
import re
import pandas as pd
from os import environ
import streamlit as st
import datetime
environ['TOKENIZERS_PARALLELISM'] = 'true'
environ['OPENAI_API_BASE'] = st.secrets['OPENAI_API_BASE']


st.set_page_config(page_title="ChatData")

st.header("ChatData")

# query_model_name = "gpt-3.5-turbo-instruct"
query_model_name = "text-davinci-003"
chat_model_name = "gpt-3.5-turbo-16k"


def hint_arxiv():
    st.info("We provides you metadata columns below for query. Please choose a natural expression to describe filters on those columns.\n\n"
            "For example: \n\n"
            "*If you want to search papers with complex filters*:\n\n"
            "- What is a Bayesian network? Please use articles published later than Feb 2018 and with more than 2 categories and whose title like `computer` and must have `cs.CV` in its category.\n\n"
            "*If you want to ask questions based on papers in database*:\n\n"
            "- What is PageRank?\n"
            "- Did Geoffrey Hinton wrote paper about Capsule Neural Networks?\n"
            "- Introduce some applications of GANs published around 2019.\n"
            "- 请根据 2019 年左右的文章介绍一下 GAN 的应用都有哪些\n"
            "- Veuillez présenter les applications du GAN sur la base des articles autour de 2019 ?\n"
            "- Is it possible to synthesize room temperature super conductive material?")


def hint_sql_arxiv():
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
    VECTOR INDEX vec_idx vector TYPE MSTG('fp16_storage=1', 'metric_type=Cosine', 'disk_mode=3'), 
    CONSTRAINT vec_len CHECK length(vector) = 768) 
ENGINE = ReplacingMergeTree ORDER BY id
```''')


def hint_wiki():
    st.info("We provides you metadata columns below for query. Please choose a natural expression to describe filters on those columns.\n\n"
            "For example: \n\n"
            "- Which company did Elon Musk found?\n"
            "- What is Iron Gwazi?\n"
            "- What is a Ring in mathematics?\n"
            "- 苹果的发源地是那里？\n")


def hint_sql_wiki():
    st.info("You can retrieve papers with button `Query` or ask questions based on retrieved papers with button `Ask`.", icon='💡')
    st.markdown('''```sql
CREATE TABLE wiki.Wikipedia (
    `id` String, 
    `title` String, 
    `text` String, 
    `url` String, 
    `wiki_id` UInt64, 
    `views` Float32, 
    `paragraph_id` UInt64, 
    `langs` UInt32, 
    `emb` Array(Float32), 
    VECTOR INDEX vec_idx emb TYPE MSTG('fp16_storage=1', 'metric_type=Cosine', 'disk_mode=3'), 
    CONSTRAINT emb_len CHECK length(emb) = 768) 
ENGINE = ReplacingMergeTree ORDER BY id
```''')


sel_map = {
    'Wikipedia': {
        "database": "wiki",
        "table": "Wikipedia",
        "hint": hint_wiki,
        "hint_sql": hint_sql_wiki,
        "doc_prompt": PromptTemplate(
            input_variables=["page_content", "url", "title", "ref_id", "views"],
            template="Title for Doc #{ref_id}: {title}\n\tviews: {views}\n\tcontent: {page_content}\nSOURCE: {url}"),
        "metadata_cols": [
            AttributeInfo(
                name="title",
                description="title of the wikipedia page",
                type="string",
            ),
            AttributeInfo(
                name="text",
                description="paragraph from this wiki page",
                type="string",
            ),
            AttributeInfo(
                name="views",
                description="number of views",
                type="float"
            ),
        ],
        "must_have_cols": ['id', 'title', 'url', 'text', 'views'],
        "vector_col": "emb",
        "text_col": "text",
        "metadata_col": "metadata",
        "emb_model": lambda: SentenceTransformerEmbeddings(
            model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',)
    },
    'ArXiv Papers': {
        "database": "default",
        "table": "ChatArXiv",
        "hint": hint_arxiv,
        "hint_sql": hint_sql_arxiv,
        "doc_prompt": PromptTemplate(
            input_variables=["page_content", "id", "title", "ref_id",
                             "authors", "pubdate", "categories"],
            template="Title for Doc #{ref_id}: {title}\n\tAbstract: {page_content}\n\tAuthors: {authors}\n\tDate of Publication: {pubdate}\n\tCategories: {categories}\nSOURCE: {id}"),
        "metadata_cols": [
            AttributeInfo(
                name=VirtualColumnName(name="pubdate"),
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
        ],
        "must_have_cols": ['title', 'id', 'categories', 'abstract', 'authors', 'pubdate'],
        "vector_col": "vector",
        "text_col": "abstract",
        "metadata_col": "metadata",
        "emb_model": lambda: HuggingFaceInstructEmbeddings(
            model_name='hkunlp/instructor-xl',
            embed_instruction="Represent the question for retrieving supporting scientific papers: ")
    }
}


def try_eval(x):
    try:
        return eval(x, {'datetime': datetime})
    except:
        return x


def display(dataframe, columns_=None, index=None):
    if len(dataframe) > 0:
        if index:
            dataframe.set_index(index)
        if columns_:
            st.dataframe(dataframe[columns_])
        else:
            st.dataframe(dataframe)
    else:
        st.write("Sorry 😵 we didn't find any articles related to your query.\n\nMaybe the LLM is too naughty that does not follow our instruction... \n\nPlease try again and use verbs that may match the datatype.", unsafe_allow_html=True)


def build_embedding_model(_sel):
    with st.spinner("Loading Model..."):
        embeddings = sel_map[_sel]["emb_model"]()
    return embeddings


def build_retriever(_sel):
    with st.spinner(f"Connecting DB for {_sel}..."):
        myscale_connection = {
            "host": st.secrets['MYSCALE_HOST'],
            "port": st.secrets['MYSCALE_PORT'],
            "username": st.secrets['MYSCALE_USER'],
            "password": st.secrets['MYSCALE_PASSWORD'],
        }

        config = MyScaleSettings(**myscale_connection,
                                 database=sel_map[_sel]["database"],
                                 table=sel_map[_sel]["table"],
                                 column_map={
                                     "id": "id",
                                     "text": sel_map[_sel]["text_col"],
                                     "vector": sel_map[_sel]["vector_col"],
                                     "metadata": sel_map[_sel]["metadata_col"]
                                 })
        doc_search = MyScaleWithoutMetadataJson(st.session_state[f"emb_model_{_sel}"], config, 
                                                must_have_cols=sel_map[_sel]['must_have_cols'])

    with st.spinner(f"Building Self Query Retriever for {_sel}..."):
        metadata_field_info = sel_map[_sel]["metadata_cols"]
        retriever = SelfQueryRetriever.from_llm(
            OpenAI(model_name=query_model_name, openai_api_key=st.secrets['OPENAI_API_KEY'], temperature=0),
            doc_search, "Scientific papers indexes with abstracts. All in English.", metadata_field_info,
            use_original_query=False, structured_query_translator=MyScaleTranslator())

    COMBINE_PROMPT = ChatPromptTemplate.from_strings(
        string_messages=[(SystemMessagePromptTemplate, combine_prompt_template),
                         (HumanMessagePromptTemplate, '{question}')])
    OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

    with st.spinner(f'Building QA Chain with Self-query for {_sel}...'):
        chain = ArXivQAwithSourcesChain(
            retriever=retriever,
            combine_documents_chain=ArXivStuffDocumentChain(
                llm_chain=LLMChain(
                    prompt=COMBINE_PROMPT,
                    llm=ChatOpenAI(model_name=chat_model_name,
                                   openai_api_key=OPENAI_API_KEY, temperature=0.6),
                ),
                document_prompt=sel_map[_sel]["doc_prompt"],
                document_variable_name="summaries",

            ),
            return_source_documents=True,
            max_tokens_limit=12000,
        )

    with st.spinner(f'Building Vector SQL Database Retriever for {_sel}...'):
        MYSCALE_USER = st.secrets['MYSCALE_USER']
        MYSCALE_PASSWORD = st.secrets['MYSCALE_PASSWORD']
        MYSCALE_HOST = st.secrets['MYSCALE_HOST']
        MYSCALE_PORT = st.secrets['MYSCALE_PORT']
        engine = create_engine(
            f'clickhouse://{MYSCALE_USER}:{MYSCALE_PASSWORD}@{MYSCALE_HOST}:{MYSCALE_PORT}/{sel_map[_sel]["database"]}?protocol=https')
        metadata = MetaData(bind=engine)
        PROMPT = PromptTemplate(
            input_variables=["input", "table_info", "top_k"],
            template=_myscale_prompt,
        )
        output_parser = VectorSQLRetrieveCustomOutputParser.from_embeddings(
            model=st.session_state[f'emb_model_{_sel}'], must_have_columns=sel_map[_sel]["must_have_cols"])
        sql_query_chain = VectorSQLDatabaseChain.from_llm(
            llm=OpenAI(model_name=query_model_name, openai_api_key=OPENAI_API_KEY, temperature=0),
            prompt=PROMPT,
            top_k=10,
            return_direct=True,
            db=SQLDatabase(engine, None, metadata, max_string_length=1024),
            sql_cmd_parser=output_parser,
            native_format=True
        )
        sql_retriever = VectorSQLDatabaseChainRetriever(
            sql_db_chain=sql_query_chain, page_content_key=sel_map[_sel]["text_col"])

    with st.spinner(f'Building QA Chain with Vector SQL for {_sel}...'):
        sql_chain = ArXivQAwithSourcesChain(
            retriever=sql_retriever,
            combine_documents_chain=ArXivStuffDocumentChain(
                llm_chain=LLMChain(
                    prompt=COMBINE_PROMPT,
                    llm=ChatOpenAI(model_name=chat_model_name,
                                   openai_api_key=OPENAI_API_KEY, temperature=0.6),
                ),
                document_prompt=sel_map[_sel]["doc_prompt"],
                document_variable_name="summaries",

            ),
            return_source_documents=True,
            max_tokens_limit=12000,
        )

    return {
        "metadata_columns": [{'name': m.name.name if type(m.name) is VirtualColumnName else m.name, 'desc': m.description, 'type': m.type} for m in metadata_field_info],
        "retriever": retriever,
        "chain": chain,
        "sql_retriever": sql_retriever,
        "sql_chain": sql_chain
    }


@st.cache_resource
def build_all():
    sel_map_obj = {}
    for k in sel_map:
        st.session_state[f'emb_model_{k}'] = build_embedding_model(k)
        sel_map_obj[k] = build_retriever(k)
    return sel_map_obj


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

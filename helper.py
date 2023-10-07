from prompts.arxiv_prompt import combine_prompt_template, _myscale_prompt
from typing import Dict, Any
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
from langchain.schema import BaseRetriever
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

# query_model_name = "gpt-3.5-turbo-instruct"
query_model_name = "text-davinci-003"
chat_model_name = "gpt-3.5-turbo-16k"


OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
OPENAI_API_BASE = st.secrets['OPENAI_API_BASE']
MYSCALE_USER = st.secrets['MYSCALE_USER']
MYSCALE_PASSWORD = st.secrets['MYSCALE_PASSWORD']
MYSCALE_HOST = st.secrets['MYSCALE_HOST']
MYSCALE_PORT = st.secrets['MYSCALE_PORT']

COMBINE_PROMPT = ChatPromptTemplate.from_strings(
    string_messages=[(SystemMessagePromptTemplate, combine_prompt_template),
                    (HumanMessagePromptTemplate, '{question}')])

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
            model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',),
        "tool_desc": ("search_among_wikipedia", "Searches among Wikipedia and returns related wiki pages"),
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
            embed_instruction="Represent the question for retrieving supporting scientific papers: "),
        "tool_desc": ("search_among_scientific_papers", "Searches among scientific papers from ArXiv and returns research papers"),
    }
}

def build_embedding_model(_sel):
    """Build embedding model
    """
    with st.spinner("Loading Model..."):
        embeddings = sel_map[_sel]["emb_model"]()
    return embeddings


def build_chains_retrievers(_sel: str) -> Dict[str, Any]:
    """build chains and retrievers

    :param _sel: selected knowledge base
    :type _sel: str
    :return: _description_
    :rtype: Dict[str, Any]
    """
    metadata_field_info = sel_map[_sel]["metadata_cols"]
    retriever = build_self_query(_sel)
    chain = build_qa_chain(_sel, retriever, name="Self Query Retriever")
    sql_retriever = build_vector_sql(_sel)
    sql_chain = build_qa_chain(_sel, sql_retriever, name="Vector SQL")

    return {
        "metadata_columns": [{'name': m.name.name if type(m.name) is VirtualColumnName else m.name, 'desc': m.description, 'type': m.type} for m in metadata_field_info],
        "retriever": retriever,
        "chain": chain,
        "sql_retriever": sql_retriever,
        "sql_chain": sql_chain
    }
    
def build_self_query(_sel: str) -> SelfQueryRetriever:
    """Build self querying retriever

    :param _sel: selected knowledge base
    :type _sel: str
    :return: retriever used by chains
    :rtype: SelfQueryRetriever
    """
    with st.spinner(f"Connecting DB for {_sel}..."):
        myscale_connection = {
            "host": MYSCALE_HOST,
            "port": MYSCALE_PORT,
            "username": MYSCALE_USER,
            "password": MYSCALE_PASSWORD,
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
            OpenAI(model_name=query_model_name, openai_api_key=OPENAI_API_KEY, temperature=0),
            doc_search, "Scientific papers indexes with abstracts. All in English.", metadata_field_info,
            use_original_query=False, structured_query_translator=MyScaleTranslator())
    return retriever

def build_vector_sql(_sel: str)->VectorSQLDatabaseChainRetriever:
    """Build Vector SQL Database Retriever

    :param _sel: selected knowledge base
    :type _sel: str
    :return: retriever used by chains
    :rtype: VectorSQLDatabaseChainRetriever
    """
    with st.spinner(f'Building Vector SQL Database Retriever for {_sel}...'):
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
    return sql_retriever
    
def build_qa_chain(_sel: str, retriever: BaseRetriever, name: str="Self-query") -> ArXivQAwithSourcesChain:
    """_summary_

    :param _sel: selected knowledge base
    :type _sel: str
    :param retriever: retriever used by chains
    :type retriever: BaseRetriever
    :param name: display name, defaults to "Self-query"
    :type name: str, optional
    :return: QA chain interacts with user
    :rtype: ArXivQAwithSourcesChain
    """
    with st.spinner(f'Building QA Chain with {name} for {_sel}...'):
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
    return chain

@st.cache_resource
def build_all() -> Dict[str, Any]:
    """build all resources

    :return: sel_map_obj
    :rtype: Dict[str, Any]
    """
    sel_map_obj = {}
    for k in sel_map:
        st.session_state[f'emb_model_{k}'] = build_embedding_model(k)
        sel_map_obj[k] = build_chains_retrievers(k)
    return sel_map_obj
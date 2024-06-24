from typing import Dict, List
import streamlit as st
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_community.embeddings import SentenceTransformerEmbeddings, HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate

from backend.types.table_config import TableConfig


def hint_arxiv():
    st.markdown("Here we provide some query samples.")
    st.markdown("- If you want to search papers with filters")
    st.markdown("1. ```What is a Bayesian network? Please use articles published later than Feb 2018 and with more "
                "than 2 categories and whose title like `computer` and must have `cs.CV` in its category. ```")
    st.markdown("2. ```What is a Bayesian network? Please use articles published later than Feb 2018```")
    st.markdown("- If you want to ask questions based on arxiv papers stored in MyScaleDB")
    st.markdown("1. ```Did Geoffrey Hinton wrote paper about Capsule Neural Networks?```")
    st.markdown("2. ```Introduce some applications of GANs published around 2019.```")
    st.markdown("3. ```请根据 2019 年左右的文章介绍一下 GAN 的应用都有哪些```")


def hint_sql_arxiv():
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
    st.markdown("Here we provide some query samples.")
    st.markdown("1. ```Which company did Elon Musk found?```")
    st.markdown("2. ```What is Iron Gwazi?```")
    st.markdown("3. ```苹果的发源地是哪里？```")
    st.markdown("4. ```What is a Ring in mathematics?```")
    st.markdown("5. ```The producer of Rick and Morty.```")
    st.markdown("6. ```How low is the temperature on Pluto?```")


def hint_sql_wiki():
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


MYSCALE_TABLES: Dict[str, TableConfig] = {
    'Wikipedia': TableConfig(
        database="wiki",
        table="Wikipedia",
        table_contents="Snapshort from Wikipedia for 2022. All in English.",
        hint=hint_wiki,
        hint_sql=hint_sql_wiki,
        # doc_prompt 对 qa source chain 有用
        doc_prompt=PromptTemplate(
            input_variables=["page_content", "url", "title", "ref_id", "views"],
            template="Title for Doc #{ref_id}: {title}\n\tviews: {views}\n\tcontent: {page_content}\nSOURCE: {url}"
        ),
        metadata_col_attributes=[
            AttributeInfo(name="title", description="title of the wikipedia page", type="string"),
            AttributeInfo(name="text", description="paragraph from this wiki page", type="string"),
            AttributeInfo(name="views", description="number of views", type="float")
        ],
        must_have_col_names=['id', 'title', 'url', 'text', 'views'],
        vector_col_name="emb",
        text_col_name="text",
        metadata_col_name="metadata",
        emb_model=lambda: SentenceTransformerEmbeddings(
            model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        ),
        tool_desc=("search_among_wikipedia", "Searches among Wikipedia and returns related wiki pages")
    ),
    'ArXiv Papers': TableConfig(
        database="default",
        table="ChatArXiv",
        table_contents="Snapshort from Wikipedia for 2022. All in English.",
        hint=hint_arxiv,
        hint_sql=hint_sql_arxiv,
        doc_prompt=PromptTemplate(
            input_variables=["page_content", "id", "title", "ref_id", "authors", "pubdate", "categories"],
            template="Title for Doc #{ref_id}: {title}\n\tAbstract: {page_content}\n\tAuthors: {authors}\n\t"
                     "Date of Publication: {pubdate}\n\tCategories: {categories}\nSOURCE: {id}"
        ),
        metadata_col_attributes=[
            AttributeInfo(name="pubdate", description="The year the paper is published", type="timestamp"),
            AttributeInfo(name="authors", description="List of author names", type="list[string]"),
            AttributeInfo(name="title", description="Title of the paper", type="string"),
            AttributeInfo(name="categories", description="arxiv categories to this paper", type="list[string]"),
            AttributeInfo(name="length(categories)", description="length of arxiv categories to this paper", type="int")
        ],
        must_have_col_names=['title', 'id', 'categories', 'abstract', 'authors', 'pubdate'],
        vector_col_name="vector",
        text_col_name="abstract",
        metadata_col_name="metadata",
        emb_model=lambda: HuggingFaceInstructEmbeddings(
            model_name='hkunlp/instructor-xl',
            embed_instruction="Represent the question for retrieving supporting scientific papers: "
        ),
        tool_desc=(
            "search_among_scientific_papers",
            "Searches among scientific papers from ArXiv and returns research papers"
        )
    )
}

ALL_TABLE_NAME: List[str] = [config.table for config in MYSCALE_TABLES.values()]

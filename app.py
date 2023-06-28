import re
import pandas as pd
from os import environ
import streamlit as st

from langchain.vectorstores import MyScale, MyScaleSettings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI

from prompts.arxiv_prompt import combine_prompt_template
from callbacks.arxiv_callbacks import ChatDataSearchCallBackHandler, ChatDataAskCallBackHandler
from langchain.prompts.prompt import PromptTemplate


environ['TOKENIZERS_PARALLELISM'] = 'true'

st.set_page_config(page_title="ChatData")

st.header("ChatData")

columns = ['title', 'id', 'categories', 'abstract', 'authors', 'pubdate']


def display(dataframe, columns):
    if len(docs) > 0:
        st.dataframe(dataframe[columns])
    else:
        st.write("Sorry 😵 we didn't find any articles related to your query.\nPlease use verbs that may match the datatype.", unsafe_allow_html=True)

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

        with st.spinner('Building RetrievalQAWith SourcesChain...'):
            document_with_metadata_prompt = PromptTemplate(
                input_variables=["page_content", "id", "title", "authors"],
                template="Content:\n\tTitle: {title}\n\tAbstract: {page_content}\n\tAuthors: {authors}\nSOURCE: {id}")
            COMBINE_PROMPT = PromptTemplate(
                template=combine_prompt_template, input_variables=["summaries", "question"])
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=ChatOpenAI(
                    openai_api_key=st.secrets['OPENAI_API_KEY'], temperature=0.6),
                document_prompt=document_with_metadata_prompt,
                combine_prompt=COMBINE_PROMPT,
                retriever=retriever,
                return_source_documents=True,)
    return [{'name': m.name, 'desc': m.description, 'type': m.type} for m in metadata_field_info], retriever, chain


if 'retriever' not in st.session_state:
    st.session_state['metadata_columns'], \
        st.session_state['retriever'], \
        st.session_state['chain'] = \
        build_retriever()
st.info("We provides you metadata columns below for query. Please choose a natural expression to describe filters on those columns.\n\n" +
        "For example: \n\n- What is a Bayesian network? Please use articles published later than Feb 2018 and with more than 2 categories and whose title like `computer` and must have `cs.CV` in its category.\n" +
        "- What is neural network? Please use articles published by Geoffrey Hinton after 2018.\n" +
        "- Introduce some applications of GANs published around 2019.")
st.info("You can retrieve papers with button `Query` or ask questions based on retrieved papers with button `Ask`.", icon='💡')
st.dataframe(st.session_state.metadata_columns)
st.text_input("Ask a question:", key='query')
cols = st.columns([1, 1, 7])
cols[0].button("Query", key='search')
cols[1].button("Ask", key='ask')
plc_hldr = st.empty()

if st.session_state.search:
    plc_hldr = st.empty()
    with plc_hldr.expander('Query Log', expanded=True):
        call_back = None
        callback = ChatDataSearchCallBackHandler()
        try:
            docs = st.session_state.retriever.get_relevant_documents(
                st.session_state.query, callbacks=[callback])
            callback.progress_bar.progress(value=1.0, text="Done!")
            docs = pd.DataFrame(
                [{**d.metadata, 'abstract': d.page_content} for d in docs])

            display(docs, columns)
        except Exception as e:
            st.write('Oops 😵 Something bad happened...')
            # raise e

if st.session_state.ask:
    plc_hldr = st.empty()
    ctx = st.container()
    with plc_hldr.expander('Chat Log', expanded=True):
        call_back = None
        callback = ChatDataAskCallBackHandler()
        try:
            ret = st.session_state.chain(
                st.session_state.query, callbacks=[callback])
            callback.progress_bar.progress(value=1.0, text="Done!")
            st.markdown(
                f"### Answer from LLM\n{ret['answer']}\n### References")
            docs = ret['source_documents']
            ref = re.findall(
                '(http://arxiv.org/abs/\d{4}.\d+v\d)', ret['sources'])
            docs = pd.DataFrame([{**d.metadata, 'abstract': d.page_content}
                                for d in docs if d.metadata['id'] in ref])
            display(docs, columns)
        except Exception as e:
            st.write('Oops 😵 Something bad happened...')
            # raise e

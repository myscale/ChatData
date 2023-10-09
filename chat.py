import json
import time
import pandas as pd
from os import environ
import datetime
import streamlit as st
from langchain.schema import Document

from callbacks.arxiv_callbacks import ChatDataSelfSearchCallBackHandler, \
    ChatDataSelfAskCallBackHandler, ChatDataSQLSearchCallBackHandler, \
    ChatDataSQLAskCallBackHandler

from langchain.schema import BaseMessage, HumanMessage, AIMessage, FunctionMessage, SystemMessage
from auth0_component import login_button


from helper import build_tools, build_agents, build_all, sel_map, display

environ['OPENAI_API_BASE'] = st.secrets['OPENAI_API_BASE']

st.set_page_config(page_title="ChatData", page_icon="https://myscale.com/favicon.ico")
st.header("ChatData")


if 'retriever' not in st.session_state:
    st.session_state["sel_map_obj"] = build_all()
    st.session_state["tools"] = build_tools()

def on_chat_submit():
    ret = st.session_state.agents[st.session_state.sel][st.session_state.ret_type]({"input": st.session_state.chat_input})
    print(ret)
    
def clear_history():
    st.session_state.agents[st.session_state.sel][st.session_state.ret_type].memory.clear()

AUTH0_CLIENT_ID = st.secrets['AUTH0_CLIENT_ID']
AUTH0_DOMAIN = st.secrets['AUTH0_DOMAIN']

def login():
    if "user_name" in st.session_state or ("jump_query_ask" in st.session_state and st.session_state.jump_query_ask):
        return True
    st.subheader("🤗 Welcom to [MyScale](https://myscale.com)'s [ChatData](https://github.com/myscale/ChatData)! 🤗 ")
    st.write("You can now chat with ArXiv and Wikipedia! You can also try to build your RAG system with those knowledge base via [our public read-only credentials!](https://github.com/myscale/ChatData#data-schema) 🌟\n")
    st.write("Built purely with streamlit 👑 , LangChain 🦜🔗 and love for AI!")
    st.write("Follow us on [Twitter](https://x.com/myscaledb) and [Discord](https://discord.gg/D2qpkqc4Jq)!")
    st.info("We used [Auth0](https://auth0.com) as our identity provider. "
            "We will **NOT** collect any of your conversation in any form for any purpose.")
    st.divider()
    col1, col2 = st.columns(2, gap='large')
    with col1.container():
        st.write("Try out MyScale's Self-query and Vector SQL retrievers!")
        st.write("In this demo, you will be able to see how those retrievers "
                 "**digest** -> **translate** -> **retrieve** -> **answer** to your question!")
        st.write("It is a step-by-step tour to understand RAG pipeline.")
        st.session_state["jump_query_ask"] = st.button("Query / Ask")
    with col2.container():
        st.write("Now with the power of LangChain's Conversantional Agents, we are able to build "
                 "conversational chatbot with RAG! The agent will decide when and what to retrieve "
                 "based on your question!")
        st.write("All those conversation history management and retrievers are provided within one MyScale instance!")
        st.write("Log in to Chat with RAG!")
        login_button(AUTH0_CLIENT_ID, AUTH0_DOMAIN, "auth0")
    if st.session_state.auth0 is not None:
        st.session_state.user_info = dict(st.session_state.auth0) 
        if 'email' in st.session_state.user_info:
            email = st.session_state.user_info["email"]
        else:
            email = f"{st.session_state.user_info['nickname']}@{st.session_state.user_info['sub']}"
        st.session_state["user_name"] = email
        del st.session_state.auth0
        st.experimental_rerun()
    if st.session_state.jump_query_ask:
        st.experimental_rerun()
        
def back_to_main():
    if "user_info" in st.session_state:
        del st.session_state.user_info
    if "user_name" in st.session_state:
        del st.session_state.user_name
    if "jump_query_ask" in st.session_state:
        del st.session_state.jump_query_ask

if login():
    if "user_name" in st.session_state:
        st.session_state["agents"] = build_agents(st.session_state.user_name)
        with st.sidebar:
            st.radio("Retriever Type", ["Self-querying retriever", "Vector SQL"], key="ret_type")
            st.selectbox("Knowledge Base", ["ArXiv Papers", "Wikipedia", "ArXiv + Wikipedia"], key="sel")
            st.button("Clear Chat History", on_click=clear_history)
            st.button("Logout", on_click=back_to_main)
        for msg in st.session_state.agents[st.session_state.sel][st.session_state.ret_type].memory.chat_memory.messages:
            speaker = "user" if isinstance(msg, HumanMessage) else "assistant"
            if isinstance(msg, FunctionMessage):
                with st.chat_message("Knowledge Base", avatar="📖"):
                    print(type(msg.content))
                    st.write(f"*{datetime.datetime.fromtimestamp(msg.additional_kwargs['timestamp']).isoformat()}*")
                    st.write("Retrieved from knowledge base:")
                    st.dataframe(pd.DataFrame.from_records(map(dict, eval(msg.content))))
            else:
                if len(msg.content) > 0:
                    with st.chat_message(speaker):
                        print(type(msg), msg.dict())
                        st.write(f"*{datetime.datetime.fromtimestamp(msg.additional_kwargs['timestamp']).isoformat()}*")
                        st.write(f"{msg.content}")
        st.chat_input("Input Message", on_submit=on_chat_submit, key="chat_input")
    elif "jump_query_ask" in st.session_state and st.session_state.jump_query_ask:
        
        sel = st.selectbox('Choose the knowledge base you want to ask with:',
                        options=['ArXiv Papers', 'Wikipedia'])
        sel_map[sel]['hint']()
        tab_sql, tab_self_query = st.tabs(['Vector SQL', 'Self-Query Retrievers'])
        with tab_sql:
            sel_map[sel]['hint_sql']()
            st.text_input("Ask a question:", key='query_sql')
            cols = st.columns([1, 1, 1, 4])
            cols[0].button("Query", key='search_sql')
            cols[1].button("Ask", key='ask_sql')
            cols[2].button("Back", key='back_sql', on_click=back_to_main)
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
            cols = st.columns([1, 1, 1, 4])
            cols[0].button("Query", key='search_self')
            cols[1].button("Ask", key='ask_self')
            cols[2].button("Back", key='back_self', on_click=back_to_main)
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
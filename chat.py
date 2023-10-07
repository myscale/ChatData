import json
import time
import pandas as pd
from os import environ
import hashlib
import datetime
from typing import Optional, Any, List
import streamlit as st
from sqlalchemy import Column, Text, create_engine, MetaData
from langchain.agents import AgentExecutor
try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from clickhouse_sqlalchemy import (
    Table, make_session, get_declarative_base, types, engines
)
from langchain.schema import Document
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.chat import MessagesPlaceholder
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits.conversational_retrieval.openai_functions import _get_default_system_message
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool


from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_memory import BaseChatMemory

from langchain.schema import BaseMessage, HumanMessage, AIMessage, FunctionMessage, SystemMessage
from langchain.memory import SQLChatMessageHistory
from langchain.memory.chat_message_histories.sql import \
    BaseMessageConverter, DefaultMessageConverter
from langchain.schema.messages import BaseMessage, _message_to_dict, messages_from_dict
from helper import MYSCALE_HOST, MYSCALE_PASSWORD, MYSCALE_PORT, MYSCALE_USER, \
    sel_map, chat_model_name, OPENAI_API_KEY, OPENAI_API_BASE, \
    build_embedding_model, build_self_query, build_vector_sql

environ['OPENAI_API_BASE'] = st.secrets['OPENAI_API_BASE']


def create_message_model(table_name, DynamicBase):  # type: ignore
    """
    Create a message model for a given table name.

    Args:
        table_name: The name of the table to use.
        DynamicBase: The base class to use for the model.

    Returns:
        The model class.

    """

    # Model decleared inside a function to have a dynamic table name
    class Message(DynamicBase):
        __tablename__ = table_name
        id = Column(types.Float64)
        session_id = Column(Text)
        msg_id = Column(Text, primary_key=True)
        type = Column(Text)
        addtionals = Column(Text)
        message = Column(Text)
        __table_args__ = (
            engines.ReplacingMergeTree(
                partition_by='session_id',
                order_by=('id', 'msg_id')),
            {'comment': 'Store Chat History'}
        )

    return Message

class DefaultClickhouseMessageConverter(DefaultMessageConverter):
    """The default message converter for SQLChatMessageHistory."""

    def __init__(self, table_name: str):
        self.model_class = create_message_model(table_name, declarative_base())

    def to_sql_model(self, message: BaseMessage, session_id: str) -> Any:
        tstamp = time.time()
        msg_id = hashlib.sha256(f"{session_id}_{message}_{tstamp}".encode('utf-8')).hexdigest()
        return self.model_class(
            id=tstamp, 
            msg_id=msg_id,
            session_id=session_id, 
            type=message.type,
            addtionals=json.dumps(message.additional_kwargs),
            message=json.dumps({
                "type": message.type, 
                "additional_kwargs": {"timestamp": tstamp},
                "data": message.dict()})
        )
    def from_sql_model(self, sql_message: Any) -> BaseMessage:
        msg_dump = json.loads(sql_message.message)
        msg = messages_from_dict([msg_dump])[0]
        msg.additional_kwargs = msg_dump["additional_kwargs"]
        return msg
    
    def get_sql_model_class(self) -> Any:
        return self.model_class
    


def create_agent_executor(name, session_id, llm, tools, **kwargs):
    name = name.replace(" ", "_")
    conn_str = f'clickhouse://{MYSCALE_USER}:{MYSCALE_PASSWORD}@{MYSCALE_HOST}:{MYSCALE_PORT}'
    chat_memory = SQLChatMessageHistory(
        session_id,
        connection_string=f'{conn_str}/chat?protocol=https',
        custom_message_converter=DefaultClickhouseMessageConverter(name))
    memory = AgentTokenBufferMemory(llm=llm, chat_memory=chat_memory)
    
    _system_message = _get_default_system_message()
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=_system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
    )
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
        **kwargs
    )
    
@st.cache_resource
def build_all():
    """build all resources

    :return: sel_map_obj
    :rtype: Dict[str, Any]
    """
    chat_llm = ChatOpenAI(model_name=chat_model_name, temperature=0.6, openai_api_base=OPENAI_API_BASE, openai_api_key=OPENAI_API_KEY)
    sel_map_obj = {}
    agents = {}
    for k in sel_map:
        st.session_state[f'emb_model_{k}'] = build_embedding_model(k)
        sel_map_obj[k] = {
            "langchain_retriever": create_retriever_tool(build_self_query(k), *sel_map[k]["tool_desc"],),
            "vecsql_retriever": create_retriever_tool(build_vector_sql(k), *sel_map[k]["tool_desc"],),
        }
    for k in [*sel_map.keys(), 'ArXiv + Wikipedia', 'Null']:
        for m, n in [("langchain_retriever", "Self-querying retriever"), ("vecsql_retriever", "Vector SQL")]:
            if k == 'ArXiv + Wikipedia':
                tools = [sel_map_obj[k][m] for k in sel_map.keys()]
            elif k == 'Null':
                tools = []
            else:
                tools = [sel_map_obj[k][m]]
            if k not in agents:
                agents[k] = {}
            agents[k][n] = create_agent_executor(
                "chat_memory",
                f"test_session_{k}",
                chat_llm,
                tools=tools,
                )
            print(f"{k}: {n}", len(agents[k][n].tools))
    return sel_map_obj, agents

if 'retriever' not in st.session_state:
    st.session_state["sel_map_obj"], st.session_state["agents"] = build_all()

def on_chat_submit():
    ret = st.session_state.agents[st.session_state.sel][st.session_state.ret_type]({"input": st.session_state.chat_input})
    print(ret)
    
def clear_history():
    st.session_state.agents[st.session_state.sel][st.session_state.ret_type].memory.clear()

with st.sidebar:
    st.radio("Retriever Type", ["Self-querying retriever", "Vector SQL"], key="ret_type")
    st.selectbox("Knowledge Base", ["ArXiv Papers", "Wikipedia", "ArXiv + Wikipedia", "Null"], key="sel")
    st.button("Clear Chat History", on_click=clear_history)
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
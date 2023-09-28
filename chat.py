import json
import time
from os import environ
import hashlib
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
from langchain.prompts.prompt import PromptTemplate

from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from prompts.arxiv_prompt import combine_prompt_template, _myscale_prompt
from chains.arxiv_chains import VectorSQLRetrieveCustomOutputParser
from langchain_experimental.sql.vector_sql import VectorSQLDatabaseChain
from langchain_experimental.retrievers.vector_sql_database import VectorSQLDatabaseChainRetriever
from langchain.utilities.sql_database import SQLDatabase


from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_memory import BaseChatMemory

from langchain.schema import BaseMessage, HumanMessage, AIMessage, FunctionMessage, SystemMessage, ChatMessage
from langchain.memory import SQLChatMessageHistory
from langchain.memory.chat_message_histories.sql import \
    BaseMessageConverter, DefaultMessageConverter
from langchain.schema.messages import BaseMessage, _message_to_dict, messages_from_dict
environ['OPENAI_API_BASE'] = st.secrets['OPENAI_API_BASE']


# query_model_name = "gpt-3.5-turbo-instruct"
query_model_name = "text-davinci-003"
chat_model_name = "gpt-3.5-turbo-16k"

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
            message=json.dumps(_message_to_dict(message))
        )

MYSCALE_USER = st.secrets['MYSCALE_USER']
MYSCALE_PASSWORD = st.secrets['MYSCALE_PASSWORD']
MYSCALE_HOST = st.secrets['MYSCALE_HOST']
MYSCALE_PORT = st.secrets['MYSCALE_PORT']
OPENAI_API_BASE = st.secrets['OPENAI_API_BASE']
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
conn_str = f'clickhouse://{MYSCALE_USER}:{MYSCALE_PASSWORD}@{MYSCALE_HOST}:{MYSCALE_PORT}'
chat_memory = SQLChatMessageHistory(
    'test',
    connection_string=f'{conn_str}/chat?protocol=https',
    custom_message_converter=DefaultClickhouseMessageConverter('test'))
chat_memory.add_user_message("hi!1")
chat_memory.add_ai_message("whats up?2")

engine = create_engine(f'{conn_str}/wiki?protocol=https')
metadata = MetaData(bind=engine)
PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_myscale_prompt,
)

with st.sidebar:
    st.radio("Retriever Type", ["Self-querying retriever", "Vector SQL"], key="ret_type")
for msg in chat_memory.messages:
    speaker = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(speaker):
        st.write(msg.content)
st.chat_input("")
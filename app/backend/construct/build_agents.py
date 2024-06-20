import os
from typing import Sequence, List

import streamlit as st
from langchain.agents import AgentExecutor
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool

from backend.chat_bot.message_converter import DefaultClickhouseMessageConverter
from backend.constants.prompts import DEFAULT_SYSTEM_PROMPT
from backend.constants.streamlit_keys import AVAILABLE_RETRIEVAL_TOOLS
from backend.constants.variables import GLOBAL_CONFIG, RETRIEVER_TOOLS

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import MessagesPlaceholder
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.memory import SQLChatMessageHistory


def create_agent_executor(
        agent_name: str,
        session_id: str,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        system_prompt: str,
        **kwargs
) -> AgentExecutor:
    agent_name = agent_name.replace(" ", "_")
    conn_str = f'clickhouse://{os.environ["MYSCALE_USER"]}:{os.environ["MYSCALE_PASSWORD"]}@{os.environ["MYSCALE_HOST"]}:{os.environ["MYSCALE_PORT"]}'
    chat_memory = SQLChatMessageHistory(
        session_id,
        connection_string=f'{conn_str}/chat?protocol=http' if GLOBAL_CONFIG.mode == "dev" else f'{conn_str}/chat?protocol=https',
        custom_message_converter=DefaultClickhouseMessageConverter(agent_name))
    memory = AgentTokenBufferMemory(llm=llm, chat_memory=chat_memory)

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=SystemMessage(content=system_prompt),
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


def build_agents(
        session_id: str,
        tool_names: List[str],
        model: str = "gpt-3.5-turbo-0125",
        temperature: float = 0.6,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT
):
    chat_llm = ChatOpenAI(
        model_name=model,
        temperature=temperature,
        base_url=GLOBAL_CONFIG.openai_api_base,
        api_key=GLOBAL_CONFIG.openai_api_key,
        streaming=True
    )
    tools = st.session_state.get(AVAILABLE_RETRIEVAL_TOOLS, st.session_state.get(RETRIEVER_TOOLS))
    selected_tools = [tools[k] for k in tool_names]
    agent = create_agent_executor(
        agent_name="chat_memory",
        session_id=session_id,
        llm=chat_llm,
        tools=selected_tools,
        system_prompt=system_prompt
    )
    return agent

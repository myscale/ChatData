import hashlib
import json
import time
from typing import Any

from langchain.memory.chat_message_histories.sql import DefaultMessageConverter
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage, ChatMessage, FunctionMessage
from langchain.schema.messages import ToolMessage
from sqlalchemy.orm import declarative_base

from backend.chat_bot.tools import create_message_history_table


def _message_from_dict(message: dict) -> BaseMessage:
    _type = message["type"]
    if _type == "human":
        return HumanMessage(**message["data"])
    elif _type == "ai":
        return AIMessage(**message["data"])
    elif _type == "system":
        return SystemMessage(**message["data"])
    elif _type == "chat":
        return ChatMessage(**message["data"])
    elif _type == "function":
        return FunctionMessage(**message["data"])
    elif _type == "tool":
        return ToolMessage(**message["data"])
    elif _type == "AIMessageChunk":
        message["data"]["type"] = "ai"
        return AIMessage(**message["data"])
    else:
        raise ValueError(f"Got unexpected message type: {_type}")


class DefaultClickhouseMessageConverter(DefaultMessageConverter):
    """The default message converter for SQLChatMessageHistory."""

    def __init__(self, table_name: str):
        super().__init__(table_name)
        self.model_class = create_message_history_table(table_name, declarative_base())

    def to_sql_model(self, message: BaseMessage, session_id: str) -> Any:
        time_stamp = time.time()
        msg_id = hashlib.sha256(
            f"{session_id}_{message}_{time_stamp}".encode('utf-8')).hexdigest()
        user_id, _ = session_id.split("?")
        return self.model_class(
            id=time_stamp,
            msg_id=msg_id,
            user_id=user_id,
            session_id=session_id,
            type=message.type,
            addtionals=json.dumps(message.additional_kwargs),
            message=json.dumps({
                "type": message.type,
                "additional_kwargs": {"timestamp": time_stamp},
                "data": message.dict()})
        )

    def from_sql_model(self, sql_message: Any) -> BaseMessage:
        msg_dump = json.loads(sql_message.message)
        msg = _message_from_dict(msg_dump)
        msg.additional_kwargs = msg_dump["additional_kwargs"]
        return msg

    def get_sql_model_class(self) -> Any:
        return self.model_class

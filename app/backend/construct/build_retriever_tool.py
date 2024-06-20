import json
from typing import List

from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import BaseRetriever, Document
from langchain.tools import Tool

from backend.chat_bot.json_decoder import CustomJSONEncoder


class RetrieverInput(BaseModel):
    query: str = Field(description="query to look up in retriever")


def create_retriever_tool(
        retriever: BaseRetriever,
        tool_name: str,
        description: str
) -> Tool:
    """Create a tool to do retrieval of documents.

    Args:
        retriever: The retriever to use for the retrieval
        tool_name: The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        description: The description for the tool. This will be passed to the language
            model, so should be descriptive.

    Returns:
        Tool class to pass to an agent
    """
    def wrap(func):
        def wrapped_retrieve(*args, **kwargs):
            docs: List[Document] = func(*args, **kwargs)
            return json.dumps([d.dict() for d in docs], cls=CustomJSONEncoder)

        return wrapped_retrieve

    return Tool(
        name=tool_name,
        description=description,
        func=wrap(retriever.get_relevant_documents),
        coroutine=retriever.aget_relevant_documents,
        args_schema=RetrieverInput,
    )

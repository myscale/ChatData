from typing import Callable
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.prompts import PromptTemplate
from dataclasses import dataclass
from typing import List


@dataclass
class TableConfig:
    database: str
    table: str
    table_contents: str
    # column names
    must_have_col_names: List[str]
    vector_col_name: str
    text_col_name: str
    metadata_col_name: str
    # hint for UI
    hint: Callable
    hint_sql: Callable
    # for langchain
    doc_prompt: PromptTemplate
    metadata_col_attributes: List[AttributeInfo]
    emb_model: Callable
    tool_desc: tuple

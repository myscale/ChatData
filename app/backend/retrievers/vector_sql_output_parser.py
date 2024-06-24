from typing import Dict, Any, List

from langchain_experimental.sql.vector_sql import VectorSQLOutputParser


class VectorSQLRetrieveOutputParser(VectorSQLOutputParser):
    """Based on VectorSQLOutputParser
    It also modify the SQL to get all columns
    """
    must_have_columns: List[str]

    @property
    def _type(self) -> str:
        return "vector_sql_retrieve_custom"

    def parse(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        start = text.upper().find("SELECT")
        if start >= 0:
            end = text.upper().find("FROM")
            text = text.replace(
                text[start + len("SELECT") + 1: end - 1], ", ".join(self.must_have_columns))
        return super().parse(text)

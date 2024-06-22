from typing import Any, Optional, List

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.myscale import MyScale, MyScaleSettings

from logger import logger


class MyScaleWithoutMetadataJson(MyScale):
    def __init__(self, embedding: Embeddings, config: Optional[MyScaleSettings] = None, must_have_cols: List[str] = [],
                 **kwargs: Any) -> None:
        try:
            super().__init__(embedding, config, **kwargs)
        except Exception as e:
            logger.error(e)
        self.must_have_cols: List[str] = must_have_cols

    def _build_qstr(
            self, q_emb: List[float], topk: int, where_str: Optional[str] = None
    ) -> str:
        q_emb_str = ",".join(map(str, q_emb))
        if where_str:
            where_str = f"PREWHERE {where_str}"
        else:
            where_str = ""

        q_str = f"""
            SELECT {self.config.column_map['text']}, dist, {','.join(self.must_have_cols)}
            FROM {self.config.database}.{self.config.table}
            {where_str}
            ORDER BY distance({self.config.column_map['vector']}, [{q_emb_str}]) 
                AS dist {self.dist_order}
            LIMIT {topk}
            """
        return q_str

    def similarity_search_by_vector(self, embedding: List[float], k: int = 4, where_str: Optional[str] = None,
                                    **kwargs: Any) -> List[Document]:
        q_str = self._build_qstr(embedding, k, where_str)
        try:
            return [
                Document(
                    page_content=r[self.config.column_map["text"]],
                    metadata={k: r[k] for k in self.must_have_cols},
                )
                for r in self.client.query(q_str).named_results()
            ]
        except Exception as e:
            logger.error(
                f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            return []

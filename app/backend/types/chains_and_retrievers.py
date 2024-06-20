from typing import Dict
from dataclasses import dataclass
from typing import List, Any
from langchain.retrievers import SelfQueryRetriever
from langchain_experimental.retrievers.vector_sql_database import VectorSQLDatabaseChainRetriever

from backend.chains.retrieval_qa_with_sources import CustomRetrievalQAWithSourcesChain


@dataclass
class MetadataColumn:
    name: str
    desc: str
    type: str


@dataclass
class ChainsAndRetrievers:
    metadata_columns: List[MetadataColumn]
    retriever: SelfQueryRetriever
    chain: CustomRetrievalQAWithSourcesChain
    sql_retriever: VectorSQLDatabaseChainRetriever
    sql_chain: CustomRetrievalQAWithSourcesChain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata_columns": self.metadata_columns,
            "retriever": self.retriever,
            "chain": self.chain,
            "sql_retriever": self.sql_retriever,
            "sql_chain": self.sql_chain
        }



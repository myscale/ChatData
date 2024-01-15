import logging
import inspect
from typing import Dict, Any, Optional, List, Tuple


from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever
from langchain.callbacks.manager import Callbacks
from langchain.schema.prompt_template import format_document
from langchain.docstore.document import Document
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.vectorstores.myscale import MyScale, MyScaleSettings
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from langchain_experimental.sql.vector_sql import VectorSQLOutputParser

logger = logging.getLogger()

class MyScaleWithoutMetadataJson(MyScale):
    def __init__(self, embedding: Embeddings, config: Optional[MyScaleSettings] = None, must_have_cols: List[str] = [], **kwargs: Any) -> None:
        super().__init__(embedding, config, **kwargs)
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
    
    def similarity_search_by_vector(self, embedding: List[float], k: int = 4, where_str: Optional[str] = None, **kwargs: Any) -> List[Document]:
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
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            return []

class VectorSQLRetrieveCustomOutputParser(VectorSQLOutputParser):
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
            text = text.replace(text[start + len("SELECT") + 1 : end - 1], ", ".join(self.must_have_columns))
        return super().parse(text)

class ArXivStuffDocumentChain(StuffDocumentsChain):
    """Combine arxiv documents with PDF reference number"""

    def _get_inputs(self, docs: List[Document], **kwargs: Any) -> dict:
        """Construct inputs from kwargs and docs.

        Format and the join all the documents together into one input with name
        `self.document_variable_name`. The pluck any additional variables
        from **kwargs.

        Args:
            docs: List of documents to format and then join into single input
            **kwargs: additional inputs to chain, will pluck any other required
                arguments from here.

        Returns:
            dictionary of inputs to LLMChain
        """
        # Format each document according to the prompt
        doc_strings = []
        for doc_id, doc in enumerate(docs):
            # add temp reference number in metadata
            doc.metadata.update({'ref_id': doc_id})
            doc.page_content = doc.page_content.replace('\n', ' ')
            doc_strings.append(format_document(doc, self.document_prompt))
        # Join the documents together to put them in the prompt.
        inputs = {
            k: v
            for k, v in kwargs.items()
            if k in self.llm_chain.prompt.input_variables
        }
        inputs[self.document_variable_name] = self.document_separator.join(
            doc_strings)
        return inputs

    def combine_docs(
        self, docs: List[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> Tuple[str, dict]:
        """Stuff all documents into one prompt and pass to LLM.

        Args:
            docs: List of documents to join together into one variable
            callbacks: Optional callbacks to pass along
            **kwargs: additional parameters to use to get inputs to LLMChain.

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """
        inputs = self._get_inputs(docs, **kwargs)
        # Call predict on the LLM.
        output = self.llm_chain.predict(callbacks=callbacks, **inputs)
        return output, {}

    @property
    def _chain_type(self) -> str:
        return "referenced_stuff_documents_chain"


class ArXivQAwithSourcesChain(RetrievalQAWithSourcesChain):
    """QA with source chain for Chat ArXiv app with references

    This chain will automatically assign reference number to the article,
    Then parse it back to titles or anything else.
    """

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(inputs, run_manager=_run_manager)
        else:
            docs = self._get_docs(inputs)  # type: ignore[call-arg]

        answer = self.combine_documents_chain.run(
            input_documents=docs, callbacks=_run_manager.get_child(), **inputs
        )
        # parse source with ref_id
        sources = []
        ref_cnt = 1
        for d in docs:
            ref_id = d.metadata['ref_id']
            if f"Doc #{ref_id}" in answer:
                answer = answer.replace(f"Doc #{ref_id}", f"#{ref_id}")
            if f"#{ref_id}" in answer:
                title = d.metadata['title'].replace('\n', '')
                d.metadata['ref_id'] = ref_cnt
                answer = answer.replace(f"#{ref_id}", f"{title} [{ref_cnt}]")
                sources.append(d)
                ref_cnt += 1
                
        
        result: Dict[str, Any] = {
            self.answer_key: answer,
            self.sources_answer_key: sources,
        }
        if self.return_source_documents:
            result["source_documents"] = docs
        return result

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def _chain_type(self) -> str:
        return "arxiv_qa_with_sources_chain"
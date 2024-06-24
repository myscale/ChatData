import inspect
from typing import Dict, Any, Optional, List

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.docstore.document import Document

from logger import logger


class CustomRetrievalQAWithSourcesChain(RetrievalQAWithSourcesChain):
    """QA with source chain for Chat ArXiv app with references

    This chain will automatically assign reference number to the article,
    Then parse it back to titles or anything else.
    """

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        logger.info(f"\033[91m\033[1m{self._chain_type}\033[0m")
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs: List[Document] = self._get_docs(inputs, run_manager=_run_manager)
        else:
            docs: List[Document] = self._get_docs(inputs)  # type: ignore[call-arg]

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
        return "custom_retrieval_qa_with_sources_chain"

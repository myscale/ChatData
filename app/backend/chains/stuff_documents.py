from typing import Any, List, Tuple

from langchain.callbacks.manager import Callbacks
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document
from langchain.schema.prompt_template import format_document


class CustomStuffDocumentChain(StuffDocumentsChain):
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
        return "custom_stuff_document_chain"

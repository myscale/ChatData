from typing import Dict, Any, List

import streamlit as st
from langchain.callbacks.streamlit.streamlit_callback_handler import (
    StreamlitCallbackHandler,
)
from langchain.schema.output import LLMResult


class CustomSelfQueryRetrieverCallBackHandler(StreamlitCallbackHandler):
    def __init__(self):
        super().__init__(st.container())
        self._current_thought = None
        self.progress_bar = st.progress(value=0.0, text="Executing ChatData SelfQuery...")

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        self.progress_bar.progress(value=0.35, text="Communicate with LLM...")
        pass

    def on_chain_end(self, outputs, **kwargs) -> None:
        if len(kwargs['tags']) == 0:
            self.progress_bar.progress(value=0.75, text="Searching in DB...")
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        st.markdown("### Generate filter by LLM \n"
                    "> Here we get `query_constructor` results \n\n")
        self.progress_bar.progress(value=0.5, text="Generate filter by LLM...")
        for item in response.generations:
            st.markdown(f"{item[0].text}")
        pass


class ChatDataSelfAskCallBackHandler(StreamlitCallbackHandler):
    def __init__(self) -> None:
        super().__init__(st.container())
        self.progress_bar = st.progress(value=0.2, text="Executing ChatData SelfQuery Chain...")

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:

        if len(kwargs['tags']) != 0:
            self.progress_bar.progress(value=0.5, text="We got filter info from LLM...")
            st.markdown("### Generate filter by LLM \n"
                        "> Here we get `query_constructor` results \n\n")
            for item in response.generations:
                st.markdown(f"{item[0].text}")
        pass

    def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        cid = ".".join(serialized["id"])
        if cid.endswith(".CustomStuffDocumentChain"):
            self.progress_bar.progress(value=0.7, text="Asking LLM with related documents...")

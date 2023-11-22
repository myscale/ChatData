import streamlit as st
import json
import textwrap
from typing import Dict, Any, List
from sql_formatter.core import format_sql
from langchain.callbacks.streamlit.streamlit_callback_handler import (
    LLMThought,
    StreamlitCallbackHandler,
)
from langchain.schema.output import LLMResult
from streamlit.delta_generator import DeltaGenerator


class ChatDataSelfSearchCallBackHandler(StreamlitCallbackHandler):
    def __init__(self) -> None:
        self.progress_bar = st.progress(value=0.0, text="Working...")
        self.tokens_stream = ""

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        pass

    def on_text(self, text: str, **kwargs) -> None:
        self.progress_bar.progress(value=0.2, text="Asking LLM...")

    def on_chain_end(self, outputs, **kwargs) -> None:
        self.progress_bar.progress(value=0.6, text="Searching in DB...")
        if "repr" in outputs:
            st.markdown("### Generated Filter")
            st.markdown(f"```python\n{outputs['repr']}\n```", unsafe_allow_html=True)

    def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        pass


class ChatDataSelfAskCallBackHandler(StreamlitCallbackHandler):
    def __init__(self) -> None:
        self.progress_bar = st.progress(value=0.0, text="Searching DB...")
        self.status_bar = st.empty()
        self.prog_value = 0.0
        self.prog_map = {
            "langchain.chains.qa_with_sources.retrieval.RetrievalQAWithSourcesChain": 0.2,
            "langchain.chains.combine_documents.map_reduce.MapReduceDocumentsChain": 0.4,
            "langchain.chains.combine_documents.stuff.StuffDocumentsChain": 0.8,
        }

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        pass

    def on_text(self, text: str, **kwargs) -> None:
        pass

    def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        cid = ".".join(serialized["id"])
        if cid != "langchain.chains.llm.LLMChain":
            self.progress_bar.progress(
                value=self.prog_map[cid], text=f"Running Chain `{cid}`..."
            )
            self.prog_value = self.prog_map[cid]
        else:
            self.prog_value += 0.1
            self.progress_bar.progress(
                value=self.prog_value, text=f"Running Chain `{cid}`..."
            )

    def on_chain_end(self, outputs, **kwargs) -> None:
        pass


class ChatDataSQLSearchCallBackHandler(StreamlitCallbackHandler):
    def __init__(self) -> None:
        self.progress_bar = st.progress(value=0.0, text="Writing SQL...")
        self.status_bar = st.empty()
        self.prog_value = 0
        self.prog_interval = 0.2

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        pass

    def on_llm_end(
        self,
        response: LLMResult,
        *args,
        **kwargs,
    ):
        text = response.generations[0][0].text
        if text.replace(" ", "").upper().startswith("SELECT"):
            st.write("We generated Vector SQL for you:")
            st.markdown(f"""```sql\n{format_sql(text, max_len=80)}\n```""")
            print(f"Vector SQL: {text}")
            self.prog_value += self.prog_interval
            self.progress_bar.progress(value=self.prog_value, text="Searching in DB...")

    def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        cid = ".".join(serialized["id"])
        self.prog_value += self.prog_interval
        self.progress_bar.progress(
            value=self.prog_value, text=f"Running Chain `{cid}`..."
        )

    def on_chain_end(self, outputs, **kwargs) -> None:
        pass


class ChatDataSQLAskCallBackHandler(ChatDataSQLSearchCallBackHandler):
    def __init__(self) -> None:
        self.progress_bar = st.progress(value=0.0, text="Writing SQL...")
        self.status_bar = st.empty()
        self.prog_value = 0
        self.prog_interval = 0.1


class LLMThoughtWithKB(LLMThought):
    def on_tool_end(
        self,
        output: str,
        color=None,
        observation_prefix=None,
        llm_prefix=None,
        **kwargs: Any,
    ) -> None:
        try:
            self._container.markdown(
                "\n\n".join(
                    ["### Retrieved Documents:"]
                    + [
                        f"**{i+1}**: {textwrap.shorten(r['page_content'], width=80)}"
                        for i, r in enumerate(json.loads(output))
                    ]
                )
            )
        except Exception as e:
            super().on_tool_end(output, color, observation_prefix, llm_prefix, **kwargs)


class ChatDataAgentCallBackHandler(StreamlitCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        if self._current_thought is None:
            self._current_thought = LLMThoughtWithKB(
                parent_container=self._parent_container,
                expanded=self._expand_new_thoughts,
                collapse_on_complete=self._collapse_completed_thoughts,
                labeler=self._thought_labeler,
            )

        self._current_thought.on_llm_start(serialized, prompts)

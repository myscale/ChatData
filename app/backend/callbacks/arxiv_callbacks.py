import json
import textwrap
from typing import Dict, Any, List

from langchain.callbacks.streamlit.streamlit_callback_handler import (
    LLMThought,
    StreamlitCallbackHandler,
)


class LLMThoughtWithKnowledgeBase(LLMThought):
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
            self._current_thought = LLMThoughtWithKnowledgeBase(
                parent_container=self._parent_container,
                expanded=self._expand_new_thoughts,
                collapse_on_complete=self._collapse_completed_thoughts,
                labeler=self._thought_labeler,
            )

        self._current_thought.on_llm_start(serialized, prompts)

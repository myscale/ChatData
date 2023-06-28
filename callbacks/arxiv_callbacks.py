import streamlit as st
from langchain.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler

class ChatDataSearchCallBackHandler(StreamlitCallbackHandler):
    def __init__(self) -> None:
        self.progress_bar = st.progress(value=0.0, text="Working...")
        self.tokens_stream = ""
    
    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        pass
        
    def on_text(self, text: str, **kwargs) -> None:
        self.progress_bar.progress(value=0.2, text="Asking LLM...")
        
    def on_chain_end(self, outputs, **kwargs) -> None:
        self.progress_bar.progress(value=0.6, text='Searching in DB...')
        st.markdown('### Generated Filter')
        st.write(outputs['text'], unsafe_allow_html=True)
    
    def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        pass

class ChatDataAskCallBackHandler(StreamlitCallbackHandler):
    def __init__(self) -> None:
        self.progress_bar = st.progress(value=0.0, text='Searching DB...')
        self.status_bar = st.empty()
        self.prog_value = 0.0
        self.prog_map = {
            'langchain.chains.qa_with_sources.retrieval.RetrievalQAWithSourcesChain': 0.2,
            'langchain.chains.combine_documents.map_reduce.MapReduceDocumentsChain': 0.4,
            'langchain.chains.combine_documents.stuff.StuffDocumentsChain': 0.8
        }

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        pass
        
    def on_text(self, text: str, **kwargs) -> None:
        pass
        
    def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        cid = '.'.join(serialized['id']) 
        if cid != 'langchain.chains.llm.LLMChain':
            self.progress_bar.progress(value=self.prog_map[cid], text=f'Running Chain `{cid}`...')
            self.prog_value = self.prog_map[cid]
        else:
            self.prog_value += 0.1
            self.progress_bar.progress(value=self.prog_value, text=f'Running Chain `{cid}`...')

    def on_chain_end(self, outputs, **kwargs) -> None:
        pass
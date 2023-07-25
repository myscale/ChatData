from typing import List
from langchain.embeddings.base import Embeddings
from transformers import CLIPModel, CLIPTokenizerFast

class HuggingfaceClipModel(Embeddings):
    def __init__(self, model_name = "openai/clip-vit-base-patch32") -> None:
        model_name = "openai/clip-vit-base-patch32"
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self.clip = CLIPModel.from_pretrained(model_name)
    
    def embed_query(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors='pt')
        out = self.clip.get_text_features(**inputs)
        xq = out.squeeze(0).cpu().detach().numpy().tolist()
        return xq
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_query(texts)
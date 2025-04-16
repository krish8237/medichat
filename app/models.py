from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
import torch

class EmbeddingGenerator:
    def __init__(self, model_name='sentence-transformers/sentence-t5-base'):
        self.model = SentenceTransformer(model_name)

    def generate(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

class ReRanker:
    def __init__(self, model_name='colbert-ir/colbertv2.0'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def rerank(self, query, docs, top_k=3):
        inputs = self.tokenizer(
            [query] * len(docs), docs, return_tensors="pt",
            padding=True, truncation=True
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            scores = logits[:, 1]  # Use positive class score
        sorted_indices = torch.argsort(scores, descending=True)
        return [docs[i.item()] for i in sorted_indices[:min(top_k, len(docs))]]
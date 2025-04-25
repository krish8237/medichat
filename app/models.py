from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name='sentence-transformers/sentence-t5-base'):
        self.embeddings = HuggingFaceEmbeddings(
            model_name = model_name,
            model_kwargs = {'device':'cpu'},
            encode_kwargs={'normalize_embeddings':False})

    def model(self):
        return self.embeddings

class ReRanker:
    def __init__(self, model_name='colbert-ir/colbertv2.0'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def __format_context(self, docs):
        context = "--------------- Information -------------------"
        for doc in docs:
            context += f"{doc.page_content}\nsrc url: {doc.metadata.get('url')}\n\n"
        return context

    def rerank(self, query, docs, top_k1):
        inputs = self.tokenizer(
            [query] * len(docs),
            [doc.page_content for doc in docs],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1).cpu().numpy()

        ranked_indices = np.argsort(scores)[::-1].flatten()[:top_k1]  # Added .flatten()
        reranked_docs = [docs[int(i)] for i in ranked_indices]  # Explicitly cast indices to int
        return reranked_docs, self.__format_context(reranked_docs)

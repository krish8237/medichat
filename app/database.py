import faiss
import numpy as np

class FAISSIndex:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def add(self, embeddings, documents):
        self.index.add(np.array(embeddings))
        self.documents = documents

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(np.array(query_embedding), top_k)
        return [self.documents[i] for i in indices[0]]
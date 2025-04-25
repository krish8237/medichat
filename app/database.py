from langchain_community.vectorstores import FAISS
import os

class FAISSdb:
    def __init__(self, documents, model, path):
        self.model = model
        self.documents = documents
        self.vector_db = None
        self.vector_db_path = path

    def process(self):
        if(os.path.exists(os.path.join(self.vector_db_path, "index.faiss"))):
            self.vector_db = FAISS.load_local(self.vector_db_path, self.model, allow_dangerous_deserialization=True)
        else:
            self.vector_db = FAISS.from_documents(documents=self.documents, embedding=self.model)

    def save(self):
        self.vector_db.save_local(self.vector_db_path)

    def search(self, query, top_k=10):
        docs = self.vector_db.similarity_search(query, top_k)
        return docs
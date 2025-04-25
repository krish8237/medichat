from dataset import MedicalDatasetLoader
from models import EmbeddingGenerator, ReRanker
from database import FAISSdb
from rag import HfllmsPipeline

class MediChatClient:

    def __init__(self, dataset_path, vector_db_path, token_size=50):
        self.dataset_path = dataset_path
        self.vector_db_path = vector_db_path
        self.dataset_loader = self.embedder = self.vector_db = self.reranker = self.hf_llms_pipeline = None
        self.token_size = token_size

    def setup(self):
        dataset_loader = MedicalDatasetLoader(self.dataset_path)
        self.embedder = EmbeddingGenerator()
        docs_sample = dataset_loader.load_documents()
        print(len(docs_sample))
        print(docs_sample[0].page_content)
        self.vector_db = FAISSdb(docs_sample, self.embedder.model(), self.vector_db_path)
        self.vector_db.process()
        self.vector_db.save()
        self.reranker = ReRanker()
        self.hf_llms_pipeline = HfllmsPipeline(self.token_size)

    def __generate_prompt(self, query, context):
        return f"Answer the query (in {self.token_size} words) by considering the following context\nContext: {context}\n\nQuery: {query}"

    def ask(self, query, term, tokens=50):
        response = {}
        docs = self.vector_db.search(query)
        print(docs)
        reranked_docs, context = self.reranker.rerank(query, docs, 5)
        print(reranked_docs)
        if(term != None):
            response["docs"] = docs
            response["context"] = context
            response["reranked_docs"] = reranked_docs
        else:
            response["results"] = self.hf_llms_pipeline.generate_results(self.__generate_prompt(query, context), tokens)
        return response
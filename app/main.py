from dataset import MedicalDatasetLoader
from models import EmbeddingGenerator, ReRanker
from database import FAISSIndex
from rag import RAGGenerator

class MedicalChatbot:
    def __init__(self, dataset_loader, embedder, index_store, reranker, rag):
        self.dataset_loader = dataset_loader
        self.embedder = embedder
        self.index_store = index_store
        self.reranker = reranker
        self.rag = rag

    def setup(self):
        docs = self.dataset_loader.load_documents()
        embeddings = self.embedder.generate(docs)
        self.index_store.add(embeddings, docs)

    def ask(self, query):
        query_embedding = self.embedder.generate([query])
        top_docs = self.index_store.search(query_embedding, top_k=10)
        reranked = self.reranker.rerank(query, top_docs)
        print(reranked)
        return self.rag.generate_answer(query, "\n".join(reranked))


if __name__ == "__main__":
    dataset_loader = MedicalDatasetLoader(r"../data/processed_data/1_CancerGov_QA.csv", r"../data/processed_data/10_MPlus_ADAM_QA.csv")
    embedder = EmbeddingGenerator()
    docs_sample = dataset_loader.load_documents()[:2]
    temp_embeddings = embedder.generate(docs_sample)
    print(temp_embeddings)
    index_store = FAISSIndex(dimension=len(temp_embeddings[0]))
    index_store.add(temp_embeddings, docs_sample)
    print("Done DB store")
    reranker = ReRanker()
    print("Done Reranking")
    rag = RAGGenerator()
    print("Done RAG Generation")
    bot = MedicalChatbot(dataset_loader, embedder, index_store, reranker, rag)
    bot.setup()

    query = "What are the symptoms of diabetes?"
    print("Bot:", bot.ask(query))
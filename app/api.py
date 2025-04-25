from fastapi import FastAPI
from evaluate import Evaluator

class MediChatAPI:

    def __init__(self, client):
        self.app = FastAPI()
        self.client = client
        self.evaluator = Evaluator(client, "../data/test_data/test.csv")
        self.setup_routes()

    def ask_query(self, query, term=None, tokens=50):
        data = self.client.ask(query, term, tokens)
        if term and term != "all":
            return data[term]
        return data

    def setup_routes(self):
        @self.app.get("/chat/context/")
        async def context(query: str):
            return {
                "query" : query,
                "context" : self.ask_query(query, term = "context")
            }

        @self.app.get("/chat/docs/all/")
        async def all_docs(query: str):
            return {
                "query" : query,
                "All Documents" : self.ask_query(query, term = "docs")
            }

        @self.app.get("/chat/docs/re-ranked/")
        async def reranked_docs(query: str):
            return {
                "query": query,
                "Re-ranked Documents": self.ask_query(query, term="reranked_docs")
            }

        @self.app.get("/chat/docs/")
        async def chat_docs(query : str):
            return self.ask_query(query, term="all")

        @self.app.get("/chat/")
        async def chat(query : str, tokens : int = 50):
            return self.ask_query(query, None, tokens)

        @self.app.get("/evaluate/")
        async def calculate_metrics(test_size : int = 3):
            return self.evaluator.evaluate(test_size)

        @self.app.get("/")
        async def greet():
            return {"message" : "Hello"}
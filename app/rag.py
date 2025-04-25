from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
import config

class HfllmsPipeline:

    def __init__(self, tokens):
        self.openai_model = ChatOpenAI(api_key=config.OAI_API_KEY, model="gpt-3.5-turbo", max_tokens=tokens)
        self.gemini_model = ChatGoogleGenerativeAI(api_key=config.G_API_KEY, model="gemini-1.5-pro", max_output_tokens=tokens)
        self.deepseek_model = ChatDeepSeek(api_key=config.DS_API_KEY, model="deepseek-chat", max_tokens=tokens)

    def __update_token_count(self, token_size):
        self.openai_model.max_tokens = token_size
        self.gemini_model.max_output_tokens = token_size
        self.deepseek_model.max_tokens = token_size

    def generate_results(self, prompt, tokens=50):

        self.__update_token_count(tokens)
        responses = {
            "ChatGPT": self.openai_model.invoke(prompt).content,
            "Gemini": self.gemini_model.invoke(prompt).content,
            "DeepSeek": self.deepseek_model.invoke(prompt).content
        }
        return responses
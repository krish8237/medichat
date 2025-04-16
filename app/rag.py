import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
import huggingface_hub
import torch

class RAGGenerator:
    def __init__(self, model_name='mistralai/Mistral-7B-Instruct-v0.1'):
        huggingface_hub.login(token=config.HG_TKN)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=config.HG_TKN)
        print("Token Invoked")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", use_auth_token=config.HG_TKN)
        print("Model Invoked")
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate_answer(self, query, context):
        prompt = f"""You are a helpful medical assistant. Use the below context to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
        output = self.pipeline(prompt, max_new_tokens=200, do_sample=True)[0]["generated_text"]
        return output.split("Answer:")[-1].strip()
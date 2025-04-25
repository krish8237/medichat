import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np
from medichat_client import MediChatClient
from tqdm import tqdm
from bert_score import score

class Evaluator:

    def __init__(self, chat_client, test_dataset_path):
        self.test_dataset_path = test_dataset_path
        self.test_data = None
        self.client = chat_client
        self.load_test_dataset()

    def load_test_dataset(self):
        data = pd.read_csv(self.test_dataset_path)
        match_pattern = r'Question:\s*(.*?)\s*URL:\s*(https?://[^\s]+)\s*Answer:\s*(.*)'
        self.test_data = data['Answer'].str.extract(match_pattern, expand=True)
        self.test_data.columns = ['Question', 'URL', 'Answer']
        self.test_data['Question'] = self.test_data['Question'].str.replace(r'\(Also called:.*?\)', '',
                                                                        regex=True).str.strip()

    def metrics(self, generated_responses, reference_answers):

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = [
            scorer.score(ref, gen) for ref, gen in zip(reference_answers, generated_responses)
        ]

        try:
            bleu_scores = [
                sentence_bleu([word_tokenize(ref)], word_tokenize(gen))
                for ref, gen in zip(reference_answers, generated_responses)
            ]
        except Exception as e:
            print(f"Error While calculating BLEU scores: {e}")
            bleu_scores = [0.0] * len(generated_responses)

        return {
            "rouge1_precision": float(np.mean([score["rouge1"].precision for score in rouge_scores])),
            "rouge1_recall": float(np.mean([score["rouge1"].recall for score in rouge_scores])),
            "rouge1_f1": float(np.mean([score["rouge1"].fmeasure for score in rouge_scores])),
            "rouge2_f1": float(np.mean([score["rouge2"].fmeasure for score in rouge_scores])),
            "rougeL_f1": float(np.mean([score["rougeL"].fmeasure for score in rouge_scores])),
            "bleu": float(np.mean(bleu_scores))
        }

    def evaluate(self, test_size):
        test_data = self.test_data.sample(n=test_size)
        questions = test_data['Question'].tolist()
        ref_answers = test_data['Answer'].tolist()
        responses = {
            "ChatGPT" : [], "Gemini" : [], "DeepSeek" : []
        }
        metrics = {}
        for i, (query, reference) in enumerate(
                tqdm(zip(questions, ref_answers), total=test_size)
        ):
            token_size = len(reference.split(" "))
            response = self.client.ask(query, None, tokens=token_size)
            print(response)
            responses["ChatGPT"].append(response["results"]["ChatGPT"])
            responses["Gemini"].append(response["results"]["Gemini"])
            responses["DeepSeek"].append(response["results"]["DeepSeek"])

        for model in responses:
            metrics[model] = self.metrics(responses[model], ref_answers)

        return {
            "queries" : questions,
            "responses" : responses,
            "metrics" : metrics
        }

if(__name__ == "__main__"):
    medichat = MediChatClient("../data/processed_data/", "../faiss_data/")
    print("Invoked Medichat ...")
    medichat.setup()
    print("Done with the setup ...")
    test_size = int(input("Enter the test size:"))
    evaluator = Evaluator(medichat, "../data/test_data/test.csv")
    print(evaluator.evaluate(test_size))

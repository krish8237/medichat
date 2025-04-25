import os
from langchain_core.documents import Document

import pandas as pd

class MedicalDatasetLoader:
    def __init__(self, medquad_path):
        self.medquad_path = medquad_path
        self.documents = []

    def __process_documents(self, medquad_data):
        for row in medquad_data.index:
            self.documents.append(Document(
                page_content=medquad_data["Answer"][row],
                metadata={"question" : medquad_data["Question"][row], "url" : medquad_data["Url"][row]}
            ))

    def load_documents(self):
        doc_df = pd.DataFrame(columns=[
            "Question",
            "Answer",
            "Url"
        ])
        for file in os.listdir(self.medquad_path):
            data = pd.read_csv(os.path.join(self.medquad_path, file))
            data = data[["question", "answer", "url"]]
            data.rename(columns={"question": "Question", "answer": "Answer", "url": "Url"}, inplace=True)
            data.fillna("", inplace=True)
            doc_df = pd.concat([doc_df, data], ignore_index=True)
        doc_df.drop_duplicates(inplace=True)
        self.__process_documents(doc_df)
        return self.documents
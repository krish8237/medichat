import pandas as pd

class MedicalDatasetLoader:
    def __init__(self, medquad_path, medline_path):
        self.medquad_path = medquad_path
        self.medline_path = medline_path

    def load_documents(self):
        medquad_df = pd.read_csv(self.medquad_path)
        medline_df = pd.read_csv(self.medline_path)
        combined_df = pd.concat([medquad_df, medline_df], ignore_index=True)
        return combined_df['answer'].dropna().tolist()
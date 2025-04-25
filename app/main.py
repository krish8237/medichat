from api import MediChatAPI
import uvicorn
from medichat_client import MediChatClient

medichat = MediChatClient("../data/processed_data/", "../faiss_data/")
print("Invoked Medichat ...")
medichat.setup()
print("Done with the setup ...")
api = MediChatAPI(medichat)
print("Done with the API setup ...")
app = api.app

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
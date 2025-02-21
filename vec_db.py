from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))

data = pd.read_csv('./wiki.csv')
data = data.dropna(subset=['detail']).reset_index(drop=True)
data['detail'] = data['detail'].str.lower()

vector_store = FAISS.from_texts(data['detail'], embeddings)
vector_store.save_local('./embedding.faiss')

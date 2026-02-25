import os
import json
import config  # still allowed if you want other configs
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

# -------------------- 1. ENV SETUP --------------------

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
chatHistory = []

# -------------------- 2. LOAD SCRAPED JSON --------------------

with open("website_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    print("Website data found!")

texts = []
metadatas = []

for page in data["pages"]:
    texts.append(page["text"])
    metadatas.append({
        "source": page["url"],
        "type": "html"
    })

for pdf in data["pdfs"]:
    texts.append(pdf["text"])
    metadatas.append({
        "source": pdf["url"],
        "type": "pdf"
    })

# -------------------- 3. VECTOR STORE SETUP --------------------

textSplitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

split_texts = []
split_metadatas = []

for text, meta in zip(texts, metadatas):
    chunks = textSplitter.split_text(text)
    split_texts.extend(chunks)
    split_metadatas.extend([meta] * len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_texts(
    texts=split_texts,
    embedding=embeddings,
    metadatas=split_metadatas
)
vector_store.save_local("faiss_store")
import requests
import os
import config # This triggers your os.environ logic
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from config import text
from langchain_chroma import Chroma
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def split_text(text, chunk_size=1000, chunk_overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - chunk_overlap)
    return chunks

chunks = split_text(text)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
)
vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
vector_store.persist()
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
user_query = input("What do you wanna know about Zendaya?")
results = retriever.invoke(user_query)
for i, doc in enumerate(results):
    print(f"\n--- Snippet {i+1} ---")
    print(doc.page_content)



# Use .content for the most reliable streaming output across providers

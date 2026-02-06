import os
import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. ENV SETUP
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
chat_history = []

# 2. VECTOR STORE SETUP (The "Library")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_text(config.text)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 3. INITIALIZE THE MODEL
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
def rewrite_query(chat_history,latest_query):
    history_text = "\n".join(
        [f"User: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}"
         for m in chat_history[-6:]]  # only last few msgs
    )

    prompt = f"""
You are a query rewriting assistant.

Conversation History:
{history_text}

Latest User Question:
{latest_query}

Rewrite the latest question into a clear, standalone search query.
Do NOT answer it. Only rewrite.
"""

    rewritten = model.invoke(prompt).content.strip()
    print(rewritten)
    return rewritten

# 4. THE AGENTIC TOOL
@tool
def logistics_search(query: str) -> str:
    """Search for information about Zendaya's career and personality. 
    Use this tool whenever you need facts to answer a question."""
    
    # Step A: Retrieve
    query = rewrite_query(chat_history,query)
    docs = retriever.invoke(query)
    if not docs:
        return "No information found in the local database."
    
    context = "\n".join([doc.page_content for doc in docs])
    
    # Step B: Evaluate (The "Agentic" Part)
    verification_prompt = (
        f"User Query: {query}\n\n"
        f"Retrieved Context: {context}\n\n"
        "Does the context above contain the answer to the query? "
        "Answer ONLY with 'YES' or 'NO'."
    )
    
    relevance_check = model.invoke(verification_prompt).content.strip().upper()
    
    if "NO" in relevance_check:
        return "The database contains some information, but it is not relevant to this specific question."
    
    return context

# 5. AGENT SETUP
tools = [logistics_search]


agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="You are a helpful assistant. Always use the logistics_search tool to find facts.If the tool says information is missing or irrelevant, inform the user honestly instead of guessing.",
)

# 6. RUN
if __name__ == "__main__":
    try:
        print("🚀 System Online...")
        while True:
            user_query = input()
            if user_query.lower() in ["exit", "quit"]:
                break
            chat_history.append(HumanMessage(content=user_query))
            res = agent.invoke({"messages":chat_history})
            final_answer = res["messages"][-1]
            print("\n✨ ANSWER:", final_answer.text)
            
            chat_history.append(AIMessage(content=final_answer.text))
    except Exception as e:
        print(f"❌ Error: {e}")
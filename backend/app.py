from fastapi import FastAPI
from pydantic import BaseModel
from main_rag import agent, f_rewrite_query, model
from langchain_core.messages import HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware
from api_config import rotate_api_key
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class Query(BaseModel):
    question: str
    history: list = []


@app.get("/")
def home():
    return {"message": "AI Logistics Assistant API is running"}


@app.post("/chat")
def chat(q: Query):

    # 🔥 Rewrite ONLY current query (stateless optimization)
    rewritten_query = f_rewrite_query(q.question)

    # 🤖 Send optimized query to agent with key rotation on quota errors
    last_error = None
    for attempt in range(4):  # max attempts = number of keys
        try:
            res = agent.invoke({
                "messages": [HumanMessage(content=rewritten_query)]
            })
            break
        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["quota", "rate", "429", "resource exhausted"]):
                print(f"⚠️ API key quota hit, rotating... (attempt {attempt + 1})")
                new_key = rotate_api_key()
                os.environ["GOOGLE_API_KEY"] = new_key
                model.google_api_key = new_key
                last_error = e
            else:
                raise e
    else:
        raise last_error

    # 📤 Extract answer safely
    if isinstance(res, dict) and "messages" in res:
        answer = res["messages"][-1].content
    else:
        answer = str(res)

    return {"answer": answer}
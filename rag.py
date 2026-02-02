import requests
from langchain.chat_models import init_chat_model
import config # This triggers your os.environ logic
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model(
    model='gemini-2.5-flash', 
    model_provider='google_genai',
    temperature=0.1
)

# Use .content for the most reliable streaming output across providers
for chunk in model.stream('Hello, what is python?'):
    print(chunk.content, end='', flush=True)
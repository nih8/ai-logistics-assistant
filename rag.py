import requests
from langchain.chat_models import init_chat_model
import config
from langchain.messages import HumanMessage, AIMessage, SystemMessage
model = init_chat_model(
  model = 'gpt-4o-mini',
  temperature = 0.1
)
for chunk in model.stream('Hello, what is python?'):
  print(chunk.text,end='',flush=True)
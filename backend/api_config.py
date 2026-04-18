"""
This file has the code for setting the api key for gemini LLM
Supports up to 4 API keys with automatic rotation on quota/rate-limit errors.
"""
import os
from dotenv import load_dotenv

load_dotenv()

_keys = [
    os.getenv("GOOGLE_API_KEY"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3"),
    os.getenv("GOOGLE_API_KEY_4"),
]
_keys = [k for k in _keys if k]  # filter out missing keys

if not _keys:
    raise ValueError("No GOOGLE_API_KEY found in .env")

_current_index = 0

def get_api_key() -> str:
    return _keys[_current_index]

def rotate_api_key() -> str:
    global _current_index
    _current_index = (_current_index + 1) % len(_keys)
    print(f"🔄 Rotated to API key index {_current_index}")
    return _keys[_current_index]

# Set initial key in environment
GOOGLE_API_KEY = get_api_key()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

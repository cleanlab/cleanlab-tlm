from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

openai_kwargs = {
    "model": "gpt-4.1-mini",
    "tools": [{"type": "web_search_preview"}],
    "input": "What was a positive news story from today?",
}

response = client.responses.create(**openai_kwargs)

print(response)

from src.cleanlab_tlm.utils.responses import TLMResponses

tlm = TLMResponses(
    api_key=os.getenv("CLEANLAB_TLM_API_KEY"), options={"log": ["explanation"]}
)

print(tlm.score(response=response, **openai_kwargs))

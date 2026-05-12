"""
Thin wrapper around the OpenAI-compatible chat endpoint (Ollama or hosted).
"""

from openai import OpenAI
from config import AGENT_MODELS

_clients: dict[str, OpenAI] = {}


def _client(agent: str) -> tuple[OpenAI, str]:
    cfg = AGENT_MODELS[agent]
    base_url = cfg["base_url"]
    if base_url not in _clients:
        _clients[base_url] = OpenAI(base_url=base_url, api_key="ollama") # Ollama uses "ollama" as a dummy key
    return _clients[base_url], cfg["model"]


def chat(agent: str, messages: list[dict], **kwargs) -> str:
    """
    Send a chat request for the given agent and return the response text.
    Extra kwargs (temperature, max_tokens, etc.) are forwarded to the API.
    """
    client, model = _client(agent)
    response = client.chat.completions.create(
        model=model,
        # max_tokens=512,
        messages=messages,
        **kwargs,
    )
    return response.choices[0].message.content
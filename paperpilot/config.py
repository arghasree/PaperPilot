import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Model routing per agent — change model/base_url here to switch backends
AGENT_MODELS: dict[str, dict] = {
    "technical_critic":  {"model": "llama3.3:70b", "base_url": OLLAMA_BASE_URL},
    "writing_critic":    {"model": "llama3.1:8b",  "base_url": OLLAMA_BASE_URL},
    "rewriter":          {"model": "llama3.1:8b",  "base_url": OLLAMA_BASE_URL},
    "citation_grounder": {"model": "llama3.1:8b",  "base_url": OLLAMA_BASE_URL},
    "reviewer":          {"model": "llama3.3:70b", "base_url": OLLAMA_BASE_URL},
}
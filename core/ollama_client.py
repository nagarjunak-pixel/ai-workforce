"""
Ollama Client — Local LLM interface for AI Workforce
Talks to Ollama running at localhost:11434
Supports: chat completion, streaming, model listing
"""
import requests
import json
from typing import Optional

OLLAMA_URL = "http://localhost:11434"

# Model assignments — each agent type gets the right model
MODEL_ASSIGNMENTS = {
    "ceo": "qwen3.5:35b-a3b",        # Big model for planning/reasoning
    "research": "qwen3.5:35b-a3b",    # Deep analysis
    "coding": "qwen2.5-coder:7b",     # Fast coding specialist
    "qa": "qwen3.5:4b",              # Lightweight for test gen
    "docs": "qwen3.5:4b",            # Lightweight for docs
    "deploy": "qwen3.5:4b",          # Lightweight for deployment
    "marketing": "qwen3.5:35b-a3b",  # Creative needs bigger model
    "architect": "qwen3.5:35b-a3b",  # Architecture needs reasoning
}

DEFAULT_MODEL = "qwen3.5:4b"


def list_models() -> list:
    """List all available Ollama models"""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if resp.status_code == 200:
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception as e:
        print(f"⚠️  Ollama not reachable: {e}")
    return []


def chat(model: str, messages: list, temperature: float = 0.7, max_tokens: int = 4096) -> str:
    """
    Send chat messages to Ollama and get response.
    messages: [{"role": "system"/"user"/"assistant", "content": "..."}]
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
        if resp.status_code == 200:
            return resp.json().get("message", {}).get("content", "").strip()
        else:
            return f"ERROR: Ollama returned {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return f"ERROR: {e}"


def get_model_for_role(role: str) -> str:
    """Get the assigned model for an agent role"""
    return MODEL_ASSIGNMENTS.get(role, DEFAULT_MODEL)
"""LLM Client — Universal async client for Ollama with stub fallback."""
import httpx
import asyncio
from typing import Optional


class LLMClient:
    """Async LLM client that talks to Ollama's OpenAI-compatible API.
    
    Falls back to a stub response when Ollama is unreachable —
    so the workforce system NEVER crashes, it just degrades.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._available: Optional[bool] = None
    
    async def call(self, model: str, system_prompt: str, user_prompt: str, 
                   temperature: float = 0.7) -> str:
        """Call Ollama chat completions API. Returns response text or stub."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    return self._stub_response(model, system_prompt, user_prompt, 
                                                f"HTTP {resp.status_code}")
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            self._available = False
            return self._stub_response(model, system_prompt, user_prompt, str(e))
    
    async def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except:
            return False
    
    async def list_models(self) -> list:
        """List available Ollama models."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    return [m["name"] for m in data.get("models", [])]
        except:
            pass
        return []
    
    def _stub_response(self, model: str, system: str, prompt: str, error: str) -> str:
        """Graceful stub — tells user what's wrong instead of crashing."""
        return (
            f"[STUB MODE — Ollama unreachable]\n"
            f"Model: {model}\n"
            f"Error: {error}\n"
            f"System prompt was: {system[:80]}...\n"
            f"Prompt was: {prompt[:80]}...\n\n"
            f"Start Ollama with: ollama serve\n"
            f"Then retry. The agent system degrades but never crashes."
        )
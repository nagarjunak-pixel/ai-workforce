"""Agent — Base class for all AI Workforce agents.

Each agent = role + system prompt + model + tools.
Think of it as an employee with a job description.
"""
import time
from dataclasses import dataclass, field
from typing import Optional
from models.client import LLMClient
from models.router import ModelRouter


@dataclass
class AgentStats:
    """Tracks agent performance — like an employee performance review."""
    name: str
    role: str
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0
    last_prompt: str = ""
    last_response: str = ""
    
    @property
    def success_rate(self) -> float:
        return self.successes / max(self.calls, 1)
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.calls, 1)


class Agent:
    """Base AI Agent — the 'employee' of the AI Workforce.
    
    Each specialized agent (CEO, Researcher, Coder, etc.) extends this class
    with a role-specific system prompt and tool access.
    """
    
    def __init__(self, name: str, role: str, system_prompt: str, 
                 model_type: str = "fast", client: Optional[LLMClient] = None):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.client = client or LLMClient()
        self.router = ModelRouter()
        self.model = self.router.select(model_type)
        self.stats = AgentStats(name=name, role=role)
    
    async def execute(self, task: str, temperature: float = 0.7) -> str:
        """Execute a task — the core agent action.
        
        This is what gets called when the orchestrator assigns work.
        """
        self.stats.calls += 1
        self.stats.last_prompt = task
        
        start = time.time()
        try:
            response = await self.client.call(
                model=self.model,
                system_prompt=self.system_prompt,
                user_prompt=task,
                temperature=temperature,
            )
            elapsed = (time.time() - start) * 1000
            self.stats.successes += 1
            self.stats.total_latency_ms += elapsed
            self.stats.last_response = response
            return response
        except Exception as e:
            self.stats.failures += 1
            self.stats.total_latency_ms += (time.time() - start) * 1000
            return f"[Agent {self.name} error] {str(e) or type(e).__name__}"
    
    def get_stats(self) -> dict:
        return {
            "name": self.stats.name,
            "role": self.stats.role,
            "model": self.model,
            "calls": self.stats.calls,
            "successes": self.stats.successes,
            "failures": self.stats.failures,
            "success_rate": f"{self.stats.success_rate:.1%}",
            "avg_latency_ms": f"{self.stats.avg_latency_ms:.0f}ms",
        }
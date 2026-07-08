"""
Base Agent Class — Every AI employee inherits from this
Implements: system prompt, task execution, handoff to next agent
"""
import json
import time
from datetime import datetime
from typing import Optional
from core.ollama_client import chat, get_model_for_role
from core.memory import save_agent_memory, get_agent_memory


class Agent:
    """Base class for all AI workforce agents"""
    
    def __init__(self, role: str, name: str, system_prompt: str, tools: list = None):
        self.role = role
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.model = get_model_for_role(role)
        self.memory = get_agent_memory(role)
    
    def execute(self, task: str, context: dict = None) -> dict:
        """
        Execute a task. Returns:
        {
            "agent": name,
            "role": role,
            "task": task,
            "result": output text,
            "handoff": next agent role or None,
            "timestamp": ISO datetime,
            "duration_seconds": float
        }
        """
        start = time.time()
        context = context or {}
        
        # Build the full prompt with context from previous agents
        context_str = ""
        if context.get("pipeline_history"):
            context_str = "\n\n--- PREVIOUS AGENT OUTPUTS ---\n"
            for entry in context["pipeline_history"]:
                context_str += f"\n[{entry['agent']}]: {entry['result'][:2000]}\n"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"TASK: {task}\n{context_str}\n\nProduce your output. Be specific and actionable."}
        ]
        
        # Add agent memory if available
        if self.memory:
            messages.insert(1, {"role": "assistant", "content": f"Previous context from my memory: {self.memory}"})
        
        print(f"  🔵 [{self.name}] Working on: {task[:80]}...")
        result = chat(self.model, messages, temperature=0.7)
        
        duration = round(time.time() - start, 2)
        
        # Determine handoff — look for NEXT: keyword in result
        handoff = None
        if "NEXT:" in result:
            for line in result.split("\n"):
                if "NEXT:" in line:
                    handoff_text = line.split("NEXT:")[1].strip().lower()
                    # Map common handoff targets
                    handoff_map = {
                        "research": "research",
                        "research agent": "research",
                        "coder": "coding",
                        "coding": "coding",
                        "coding agent": "coding",
                        "architect": "architect",
                        "qa": "qa",
                        "qa agent": "qa",
                        "testing": "qa",
                        "docs": "docs",
                        "documentation": "docs",
                        "documentation agent": "docs",
                        "deploy": "deploy",
                        "deployment": "deploy",
                        "devops": "deploy",
                        "marketing": "marketing",
                    }
                    for key, val in handoff_map.items():
                        if key in handoff_text:
                            handoff = val
                            break
                    break
        
        # Save to memory
        save_agent_memory(self.role, task, result[:500])
        
        output = {
            "agent": self.name,
            "role": self.role,
            "task": task,
            "result": result,
            "handoff": handoff,
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
        }
        
        print(f"  ✅ [{self.name}] Done in {duration}s → Handoff: {handoff or 'END'}")
        return output
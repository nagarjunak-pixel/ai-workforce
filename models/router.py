"""Model Router — Routes tasks to the right local model based on task type.

Architecture decision: Use the RIGHT model for the job, not the biggest.
- Coding tasks → qwen2.5-coder:7b (specialized for code, fast 4.7GB)
- Quick reasoning → qwen3.5:4b (fast, 3.4GB, good for planning)
- Heavy reasoning → qwen3.5:35b-a3b (MoE, 23GB, best quality, slower)

This is the 'CEO' decision: which employee gets which brain.
"""
from typing import Literal


class ModelRouter:
    """Routes tasks to appropriate Ollama models."""
    
    # Model assignments — verified via `ollama list`
    MODELS = {
        "coder": "qwen2.5-coder:7b",        # Code generation, refactoring
        "fast": "qwen3.5:4b",                # Quick tasks, planning, research
        "heavy": "qwen3.5:35b-a3b",          # Deep reasoning, architecture
    }
    
    def __init__(self):
        self._fallback = "qwen3.5:4b"
    
    def select(self, task_type: str) -> str:
        """Select model based on task type.
        
        Args:
            task_type: One of 'coder', 'fast', 'heavy', or a custom key
        Returns:
            Model name string for Ollama
        """
        return self.MODELS.get(task_type, self._fallback)
    
    def select_by_content(self, prompt: str) -> str:
        """Auto-select model based on prompt content heuristics."""
        prompt_lower = prompt.lower()
        
        # Code-related → coder model
        code_keywords = ["code", "function", "class", "def ", "import ", 
                         "api", "endpoint", "bug", "refactor", "test"]
        if any(kw in prompt_lower for kw in code_keywords):
            return self.MODELS["coder"]
        
        # Architecture/complex reasoning → heavy model
        heavy_keywords = ["architecture", "design", "plan", "strategy",
                          "analyze", "evaluate", "comprehensive"]
        if any(kw in prompt_lower for kw in heavy_keywords):
            return self.MODELS["heavy"]
        
        # Default → fast model
        return self.MODELS["fast"]
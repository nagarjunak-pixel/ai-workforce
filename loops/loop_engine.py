"""Loop Engine — Self-improvement cycle.

The Loop is what makes the workforce LEARN from each attempt.
Instead of one-shot generation, it runs multiple cycles:
  1. Generate (agent produces output)
  2. Score (evaluate quality)
  3. Check convergence (is it good enough?)
  4. Adapt (adjust parameters if stuck)
  5. Repeat

This is the 'self-improving' in 'self-improving AI workforce'.
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable, List


@dataclass
class CycleResult:
    """Result of one loop cycle."""
    cycle: int
    response: str
    score: float
    duration_ms: float
    meta: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "score": round(self.score, 3),
            "duration_ms": round(self.duration_ms, 0),
            "response_preview": self.response[:120] + "..." if len(self.response) > 120 else self.response,
            "meta": self.meta,
        }


@dataclass
class LoopStats:
    """Tracks loop engine performance across cycles."""
    iterations: List[CycleResult] = field(default_factory=list)
    converged: bool = False
    best_cycle: int = 0
    best_score: float = 0.0
    
    def summary(self) -> dict:
        return {
            "converged": self.converged,
            "total_cycles": len(self.iterations),
            "best_cycle": self.best_cycle,
            "best_score": round(self.best_score, 3),
            "iterations": [r.to_dict() for r in self.iterations],
        }


class Scorer:
    """Heuristic scorer — evaluates response quality.
    
    Scores on a 0-1 scale based on:
    - Length (longer = more complete, to a point)
    - Code blocks (presence of working code)
    - Examples (real usage examples)
    - Structure (headers, lists, formatting)
    
    This is deliberately simple — no LLM needed for scoring.
    """
    
    @staticmethod
    def score(response: str) -> float:
        score = 0.3  # baseline
        
        # Length bonus
        if len(response) > 100:
            score += 0.2
        if len(response) > 500:
            score += 0.15
        
        # Code blocks
        if "```" in response:
            score += 0.15
        
        # Examples
        if "example" in response.lower() or "Example" in response:
            score += 0.1
        
        # Structure (headers/lists)
        if "#" in response or "- " in response or "1." in response:
            score += 0.1
        
        # Penalty for stub/error responses
        if "[STUB" in response or "[HARNESS]" in response or "error" in response.lower()[:20]:
            score = 0.1
        
        return min(score, 1.0)


class MetaLearner:
    """Adjusts generation parameters based on loop history.
    
    If the agent is stuck (score not improving), bump temperature
    to encourage exploration. If improving, keep stable.
    """
    
    @staticmethod
    def adapt(history: List[CycleResult], current_temp: float) -> float:
        if len(history) < 2:
            return current_temp
        
        recent_scores = [r.score for r in history[-3:]]
        if len(recent_scores) >= 2:
            delta = recent_scores[-1] - recent_scores[-2]
            if abs(delta) < 0.05:  # plateau
                return min(current_temp + 0.1, 1.0)  # explore more
            elif delta > 0.05:  # improving
                return current_temp  # keep going
        
        return current_temp


class LoopEngine:
    """Self-improvement loop — runs generate→score→adapt until convergence.
    
    Args:
        generate_fn: Async function that produces a response
        max_cycles: Maximum iterations (default 3)
        convergence_threshold: Score above which we stop (default 0.7)
        patience: Cycles without improvement before meta-adaptation (default 2)
    """
    
    def __init__(self, generate_fn: Callable[[], Awaitable[str]],
                 max_cycles: int = 3, convergence_threshold: float = 0.7,
                 patience: int = 2, initial_temp: float = 0.7):
        self.generate_fn = generate_fn
        self.max_cycles = max_cycles
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.current_temp = initial_temp
        self.scorer = Scorer()
        self.meta_learner = MetaLearner()
    
    async def run(self) -> LoopStats:
        """Execute the self-improvement loop."""
        stats = LoopStats()
        stall_count = 0
        
        for cycle in range(1, self.max_cycles + 1):
            start = time.time()
            response = await self.generate_fn()
            elapsed = (time.time() - start) * 1000
            
            score = self.scorer.score(response)
            result = CycleResult(
                cycle=cycle, response=response, score=score,
                duration_ms=elapsed,
                meta={"temperature": round(self.current_temp, 2)}
            )
            stats.iterations.append(result)
            
            # Track best
            if score > stats.best_score:
                stats.best_score = score
                stats.best_cycle = cycle
                stall_count = 0
            else:
                stall_count += 1
            
            # Convergence check
            if score >= self.convergence_threshold:
                stats.converged = True
                break
            
            # Adapt if stuck
            if stall_count >= self.patience:
                self.current_temp = self.meta_learner.adapt(
                    stats.iterations, self.current_temp
                )
        
        return stats
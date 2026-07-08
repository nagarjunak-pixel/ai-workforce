"""Orchestrator — The pipeline that chains agents together.

This is the 'workflow' — how work flows through the AI company:
  Goal → CEO → Research → Architect → Code → QA → Docs → Deploy

Each agent's output feeds into the next agent's input.
The orchestrator manages the handoffs and tracks the full pipeline state.
"""
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agents.workforce import create_workforce
from harness.harness import Harness
from loops.loop_engine import LoopEngine
from memory.memory_system import MemorySystem
from models.client import LLMClient


@dataclass
class PipelineStep:
    """One step in the workforce pipeline."""
    agent_name: str
    agent_role: str
    task: str
    response: str = ""
    score: float = 0
    duration_ms: float = 0
    status: str = "pending"  # pending → running → completed → failed
    
    def to_dict(self) -> dict:
        return {
            "agent": self.agent_name,
            "role": self.agent_role,
            "task": self.task[:100] + "..." if len(self.task) > 100 else self.task,
            "response_preview": self.response[:200] + "..." if len(self.response) > 200 else self.response,
            "score": round(self.score, 3),
            "duration_ms": round(self.duration_ms, 0),
            "status": self.status,
        }


@dataclass
class PipelineResult:
    """Full pipeline execution result."""
    goal: str
    steps: List[PipelineStep] = field(default_factory=list)
    total_duration_ms: float = 0
    converged: bool = False
    
    def summary(self) -> dict:
        return {
            "goal": self.goal,
            "total_steps": len(self.steps),
            "completed": sum(1 for s in self.steps if s.status == "completed"),
            "failed": sum(1 for s in self.steps if s.status == "failed"),
            "total_duration_ms": round(self.total_duration_ms, 0),
            "steps": [s.to_dict() for s in self.steps],
        }


class Orchestrator:
    """Manages the AI Workforce pipeline.
    
    Usage:
        orch = Orchestrator()
        result = await orch.run("Build a CRM application")
        print(json.dumps(result.summary(), indent=2))
    """
    
    def __init__(self, use_loops: bool = True, use_harness: bool = True,
                 max_loop_cycles: int = 2, memory: Optional[MemorySystem] = None):
        self.client = LLMClient()
        self.workforce = create_workforce(self.client)
        self.harness = Harness(max_retries=2, timeout=180.0)
        self.use_loops = use_loops
        self.use_harness = use_harness
        self.max_loop_cycles = max_loop_cycles
        self.memory = memory or MemorySystem()
    
    async def run(self, goal: str) -> PipelineResult:
        """Execute the full workforce pipeline for a goal.
        
        Pipeline: CEO → Research → Architect → Code → QA → Docs
        """
        result = PipelineResult(goal=goal)
        start_time = time.time()
        
        # Save goal to project state
        self.memory.set_project_state("current_goal", goal)
        
        # Step 1: CEO breaks down the goal
        ceo_step = await self._execute_agent(
            "ceo", f"Break down this goal into tasks: {goal}"
        )
        result.steps.append(ceo_step)
        
        if ceo_step.status != "completed":
            return result
        
        # Step 2: Research (based on CEO's plan)
        research_task = f"Based on this project plan, research the best approaches:\n\n{ceo_step.response}"
        research_step = await self._execute_agent("researcher", research_task)
        result.steps.append(research_step)
        
        # Step 3: Architect (based on research)
        arch_task = f"Design the architecture based on this research:\n\n{research_step.response}"
        arch_step = await self._execute_agent("architect", arch_task)
        result.steps.append(arch_step)
        
        # Step 4: Code (based on architecture)
        code_task = f"Write the code based on this architecture:\n\n{arch_step.response}"
        code_step = await self._execute_agent("coder", code_task)
        result.steps.append(code_step)
        
        # Step 5: QA (review the code)
        qa_task = f"Review this code for bugs and write test cases:\n\n{code_step.response}"
        qa_step = await self._execute_agent("qa", qa_task)
        result.steps.append(qa_step)
        
        # Step 6: Documentation
        docs_task = f"Write a README.md for this project:\n\nArchitecture:\n{arch_step.response}\n\nCode:\n{code_step.response}"
        docs_step = await self._execute_agent("docs", docs_task)
        result.steps.append(docs_step)
        
        result.total_duration_ms = (time.time() - start_time) * 1000
        result.converged = all(s.status == "completed" for s in result.steps)
        
        # Save result to memory
        self.memory.set_project_state("last_result", json.dumps(result.summary()))
        
        return result
    
    async def run_marketing(self, goal: str, project_result: PipelineResult) -> PipelineStep:
        """Optional: Run marketing agent after project is built."""
        arch_response = ""
        code_response = ""
        for s in project_result.steps:
            if s.agent_name == "Architect":
                arch_response = s.response
            if s.agent_name == "Coder":
                code_response = s.response
        
        task = f"Create marketing content for this project:\n\nGoal: {goal}\nArchitecture: {arch_response}\nCode: {code_response}"
        return await self._execute_agent("marketing", task)
    
    async def _execute_agent(self, agent_key: str, task: str) -> PipelineStep:
        """Execute a single agent with harness + optional loop."""
        agent = self.workforce[agent_key]
        step = PipelineStep(
            agent_name=agent.name, agent_role=agent.role,
            task=task, status="running"
        )
        
        start = time.time()
        
        if self.use_loops:
            # Run with self-improvement loop
            loop = LoopEngine(
                generate_fn=lambda: agent.execute(task),
                max_cycles=self.max_loop_cycles,
                convergence_threshold=0.7,
            )
            loop_stats = await loop.run()
            step.response = loop_stats.iterations[loop_stats.best_cycle - 1].response if loop_stats.iterations else ""
            step.score = loop_stats.best_score
        elif self.use_harness:
            # Run with harness only (no loop)
            step.response = await self.harness.run(lambda: agent.execute(task))
            step.score = 0.5  # default if no loop scoring
        else:
            # Raw execution
            step.response = await agent.execute(task)
            step.score = 0.5
        
        step.duration_ms = (time.time() - start) * 1000
        step.status = "completed" if step.response and not step.response.startswith("[") else "failed"
        
        # Record to memory
        self.memory.record_interaction(
            agent_name=agent.name, agent_role=agent.role,
            prompt=task, response=step.response, score=step.score
        )
        
        return step
    
    def get_workforce_stats(self) -> dict:
        """Get stats for all agents — the company dashboard."""
        return {
            "agents": [a.get_stats() for a in self.workforce.values()],
            "harness": self.harness.get_stats(),
            "memory": self.memory.stats(),
        }
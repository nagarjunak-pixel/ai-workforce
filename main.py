"""AI Workforce — FastAPI Server

The web dashboard for the AI Workforce system.
Endpoints:
  GET  /            → Dashboard UI
  GET  /health       → System health + Ollama status
  POST /run          → Run the full workforce pipeline
  GET  /agents       → List all agents + their stats
  GET  /memory        → Recent memory interactions
  GET  /memory/stats  → Memory statistics
  POST /agent/{name}  → Run a single agent
  GET  /state         → Project state
"""
import asyncio
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from models.client import LLMClient
from agents.workforce import create_workforce
from agents.orchestrator import Orchestrator
from memory.memory_system import MemorySystem

app = FastAPI(title="AI Workforce", description="Your personal AI company on a laptop")

# Global instances
client = LLMClient()
workforce = create_workforce(client)
memory = MemorySystem()


# ─── Models ───

class RunRequest(BaseModel):
    goal: str
    use_loops: bool = True
    use_harness: bool = True
    max_loop_cycles: int = 2

class AgentRequest(BaseModel):
    task: str
    use_harness: bool = True


# ─── Endpoints ───

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the interactive dashboard."""
    html_path = os.path.join(os.path.dirname(__file__), "ui", "dashboard.html")
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Dashboard not found</h1><p>ui/dashboard.html missing</p>")


@app.get("/health")
async def health():
    """Check system health — is Ollama running? What models are available?"""
    available = await client.is_available()
    models = await client.list_models()
    return {
        "ollama_running": available,
        "available_models": models,
        "workforce_agents": len(workforce),
        "memory": memory.stats(),
    }


@app.post("/run")
async def run_pipeline(req: RunRequest):
    """Run the full workforce pipeline — CEO → Research → Architect → Code → QA → Docs."""
    orch = Orchestrator(
        use_loops=req.use_loops,
        use_harness=req.use_harness,
        max_loop_cycles=req.max_loop_cycles,
        memory=memory,
    )
    result = await orch.run(req.goal)
    return result.summary()


@app.get("/agents")
async def get_agents():
    """List all workforce agents and their stats."""
    stats = []
    for key, agent in workforce.items():
        s = agent.get_stats()
        s["key"] = key
        s["system_prompt_preview"] = agent.system_prompt[:100] + "..."
        stats.append(s)
    return {"agents": stats, "harness": {"timeout": 180.0, "max_retries": 2}}


@app.post("/agent/{agent_key}")
async def run_single_agent(agent_key: str, req: AgentRequest):
    """Run a single agent on a task."""
    if agent_key not in workforce:
        return JSONResponse(
            status_code=404,
            content={"error": f"Agent '{agent_key}' not found. Available: {list(workforce.keys())}"}
        )
    
    agent = workforce[agent_key]
    
    if req.use_harness:
        from harness.harness import Harness
        h = Harness(max_retries=2, timeout=180.0)
        response = await h.run(lambda: agent.execute(req.task))
        harness_stats = h.get_stats()
    else:
        response = await agent.execute(req.task)
        harness_stats = None
    
    memory.record_interaction(
        agent_name=agent.name, agent_role=agent.role,
        prompt=req.task, response=response
    )
    
    return {
        "agent": agent.name,
        "role": agent.role,
        "model": agent.model,
        "response": response,
        "harness_stats": harness_stats,
        "agent_stats": agent.get_stats(),
    }


@app.get("/memory")
async def get_memory(limit: int = Query(10, ge=1, le=100)):
    """Get recent memory interactions."""
    return {"interactions": memory.get_history(limit)}


@app.get("/memory/stats")
async def get_memory_stats():
    """Memory statistics."""
    return memory.stats()


@app.get("/state")
async def get_project_state():
    """Get all project state — the workforce's working memory."""
    return memory.get_all_project_state()


if __name__ == "__main__":
    import uvicorn
    # Kill any existing process on 8181
    os.system("kill $(lsof -t -i :8181) 2>/dev/null")
    uvicorn.run(app, host="0.0.0.0", port=8181)
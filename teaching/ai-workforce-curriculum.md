# AI Workforce Architecture
## Building Your Personal AI Company on a Laptop

**Curriculum Document — Surya Agentic Academy**
**Trainer:** Nagarjuna Reddy K
**Audience:** Intermediate Python developers learning Agentic Engineering
**Prerequisites:** Python basics, Ollama installed, models pulled

---

## Module 0 — Before We Begin: Mental Model

**Think of a company.** A startup has a CEO, researchers, architects, developers, QA engineers, documentation writers, DevOps engineers, and marketing folks. Each person specializes. Each person hands off to the next. The CEO doesn't write code. The coder doesn't deploy. The deployer doesn't write the README.

**Now imagine all those "people" are LLM agents running on your laptop.** That's an AI Workforce.

> **The one-sentence definition:** An AI Workforce is a collection of specialized LLM agents — each with a role, a system prompt, a toolset, and a memory — that collaborate through a pipeline to complete end-to-end projects without human intervention at every step.

This curriculum teaches you how to build one.

---

## Module 1 — Concept Overview

### 1.1 What is an AI Workforce?

An AI Workforce is **not** one big chatbot that does everything. It is a **team** of small, focused agents. Each agent:

- Has a **job title** (Researcher, Coder, QA Engineer…)
- Has a **system prompt** that defines its personality and constraints
- Has **tools** it's allowed to use (and only those)
- Has **memory** so it remembers what happened in previous sessions
- Hands **structured output** to the next agent in the pipeline

### 1.2 Why 10–30 Specialized Agents > 1 General Agent

| One General Agent | A Workforce of Specialized Agents |
|---|---|
| One huge prompt trying to cover everything | Each agent has a small, focused prompt |
| Context window gets bloated, quality drops | Each agent stays focused, quality stays high |
| No tool isolation — it can do anything (dangerous) | Least-privilege tool access per agent (safe) |
| If it fails, the whole thing fails | Circuit breaker isolates failures per agent |
| Can't scale — you hit token limits | Scales horizontally — add agents as needed |
| No memory separation — everything in one blob | Per-agent memory — each remembers its own domain |

**Analogy:** Would you rather hire one person who "kind of does everything okay," or a team of 10 specialists? The team wins every time. Same for agents.

### 1.3 The AI Employees Analogy

| Human Role | AI Agent Equivalent |
|---|---|
| CEO | CEO Agent — breaks down goals into tasks |
| Research Analyst | Research Agent — searches web, reads docs |
| Software Architect | Architect Agent — designs systems, picks tech |
| Software Developer | Coding Agent — writes and fixes code |
| QA Engineer | QA Agent — runs tests, finds edge cases |
| Technical Writer | Documentation Agent — writes READMEs, API docs |
| DevOps Engineer | Deployment Agent — Docker, CI/CD, hosting |
| Marketing Manager | Marketing Agent — landing pages, blog posts, SEO |

Each "hire" is just a Python class with a system prompt and a toolset. You "onboard" an agent by writing its prompt. You "promote" an agent by giving it more tools or memory.

---

## Module 2 — The Agent → Harness → Loops → Memory Framework

This is the **core architecture** of every agent in the workforce. Four layers, stacked.

```
┌─────────────────────────────────────┐
│           MEMORY LAYER              │  ← SQLite: remembers across sessions
├─────────────────────────────────────┤
│             LOOPS LAYER             │  ← Self-improvement cycle
├─────────────────────────────────────┤
│            HARNESS LAYER            │  ← Safety: retry, timeout, validation
├─────────────────────────────────────┤
│            AGENT LAYER              │  ← LLM + system prompt + tools
└─────────────────────────────────────┘
```

### 2.1 Layer 1 — Agent (The Brain)

An agent is an **LLM wrapper** with:
- A **system prompt** that defines its role and rules
- A **model** (from Ollama)
- A set of **tools** it can call
- An **input** (task from previous agent) and an **output** (structured handoff)

```python
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

def call_agent(model: str, system_prompt: str, user_message: str) -> str:
    """The simplest agent: a model + a system prompt."""
    response = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    })
    return response.json()["message"]["content"]

# --- Example: Research Agent ---
RESEARCH_SYSTEM = """You are a Research Agent.
Your job: search the web, read documents, and produce a concise summary.
Rules:
- Always cite sources.
- Maximum 500 words.
- Output as bullet points.
- If you don't know, say 'NEED_MORE_INFO'.
"""

summary = call_agent("qwen3.5:4b", RESEARCH_SYSTEM, "Research fastapi best practices for 2025")
print(summary)
```

**Key insight:** The system prompt *is* the agent's identity. Change the prompt, change the agent. Same model, different "employee."

### 2.2 Layer 2 — Harness (The Safety Net)

Agents fail. LLMs hallucinate. Network calls time out. The harness wraps every agent call with **protective logic**:

- **Retry** — if the LLM call fails, retry N times with backoff
- **Timeout** — don't wait forever; kill the call after X seconds
- **Circuit breaker** — if an agent fails 5 times in a row, stop calling it (it's "down")
- **Validation** — check the output matches expected format before passing it on

```python
import time
from functools import wraps

class AgentHarness:
    """Safety wrapper around any agent call."""

    def __init__(self, max_retries=3, timeout_seconds=60, failure_threshold=5):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.failure_threshold = failure_threshold
        self.failure_count = 0
        self.circuit_open = False

    def run(self, agent_fn, *args, **kwargs):
        # Circuit breaker check
        if self.circuit_open:
            raise RuntimeError("Circuit breaker OPEN — agent is down. Fix and reset.")

        for attempt in range(self.max_retries):
            try:
                start = time.time()
                result = agent_fn(*args, **kwargs)
                elapsed = time.time() - start

                if elapsed > self.timeout_seconds:
                    raise TimeoutError(f"Agent timed out after {elapsed:.1f}s")

                # Reset failure count on success
                self.failure_count = 0
                return result

            except Exception as e:
                self.failure_count += 1
                print(f"  ⚠️  Attempt {attempt+1} failed: {e}")
                if self.failure_count >= self.failure_threshold:
                    self.circuit_open = True
                    raise RuntimeError(
                        f"Circuit breaker tripped after {self.failure_count} failures"
                    )
                time.sleep(2 ** attempt)  # exponential backoff

        raise RuntimeError(f"Agent failed after {self.max_retries} retries")
```

**Analogy:** The harness is like a seatbelt. You don't plan to crash, but when you do, you're glad it's there.

### 2.3 Layer 3 — Loops (Self-Improvement)

A single LLM call is a **one-shot**. But real work is iterative. The loops layer adds a **self-improvement cycle**:

```
generate → score → adapt → repeat (until score is good enough)
```

1. **Generate** — agent produces output
2. **Score** — a judge (can be the same model or another) evaluates the output on a rubric
3. **Adapt** — if score is below threshold, feed the critique back and regenerate
4. **Repeat** — until the score passes or max iterations reached

```python
def self_improve_loop(agent_fn, judge_fn, task: str, max_iters: int = 3, threshold: float = 0.8):
    """Generate → Score → Adapt → Repeat"""
    current_output = agent_fn(task)
    
    for i in range(max_iters):
        score, critique = judge_fn(task, current_output)
        print(f"  Loop {i+1}: score={score:.2f}, critique={critique[:80]}...")

        if score >= threshold:
            print("  ✅ Quality threshold met — returning output.")
            return current_output

        # Adapt: feed critique back into the agent
        improved_task = f"""Original task: {task}

Previous output:
{current_output}

Critique (score {score:.2f}/1.0):
{critique}

Please improve your output based on this critique."""
        current_output = agent_fn(improved_task)

    print("  ⚠️  Max iterations reached — returning best effort.")
    return current_output
```

**Analogy:** It's like a writer drafting → editor marking up → writer revising → editor approving. The loop *is* that back-and-forth, automated.

### 2.4 Layer 4 — Memory (Persistence)

Without memory, every session starts from zero. The agent doesn't remember what it did yesterday. Memory fixes this with **SQLite** — a lightweight, file-based database that's built into Python.

```python
import sqlite3
import json
from datetime import datetime

class AgentMemory:
    """SQLite-backed memory for an agent."""

    def __init__(self, db_path: str, agent_name: str):
        self.conn = sqlite3.connect(db_path)
        self.agent_name = agent_name
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT,
                key TEXT,
                value TEXT,
                created_at TEXT
            )
        """)
        self.conn.commit()

    def remember(self, key: str, value: str):
        self.conn.execute(
            "INSERT INTO memories (agent_name, key, value, created_at) VALUES (?,?,?,?)",
            (self.agent_name, key, value, datetime.now().isoformat())
        )
        self.conn.commit()

    def recall(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM memories WHERE agent_name=? AND key=? ORDER BY id DESC LIMIT 1",
            (self.agent_name, key)
        ).fetchone()
        return row[0] if row else None

    def recall_all(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT key, value, created_at FROM memories WHERE agent_name=? ORDER BY id DESC LIMIT 50",
            (self.agent_name,)
        ).fetchall()
        return [{"key": r[0], "value": r[1], "when": r[2]} for r in rows]
```

**Analogy:** Memory is the agent's notebook. When it learns something (a bug it fixed, a preference the user has), it writes it down. Next session, it reads its notes first.

---

## Module 3 — Agent Roles

Each role below is a **Python class** with a system prompt, a model, tools, and memory. Here's the full roster.

### 3.1 CEO Agent — The Planner

| Field | Value |
|---|---|
| **Model** | `qwen3.5:35b-a3b` (the powerful MoE — needs the brain) |
| **Tools** | calendar, email, task manager |
| **Memory** | project goals, decisions, milestones |
| **Output** | structured task list (JSON) |

**Responsibilities:**
- Takes a high-level goal ("build a weather app") and decomposes it into sub-tasks
- Assigns each sub-task to the right agent
- Tracks progress and re-plans when things fail
- Makes go/no-go decisions

**System prompt (excerpt):**
```
You are the CEO Agent. You decompose goals into actionable sub-tasks.
Output format: JSON array of {task, assigned_agent, priority}.
Agents available: research, architect, coding, qa, docs, deploy, marketing.
Always include a 'definition_of_done' for each task.
```

**Example output:**
```json
[
  {"task": "Research weather APIs with free tiers", "assigned_agent": "research", "priority": "high", "definition_of_done": "List of 3+ APIs with pricing and rate limits"},
  {"task": "Design FastAPI backend architecture", "assigned_agent": "architect", "priority": "high", "definition_of_done": "Architecture diagram + tech stack list"},
  {"task": "Implement /weather endpoint", "assigned_agent": "coding", "priority": "high", "definition_of_done": "Working endpoint returning JSON weather data"},
  {"task": "Write tests for /weather endpoint", "assigned_agent": "qa", "priority": "medium", "definition_of_done": "5+ test cases including edge cases"},
  {"task": "Write README and API docs", "assigned_agent": "docs", "priority": "low", "definition_of_done": "README.md with setup + usage + endpoint docs"},
  {"task": "Create Dockerfile and deploy", "assigned_agent": "deploy", "priority": "low", "definition_of_done": "Docker image builds and runs locally"}
]
```

### 3.2 Research Agent — The Investigator

| Field | Value |
|---|---|
| **Model** | `qwen3.5:4b` (fast — needs to be quick) |
| **Tools** | browser, web search, PDF reader |
| **Memory** | past research findings, source credibility ratings |
| **Output** | bullet-point summary with citations |

**Responsibilities:**
- Searches the web for information
- Reads PDFs and long documents, extracts key points
- Summarizes findings into structured format for the next agent
- Flags conflicting information

**System prompt (excerpt):**
```
You are the Research Agent. You find and summarize information.
Rules:
- Search the web, read documents, extract key points.
- Cite every claim with a source URL.
- If sources conflict, flag the conflict.
- Output: bullet points, max 500 words.
- End with 'RESEARCH_COMPLETE' or 'NEED_MORE_INFO'.
```

### 3.3 Architect Agent — The Designer

| Field | Value |
|---|---|
| **Model** | `qwen3.5:35b-a3b` (needs reasoning power) |
| **Tools** | filesystem (read-only), diagram tools |
| **Memory** | past architecture decisions, tech stack preferences |
| **Output** | architecture spec (tech stack + component list + data flow) |

**Responsibilities:**
- Takes research findings and designs the system
- Selects tech stack (framework, database, deployment target)
- Defines component interfaces (what calls what)
- Produces a spec the Coding Agent can implement from

**System prompt (excerpt):**
```
You are the Architect Agent. You design systems from requirements.
Output format:
  tech_stack: {frontend, backend, database, deployment}
  components: [{name, responsibility, interfaces}]
  data_flow: step-by-step description
  file_structure: tree of files to create
Keep it practical. Prefer simplicity over cleverness.
```

### 3.4 Coding Agent — The Builder

| Field | Value |
|---|---|
| **Model** | `qwen2.5-coder:7b` (specialized for code) |
| **Tools** | filesystem (read+write), GitHub, terminal |
| **Memory** | codebase patterns, known bugs and fixes |
| **Output** | code files written to disk |

**Responsibilities:**
- Implements features from the architecture spec
- Fixes bugs reported by QA Agent
- Refactors code when asked
- Commits to GitHub

**System prompt (excerpt):**
```
You are the Coding Agent. You write clean, tested Python code.
Rules:
- Follow PEP 8.
- Include type hints.
- Write docstrings for all functions.
- If you change code, run existing tests to check nothing broke.
- Output the full file content, not just diffs.
- End with 'CODE_COMPLETE' or 'NEED_CLARIFICATION'.
```

### 3.5 QA Agent — The Tester

| Field | Value |
|---|---|
| **Model** | `qwen3.5:4b` (fast iteration) |
| **Tools** | Python (pytest), Playwright, browser |
| **Memory** | past test results, known flaky tests |
| **Output** | test results (pass/fail + failure details) |

**Responsibilities:**
- Writes test cases from the spec
- Runs tests and reports results
- Finds edge cases (empty input, huge input, bad input)
- Sends bugs back to Coding Agent with reproduction steps

**System prompt (excerpt):**
```
You are the QA Agent. You test code and find bugs.
Rules:
- Write pytest test cases for every function.
- Include edge cases: empty, None, very large, negative, unicode.
- Run tests with: python -m pytest tests/ -v
- Report: pass/fail count, failure details, reproduction steps.
- If all pass, output 'QA_PASS'. If any fail, output 'QA_FAIL' + details.
```

### 3.6 Documentation Agent — The Writer

| Field | Value |
|---|---|
| **Model** | `qwen3.5:4b` |
| **Tools** | filesystem (read+write), diagram tools |
| **Memory** | documentation style guide, past docs |
| **Output** | README.md, API docs, architecture diagrams |

**Responsibilities:**
- Writes README.md with setup, usage, examples
- Generates API documentation from code
- Creates architecture diagrams
- Keeps docs in sync with code changes

**System prompt (excerpt):**
```
You are the Documentation Agent. You write clear, practical docs.
Rules:
- README must include: what it is, how to install, how to run, API reference.
- Use code examples that actually work (test them mentally).
- Keep it concise. No walls of text. Use headers and lists.
- Generate a Mermaid diagram for the architecture.
```

### 3.7 Deployment Agent — The Shipper

| Field | Value |
|---|---|
| **Model** | `qwen3.5:4b` |
| **Tools** | Docker, CI/CD (GitHub Actions), cloud CLI |
| **Memory** | deployment configs, past deployment issues |
| **Output** | live URL + Docker image + CI/CD pipeline |

**Responsibilities:**
- Creates Dockerfile and docker-compose.yml
- Sets up GitHub Actions for CI/CD
- Deploys to staging, then production
- Monitors deployment health

**System prompt (excerpt):**
```
You are the Deployment Agent. You ship code to production.
Rules:
- Create a minimal Dockerfile (multi-stage build preferred).
- Set up GitHub Actions: lint → test → build → deploy.
- Deploy to staging first. Verify health. Then promote to production.
- If deployment fails, rollback and report the error.
```

### 3.8 Marketing Agent — The Storyteller

| Field | Value |
|---|---|
| **Model** | `qwen3.5:4b` |
| **Tools** | filesystem (write), browser (read-only for SEO research) |
| **Memory** | brand voice, past campaign performance |
| **Output** | landing page HTML, blog posts, SEO keywords |

**Responsibilities:**
- Writes landing page copy from the product description
- Creates blog posts about features
- Researches SEO keywords and optimizes content
- Generates social media posts

**System prompt (excerpt):**
```
You are the Marketing Agent. You write copy that converts.
Rules:
- Landing page: hero headline, 3 benefit bullets, CTA.
- Blog posts: 800-1200 words, SEO-optimized, includes code examples.
- Always include meta description and keywords.
- Tone: confident, practical, no hype. Like a senior engineer explaining.
```

---

## Module 4 — The Pipeline

The workforce isn't a free-for-all. Agents work in a **pipeline** — each agent's output is the next agent's input.

```
User Goal
    │
    ▼
┌─────────┐     ┌──────────┐     ┌────────────┐     ┌─────────┐
│   CEO   │────▶│ Research  │────▶│ Architect  │────▶│  Coding │
└─────────┘     └──────────┘     └────────────┘     └─────────┘
                   task list       architecture         code
                                                     │
┌─────────┐     ┌──────┐     ┌───────────┐           │
│  Deploy │◀────│ Docs │◀────│    QA     │◀──────────┘
└─────────┘     └──────┘     └───────────┘
   live URL     README       test results
```

### 4.1 Pipeline Implementation (Simplified)

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class Task:
    goal: str
    context: dict[str, Any] = None  # carries outputs between agents

def run_pipeline(goal: str) -> dict:
    """The full AI Workforce pipeline."""
    task = Task(goal=goal, context={})

    # Each agent reads from task.context, writes its output back
    # CEO: decompose goal into sub-tasks
    task.context["task_list"] = ceo_agent.run(task)
    
    # Research: gather info for the task list
    task.context["research"] = research_agent.run(task)
    
    # Architect: design the system from research
    task.context["architecture"] = architect_agent.run(task)
    
    # Coding: implement the architecture
    task.context["code"] = coding_agent.run(task)
    
    # QA: test the code
    task.context["test_results"] = qa_agent.run(task)
    
    # If QA fails, loop back to Coding
    if task.context["test_results"]["status"] == "FAIL":
        task.context["bug_report"] = task.context["test_results"]["failures"]
        task.context["code"] = coding_agent.run(task)  # fix bugs
        task.context["test_results"] = qa_agent.run(task)  # re-test
    
    # Docs: document the working code
    task.context["docs"] = docs_agent.run(task)
    
    # Deploy: ship it
    task.context["deployment"] = deploy_agent.run(task)
    
    return task.context
```

### 4.2 Key Pipeline Principles

1. **Structured handoffs** — each agent outputs JSON or structured text, not free-form chat
2. **Context carries forward** — the `context` dict accumulates outputs; each agent reads what it needs
3. **Feedback loops** — QA can send bugs back to Coding (not strictly linear)
4. **Failure isolation** — if one agent fails, the harness catches it; the pipeline can retry or skip
5. **Human checkpoints** — you can insert a human approval step between any two agents

---

## Module 5 — Tool Access Per Agent (Principle of Least Privilege)

**Critical security principle:** Each agent gets only the tools it needs. Nothing more.

| Agent | Allowed Tools | Forbidden Tools |
|---|---|---|
| CEO | calendar, email, task manager | filesystem, terminal, browser |
| Research | browser (read-only), web search, PDF reader | filesystem (write), terminal |
| Architect | filesystem (read-only), diagram tools | terminal, browser, email |
| Coding | filesystem (read+write), GitHub, terminal | browser, email, calendar |
| QA | Python, Playwright, browser (test mode) | email, calendar, GitHub (write) |
| Docs | filesystem (read+write), diagram tools | terminal, browser, email |
| Deploy | Docker, CI/CD CLI, cloud CLI | browser, email, task manager |
| Marketing | filesystem (write), browser (read-only) | terminal, GitHub, Docker |

**Why?** If the Research Agent can't write to the filesystem, it can't accidentally delete your code. If the Marketing Agent can't run terminal commands, it can't `rm -rf` your project. **Least privilege is not optional — it's the difference between a safe workforce and a dangerous one.**

### 5.1 Implementing Tool Access

```python
from abc import ABC, abstractmethod

class Tool(ABC):
    @abstractmethod
    def execute(self, *args, **kwargs) -> str:
        pass

class WebSearchTool(Tool):
    def execute(self, query: str) -> str:
        # implementation using requests to a search API
        ...

class FilesystemReadTool(Tool):
    def execute(self, path: str) -> str:
        with open(path, "r") as f:
            return f.read()

class FilesystemWriteTool(Tool):
    def execute(self, path: str, content: str) -> str:
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {path}"

class TerminalTool(Tool):
    def execute(self, command: str) -> str:
        import subprocess
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout + result.stderr

# --- Agent with restricted tools ---
class Agent:
    def __init__(self, name: str, model: str, system_prompt: str, tools: list[Tool]):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.tools = {t.__class__.__name__: t for t in tools}

    def run(self, task: str) -> str:
        # The agent can only call tools in self.tools
        # It literally cannot access tools not in this list
        ...

# Research Agent: can search the web, but CANNOT write files
research_agent = Agent(
    name="research",
    model="qwen3.5:4b",
    system_prompt=RESEARCH_SYSTEM,
    tools=[WebSearchTool()]  # ← no FilesystemWriteTool, no TerminalTool
)

# Coding Agent: can write files and run terminal, but CANNOT search the web
coding_agent = Agent(
    name="coding",
    model="qwen2.5-coder:7b",
    system_prompt=CODING_SYSTEM,
    tools=[FilesystemReadTool(), FilesystemWriteTool(), TerminalTool()]
)
```

---

## Module 6 — MCP Servers as Departments

### 6.1 What is MCP?

**MCP (Model Context Protocol)** is a standard way to connect LLMs to external tools. Think of it as **USB for AI agents** — a universal plug that any agent can use to connect to any tool.

### 6.2 The Department Analogy

In a real company:
- **IT Department** manages computers and software
- **HR Department** manages people
- **Finance Department** manages money

In your AI Workforce:
- **MCP Server: filesystem** → gives agents access to files (like the IT department)
- **MCP Server: GitHub** → gives agents access to repos (like the development department)
- **MCP Server: browser** → gives agents access to the web (like the research department)
- **MCP Server: database** → gives agents access to data (like the finance department)

```
┌──────────────────────────────────────────────────┐
│                  AI WORKFORCE                     │
│                                                  │
│  CEO Agent ──── Research Agent ──── Coding Agent │
│      │                │                   │       │
│      │    ┌───────────┼───────────┐       │       │
│      │    │           │           │       │       │
│  ┌───▼────▼───┐ ┌────▼────┐ ┌─────▼─────┐ ┌────▼───┐
│  │ MCP:       │ │ MCP:    │ │ MCP:       │ │ MCP:   │
│  │ calendar   │ │ browser │ │ filesystem │ │ GitHub │
│  │ email      │ │ search  │ │ terminal   │ │ Docker │
│  └────────────┘ └─────────┘ └────────────┘ └────────┘
└──────────────────────────────────────────────────┘
```

### 6.3 How MCP Works in Practice

```python
# An MCP server exposes tools via a standard protocol
# Any agent that speaks MCP can use any MCP server

# Example: Filesystem MCP Server
{
  "server_name": "filesystem",
  "tools": [
    {
      "name": "read_file",
      "description": "Read a file from the filesystem",
      "parameters": {"path": "string"}
    },
    {
      "name": "write_file",
      "description": "Write content to a file",
      "parameters": {"path": "string", "content": "string"}
    }
  ]
}

# The agent calls: mcp_call("filesystem", "read_file", {"path": "README.md"})
# The MCP server executes it and returns the result.
```

### 6.4 Why MCP Matters for the Workforce

| Without MCP | With MCP |
|---|---|
| Each agent needs custom tool integrations | One protocol, any tool |
| Adding a new tool = rewriting agent code | Add an MCP server, agents auto-discover it |
| Tool access is ad-hoc, hard to audit | MCP servers log all tool calls |
| Hard to share tools between agents | Any agent can connect to any MCP server |

**The big idea:** MCP lets you grow your workforce without rewriting agents. Want to add a "Slack Agent"? Stand up a Slack MCP server. Every agent can now use Slack. No code changes to existing agents.

---

## Module 7 — Practical Build Plan

Don't build 20 agents on day one. **Start with 3, grow to 20.**

### Phase 1 — Foundation (Week 1)

**Goal:** One working agent with a system prompt.

- Install Ollama, pull `qwen3.5:4b`
- Write a single agent: `call_agent(model, system_prompt, user_message)`
- Test it: give it a task, get a response
- **Deliverable:** One Python script that calls Ollama and gets a response

### Phase 2 — Harness (Week 2)

**Goal:** Add safety to your agent.

- Add retry logic (3 attempts, exponential backoff)
- Add timeout (60 seconds max)
- Add output validation (does the response match expected format?)
- **Deliverable:** `AgentHarness` class wrapping your agent

### Phase 3 — Loops (Week 3)

**Goal:** Agent improves its own output.

- Add a judge function that scores output 0.0–1.0
- Add the generate→score→adapt→repeat cycle
- Set threshold (0.8) and max iterations (3)
- **Deliverable:** `self_improve_loop()` function

### Phase 4 — Memory (Week 4)

**Goal:** Agent remembers across sessions.

- Add SQLite database
- `remember(key, value)` and `recall(key)` methods
- Load past memories into the system prompt at session start
- **Deliverable:** `AgentMemory` class

### Phase 5 — Second Agent (Week 5)

**Goal:** Two agents working together.

- Build Research Agent (search + summarize)
- Build Coding Agent (takes research output, writes code)
- Chain them: Research output → Coding input
- **Deliverable:** 2-agent pipeline

### Phase 6 — Full Pipeline (Week 6-7)

**Goal:** The complete workforce pipeline.

- Add CEO, Architect, QA, Docs, Deploy agents
- Implement the pipeline: Goal → CEO → Research → Architect → Code → QA → Docs → Deploy
- Add feedback loop: QA → Coding when tests fail
- **Deliverable:** Full 7-agent pipeline

### Phase 7 — MCP + Scale (Week 8+)

**Goal:** Connect external tools and scale up.

- Set up MCP servers (filesystem, browser, GitHub)
- Add Marketing Agent
- Add monitoring and logging
- Grow to 15-20 agents as needed
- **Deliverable:** Production-grade AI Workforce

### Growth Summary

```
Phase 1: 1 agent (just calls LLM)
Phase 2: 1 agent + harness (safe)
Phase 3: 1 agent + harness + loops (self-improving)
Phase 4: 1 agent + harness + loops + memory (persistent)
Phase 5: 2 agents (Researcher + Coder pipeline)
Phase 6: 7 agents (full pipeline)
Phase 7: 15-20 agents (MCP-connected, production-grade)
```

---

## Module 8 — Comparison with Real Systems

### 8.1 What Companies Are Doing

**ClickUp** launched "ClickUp Brain" — an AI that connects tasks, docs, people, and company knowledge. It's essentially a workforce of specialized AI agents behind a single interface.

**Devin (Cognition)** is an "AI software engineer" — but under the hood, it's multiple agents (planner, coder, debugger, reviewer) working in a pipeline.

**Cursor / Windsurf** use agent pipelines internally: one agent understands the codebase, another writes code, another reviews it.

**OpenAI's Swarm framework** is literally this concept — lightweight agents with handoffs.

### 8.2 How Our Approach Compares

| Aspect | Enterprise (ClickUp, Devin) | Our AI Workforce |
|---|---|---|
| Number of agents | 50-200+ | 10-30 |
| Infrastructure | Kubernetes, cloud, GPU clusters | Your laptop + Ollama |
| Cost | $10K-$100K/month in API calls | $0 (local models) |
| Latency | Cloud round-trips | Local, sub-second |
| Privacy | Data goes to cloud | Data stays on your machine |
| Customization | Closed, vendor-controlled | Full control, open source |
| Setup time | Months, team of engineers | Days, solo builder |

### 8.3 The Key Insight

**The architecture is the same.** ClickUp has 200 agents; you have 20. But the pattern — specialized agents, pipeline, harness, memory — is identical. You're building a **microcosm** of what big companies build, but on a laptop, for free, with total control.

> **You don't need a billion-dollar company to have an AI workforce. You need a laptop, Ollama, and the right framework.**

---

## Module 9 — Hands-On Exercise: Build Your First 2-Agent Pipeline

**Goal:** Build a Researcher + Coder pipeline that researches a topic and writes code based on the research.

### 9.1 Setup

```bash
# Install Ollama (if not already installed)
# macOS:
brew install ollama
# Start Ollama:
ollama serve

# Pull the models we'll use:
ollama pull qwen3.5:4b          # research (fast)
ollama pull qwen2.5-coder:7b    # coding (specialized)
```

### 9.2 The Full Pipeline Code

```python
"""
AI Workforce — First 2-Agent Pipeline
Surya Agentic Academy | Trainer: Nagarjuna Reddy K

Research Agent (qwen3.5:4b) → Coding Agent (qwen2.5-coder:7b)
"""

import requests
import time
import json

OLLAMA_URL = "http://localhost:11434/api/chat"

# ─────────────────────────────────────────────
# AGENT LAYER: Call Ollama with a system prompt
# ─────────────────────────────────────────────

def call_agent(model: str, system_prompt: str, user_message: str) -> str:
    """Call an Ollama model with a system prompt."""
    response = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": 0.3},  # low temp = more consistent
    }, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"]


# ─────────────────────────────────────────────
# HARNESS LAYER: Retry + timeout + validation
# ─────────────────────────────────────────────

def run_with_harness(agent_fn, *args, max_retries=3, validate_fn=None, **kwargs):
    """Run an agent with retry and validation."""
    for attempt in range(max_retries):
        try:
            result = agent_fn(*args, **kwargs)
            if validate_fn and not validate_fn(result):
                print(f"  ⚠️  Validation failed on attempt {attempt+1}")
                continue
            return result
        except Exception as e:
            print(f"  ⚠️  Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Agent failed after {max_retries} retries")


# ─────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────

RESEARCH_SYSTEM = """You are the Research Agent in an AI Workforce.
Your job: research a topic and produce a technical brief for a coder.

Rules:
- Focus on practical, actionable information.
- Include specific libraries, APIs, and code patterns.
- Output as structured markdown with sections.
- Keep it under 400 words.
- End with 'RESEARCH_COMPLETE'.
"""

CODER_SYSTEM = """You are the Coding Agent in an AI Workforce.
Your job: write Python code based on a research brief.

Rules:
- Write clean, runnable Python 3.11+ code.
- Include type hints and docstrings.
- Include a main() function that demonstrates usage.
- The code must be complete and runnable as-is.
- End with 'CODE_COMPLETE'.
"""


# ─────────────────────────────────────────────
# VALIDATORS
# ─────────────────────────────────────────────

def validate_research(output: str) -> bool:
    """Check that research output is valid."""
    return "RESEARCH_COMPLETE" in output and len(output) > 100

def validate_code(output: str) -> bool:
    """Check that code output looks like code."""
    return "CODE_COMPLETE" in output and ("def " in output or "class " in output)


# ─────────────────────────────────────────────
# THE PIPELINE
# ─────────────────────────────────────────────

def research_then_code(topic: str) -> dict:
    """2-agent pipeline: Research → Code"""
    print(f"\n{'='*60}")
    print(f"PIPELINE START: {topic}")
    print(f"{'='*60}")

    # Step 1: Research Agent
    print("\n📋 Step 1: Research Agent (qwen3.5:4b)")
    research_brief = run_with_harness(
        call_agent,
        model="qwen3.5:4b",
        system_prompt=RESEARCH_SYSTEM,
        user_message=f"Research this topic for a coder: {topic}",
        validate_fn=validate_research,
    )
    print(f"   Research output: {len(research_brief)} chars")

    # Step 2: Coding Agent (takes research as input)
    print("\n💻 Step 2: Coding Agent (qwen2.5-coder:7b)")
    code = run_with_harness(
        call_agent,
        model="qwen2.5-coder:7b",
        system_prompt=CODER_SYSTEM,
        user_message=f"""Based on this research, write a Python script:

{research_brief}

Write complete, runnable Python code.""",
        validate_fn=validate_code,
    )
    print(f"   Code output: {len(code)} chars")

    return {
        "topic": topic,
        "research": research_brief,
        "code": code,
    }


# ─────────────────────────────────────────────
# RUN IT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    result = research_then_code("How to build a CLI tool that fetches weather data from a free API")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print("\n--- RESEARCH BRIEF ---")
    print(result["research"][:500] + "...")
    print("\n--- CODE (first 500 chars) ---")
    print(result["code"][:500] + "...")
```

### 9.3 Run It

```bash
python pipeline.py
```

**Expected output:**
```
============================================================
PIPELINE START: How to build a CLI tool that fetches weather data from a free API
============================================================

📋 Step 1: Research Agent (qwen3.5:4b)
   Research output: 1200 chars

💻 Step 2: Coding Agent (qwen2.5-coder:7b)
   Code output: 2400 chars

============================================================
PIPELINE COMPLETE
============================================================
```

---

## Module 10 — Key Takeaways

1. **An AI Workforce is a team, not a tool.** Think employees, not functions. Each agent has a role, tools, and memory.

2. **Specialization beats generalization.** 10 focused agents with small prompts outperform 1 agent with a giant prompt. Less context, better quality.

3. **The four layers are non-negotiable:**
   - **Agent** — the LLM + system prompt (the brain)
   - **Harness** — retry, timeout, validation (the safety net)
   - **Loops** — generate → score → adapt → repeat (self-improvement)
   - **Memory** — SQLite persistence (continuity across sessions)

4. **Least privilege is a security requirement.** Each agent gets only the tools it needs. Research Agent can't delete files. Marketing Agent can't run `rm -rf`.

5. **The pipeline is the product.** Individual agents are components; the pipeline — how they chain together — is what delivers end-to-end results.

6. **Start small, grow incrementally.** Phase 1: one agent. Phase 5: two agents. Phase 7: twenty. Don't skip phases.

7. **MCP is the integration layer.** It lets you add tools without rewriting agents. Think of MCP servers as departments in your AI company.

8. **Local is viable.** Ollama + qwen models on a laptop can run a real workforce. You don't need cloud APIs or GPU clusters for most tasks.

9. **Memory makes agents feel alive.** Without it, every session is amnesia. With it, your Research Agent remembers what it found last week. Your Coder remembers the bugs it fixed.

10. **The architecture scales.** What you build on a laptop with 20 agents is the same pattern ClickUp uses with 200 agents. The framework is the same; the scale is different.

---

## Appendix A — Ollama Model Quick Reference

| Model | Size | Best For | Speed |
|---|---|---|---|
| `qwen3.5:4b` | ~2.5 GB | Research, QA, Docs, Deploy, Marketing | Fast |
| `qwen3.5:35b-a3b` | ~20 GB | CEO, Architect (needs reasoning) | Moderate |
| `qwen2.5-coder:7b` | ~4.5 GB | Coding Agent (specialized for code) | Fast |

```bash
# Pull all models for the course:
ollama pull qwen3.5:4b
ollama pull qwen3.5:35b-a3b
ollama pull qwen2.5-coder:7b

# List installed models:
ollama list

# Check Ollama is running:
curl http://localhost:11434/api/tags
```

## Appendix B — Recommended Project Structure

```
ai-workforce/
├── agents/
│   ├── __init__.py
│   ├── base.py          # Agent class, call_agent()
│   ├── ceo.py
│   ├── research.py
│   ├── architect.py
│   ├── coding.py
│   ├── qa.py
│   ├── docs.py
│   ├── deploy.py
│   └── marketing.py
├── core/
│   ├── __init__.py
│   ├── harness.py       # AgentHarness
│   ├── loops.py         # self_improve_loop()
│   └── memory.py        # AgentMemory
├── tools/
│   ├── __init__.py
│   ├── web_search.py
│   ├── filesystem.py
│   ├── terminal.py
│   └── browser.py
├── pipeline.py          # The full pipeline
├── config.py            # Model assignments, prompts
├── memory.db            # SQLite database
└── main.py              # Entry point
```

---

**End of Curriculum — Surya Agentic Academy**
**Trainer:** Nagarjuna Reddy K
**Framework:** Agent → Harness → Loops → Memory
*"Build your AI company. One agent at a time."*
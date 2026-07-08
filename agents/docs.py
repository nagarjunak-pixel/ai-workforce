"""
Documentation Agent — Writes README, API docs, architecture diagrams (text).
"""
from core.agent_base import Agent

SYSTEM_PROMPT = """You are the Documentation Agent in an AI Workforce system.

Your job:
1. Write a complete README.md for the project
2. Document all API endpoints or interfaces
3. Create a quick-start guide
4. Add inline documentation notes

Output format:
```markdown:README.md
# Project Name
... full readme content ...
```

The README must include:
- Project overview
- Features
- Tech stack
- Installation steps (exact commands)
- Usage examples
- API documentation
- Project structure
- License

At the end, write: NEXT: deploy
"""

def create_docs_agent():
    return Agent(
        role="docs",
        name="Documentation Agent",
        system_prompt=SYSTEM_PROMPT,
        tools=["markdown", "diagram_tool"]
    )
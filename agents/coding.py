"""
Coding Agent — Writes actual code files based on architecture spec.
"""
from core.agent_base import Agent
import os
import re
from pathlib import Path

SYSTEM_PROMPT = """You are the Coding Agent in an AI Workforce system.

Your job:
1. Take the architecture spec and write actual, working code
2. Output code in clearly delimited blocks with file paths
3. Each file must be complete and runnable — no placeholders, no TODOs
4. Follow the exact tech stack chosen by the Architect Agent

Output format for each file:
```python:filename.py
# actual code here
```

Or for other languages:
```html:templates/index.html
<!-- actual code -->
```

Rules:
- Every file must have its path as the language tag (e.g. ```python:app/main.py)
- No placeholders like "add your logic here" — write real code
- Include error handling and logging
- Keep it simple — MVP quality, not over-engineered
- Include a requirements.txt or package.json if needed

At the end, write: NEXT: qa
"""

def create_coding_agent():
    return Agent(
        role="coding",
        name="Coding Agent",
        system_prompt=SYSTEM_PROMPT,
        tools=["file_system", "github", "terminal"]
    )


def extract_code_files(result: str) -> dict:
    """Extract code blocks from agent result. Returns {filepath: content}"""
    files = {}
    # Pattern: ```lang:filepath ... ```
    pattern = r'```[\w]*:?([^\n]+)\n(.*?)```'
    matches = re.findall(pattern, result, re.DOTALL)
    for filepath, code in matches:
        filepath = filepath.strip().lstrip(':')
        # Skip if it's a language name without a path
        if filepath in ('python', 'html', 'css', 'js', 'javascript', 'json', 'bash', 'sh', 'text'):
            continue
        if '/' in filepath or '.' in filepath:
            files[filepath] = code.strip()
    return files


def write_code_files(files: dict, output_dir: str = None) -> list:
    """Write extracted code files to disk. Returns list of written paths."""
    written = []
    base = Path(output_dir) if output_dir else Path.cwd()
    
    for filepath, content in files.items():
        full_path = base / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        written.append(str(full_path))
        print(f"  📝 Wrote {filepath} ({len(content)} chars)")
    
    return written
# AI Workforce Architecture — Student Exercises
## Progressive Build: From Zero Agent to 2-Agent Pipeline

**Surya Agentic Academy | Trainer: Nagarjuna Reddy K**
**Prerequisites:** Python 3.11+, Ollama installed and running, `qwen3.5:4b` and `qwen2.5-coder:7b` pulled

```
Exercise 1: Single Agent        →  You can call an LLM with a system prompt
Exercise 2: Add a Harness       →  Your agent is safe (retry + timeout)
Exercise 3: Add a Loop          →  Your agent improves its own output
Exercise 4: Add Memory          →  Your agent remembers across sessions
Exercise 5: Chain Two Agents    →  Researcher → Coder pipeline
```

**How to use these exercises:**
- Do them in order. Each builds on the previous.
- Create a new Python file for each exercise.
- Run every code block. Read every output. Don't skip.
- If something fails, debug it — that's part of learning.

---

## Exercise 1 — Build a Single Agent with a System Prompt

### Objective
Call Ollama with a role-specific system prompt and get a structured response. Understand that the system prompt *is* the agent's identity.

### Setup

```bash
# Make sure Ollama is running
ollama serve

# Pull the model we'll use
ollama pull qwen3.5:4b

# Verify it works
curl http://localhost:11434/api/tags
```

### Steps

**Step 1:** Create `exercise1_agent.py`:

```python
"""
Exercise 1: Build a Single Agent
Surya Agentic Academy | Student Exercises

Goal: Call Ollama with a system prompt and get a structured response.
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

def call_agent(model: str, system_prompt: str, user_message: str) -> str:
    """Call an Ollama model with a system prompt."""
    response = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"]


# ─── Define the agent's identity ───
SUMMARIZER_SYSTEM = """You are a Summarizer Agent.
Your job: read any text and produce a 3-bullet-point summary.

Rules:
- Exactly 3 bullet points. No more, no less.
- Each bullet must be under 20 words.
- Start each bullet with '•'.
- End your response with 'SUMMARY_COMPLETE'.
"""

# ─── Run the agent ───
if __name__ == "__main__":
    text_to_summarize = """
    Python is a high-level, general-purpose programming language. Its design philosophy
    emphasizes code readability with the use of significant indentation. Python is
    dynamically typed and garbage-collected. It supports multiple programming paradigms,
    including structured, object-oriented, and functional programming. Python was created
    by Guido van Rossum and released in 1991. It has become one of the most popular
    programming languages in the world, used for web development, data science,
    automation, artificial intelligence, and more.
    """

    print("Calling Summarizer Agent (qwen3.5:4b)...")
    print("-" * 50)

    result = call_agent(
        model="qwen3.5:4b",
        system_prompt=SUMMARIZER_SYSTEM,
        user_message=f"Summarize this text:\n\n{text_to_summarize}"
    )

    print(result)
    print("-" * 50)
```

**Step 2:** Run it:

```bash
python exercise1_agent.py
```

### Expected Output

```
Calling Summarizer Agent (qwen3.5:4b)...
--------------------------------------------------
• Python is a high-level, general-purpose language emphasizing code readability
• It supports multiple paradigms: structured, object-oriented, and functional programming
• Created by Guido van Rossum in 1991, now used widely in web, data science, and AI
SUMMARY_COMPLETE
--------------------------------------------------
```

(Your exact wording may differ — that's fine. The structure should match: 3 bullets + `SUMMARY_COMPLETE`.)

### Verification

| Check | How to verify |
|---|---|
| Agent responds | Output is not empty |
| 3 bullet points | Count `•` markers — should be exactly 3 |
| Each bullet < 20 words | Count words per bullet |
| Ends with `SUMMARY_COMPLETE` | Last line contains this string |

**Bonus:** Change the system prompt to make the agent a "Translator Agent" that translates text to Telugu or Hindi. Observe how the same model becomes a completely different "employee" just by changing the prompt.

---

## Exercise 2 — Add a Harness (Retry + Timeout)

### Objective
Wrap your agent call with safety: retry on failure, timeout if the model hangs, and validate the output format.

### Steps

**Step 1:** Create `exercise2_harness.py`:

```python
"""
Exercise 2: Add a Harness (Retry + Timeout + Validation)
Surya Agentic Academy | Student Exercises

Goal: Make your agent safe. If it fails, retry. If it hangs, timeout.
      If the output is wrong, catch it.
"""

import requests
import time

OLLAMA_URL = "http://localhost:11434/api/chat"

def call_agent(model: str, system_prompt: str, user_message: str, timeout: int = 120) -> str:
    """Call Ollama with a timeout."""
    response = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }, timeout=timeout)
    response.raise_for_status()
    return response.json()["message"]["content"]


# ─── The Harness ───
class AgentHarness:
    """Safety wrapper: retry, timeout, validation."""

    def __init__(self, max_retries=3, failure_threshold=5):
        self.max_retries = max_retries
        self.failure_threshold = failure_threshold
        self.failure_count = 0
        self.circuit_open = False

    def run(self, agent_fn, *args, validate_fn=None, **kwargs):
        """Run an agent with retry, timeout, and validation."""
        if self.circuit_open:
            raise RuntimeError("Circuit breaker OPEN — too many failures. Reset manually.")

        for attempt in range(self.max_retries):
            try:
                print(f"  [Harness] Attempt {attempt + 1}/{self.max_retries}...")
                result = agent_fn(*args, **kwargs)

                # Validate output
                if validate_fn:
                    if not validate_fn(result):
                        print(f"  [Harness] Validation failed — retrying...")
                        continue
                    print(f"  [Harness] ✅ Validation passed")

                # Success — reset failure count
                self.failure_count = 0
                return result

            except Exception as e:
                self.failure_count += 1
                print(f"  [Harness] ❌ Error: {e}")
                if self.failure_count >= self.failure_threshold:
                    self.circuit_open = True
                    raise RuntimeError(
                        f"Circuit breaker tripped after {self.failure_count} failures"
                    )
                wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                print(f"  [Harness] Waiting {wait}s before retry...")
                time.sleep(wait)

        raise RuntimeError(f"Agent failed after {self.max_retries} retries")


# ─── Validator ───
def validate_summary(output: str) -> bool:
    """Check that output has 3 bullet points and ends with SUMMARY_COMPLETE."""
    bullet_count = output.count("•")
    has_marker = "SUMMARY_COMPLETE" in output
    if bullet_count != 3:
        print(f"  [Validator] Expected 3 bullets, got {bullet_count}")
        return False
    if not has_marker:
        print(f"  [Validator] Missing SUMMARY_COMPLETE marker")
        return False
    return True


# ─── Run it ───
if __name__ == "__main__":
    SUMMARIZER_SYSTEM = """You are a Summarizer Agent.
Rules:
- Exactly 3 bullet points starting with '•'.
- Each bullet under 20 words.
- End with 'SUMMARY_COMPLETE'.
"""

    text = """
    FastAPI is a modern, fast web framework for building APIs with Python 3.7+
    based on standard Python type hints. It is built on top of Starlette for the
    web parts and Pydantic for the data parts. FastAPI is designed to be easy to
    use and learn, fast to code, and ready for production. It automatically
    generates OpenAPI documentation and provides automatic data validation.
    """

    harness = AgentHarness(max_retries=3)

    print("Running Summarizer Agent through harness...")
    print("=" * 50)

    result = harness.run(
        call_agent,
        model="qwen3.5:4b",
        system_prompt=SUMMARIZER_SYSTEM,
        user_message=f"Summarize:\n\n{text}",
        timeout=60,
        validate_fn=validate_summary,
    )

    print("=" * 50)
    print("FINAL OUTPUT:")
    print(result)

    # ─── Test the failure path ───
    print("\n" + "=" * 50)
    print("TESTING FAILURE: Calling a non-existent model...")

    bad_harness = AgentHarness(max_retries=2, failure_threshold=3)
    try:
        bad_harness.run(
            call_agent,
            model="nonexistent-model:0b",
            system_prompt="test",
            user_message="test",
            timeout=5,
            validate_fn=None,
        )
    except Exception as e:
        print(f"Expected failure: {e}")
```

**Step 2:** Run it:

```bash
python exercise2_harness.py
```

### Expected Output

```
Running Summarizer Agent through harness...
==================================================
  [Harness] Attempt 1/3...
  [Harness] ✅ Validation passed
==================================================
FINAL OUTPUT:
• FastAPI is a modern, fast web framework for Python 3.7+ using type hints
• Built on Starlette (web) and Pydantic (data), with auto OpenAPI docs
• Designed to be easy to use, fast to code, and production-ready
SUMMARY_COMPLETE

==================================================
TESTING FAILURE: Calling a non-existent model...
  [Harness] Attempt 1/2...
  [Harness] ❌ Error: ...
  [Harness] Waiting 1s before retry...
  [Harness] Attempt 2/2...
  [Harness] ❌ Error: ...
  [Harness] Waiting 2s before retry...
Expected failure: Agent failed after 2 retries
```

### Verification

| Check | How to verify |
|---|---|
| Successful call validates | Output passes `validate_summary()` |
| Retry works | Watch the console for "Attempt 2/3" on failures |
| Timeout works | Set `timeout=1` — should fail fast |
| Circuit breaker works | Call with bad model 5 times — circuit opens |
| Exponential backoff | Console shows wait times: 1s, 2s, 4s |

**Bonus:** Add a `circuit_reset()` method that resets `circuit_open = False` and `failure_count = 0`. Call it after the circuit trips to "recover" the agent.

---

## Exercise 3 — Add a Loop (Self-Improvement Cycle)

### Objective
Make your agent evaluate and improve its own output. Implement the generate → score → adapt → repeat cycle.

### Steps

**Step 1:** Create `exercise3_loop.py`:

```python
"""
Exercise 3: Add a Loop (Self-Improvement Cycle)
Surya Agentic Academy | Student Exercises

Goal: Agent generates output → Judge scores it → If low, feed critique back → Repeat
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

def call_agent(model: str, system_prompt: str, user_message: str) -> str:
    """Call Ollama."""
    response = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"]


# ─── The Generator Agent ───
GENERATOR_SYSTEM = """You are a Code Reviewer Agent.
Your job: review Python code and list issues.

Output format:
ISSUES_FOUND: <count>
1. <issue description>
2. <issue description>
...

QUALITY_SCORE: <0.0 to 1.0>

End with 'REVIEW_COMPLETE'.
"""

# ─── The Judge Agent ───
JUDGE_SYSTEM = """You are a Judge Agent.
Your job: evaluate a code review for quality.

Score the review from 0.0 to 1.0 based on:
- Did it find real issues? (0.4)
- Is each issue specific and actionable? (0.3)
- Is the format correct? (0.3)

Output EXACTLY:
SCORE: <float>
CRITIQUE: <one sentence describing what to improve>
END
"""


def parse_judge_output(output: str) -> tuple[float, str]:
    """Extract score and critique from judge output."""
    lines = output.strip().split("\n")
    score = 0.0
    critique = ""
    for line in lines:
        if line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
            except ValueError:
                pass
        elif line.startswith("CRITIQUE:"):
            critique = line.replace("CRITIQUE:", "").strip()
    return score, critique


def self_improve_loop(
    task: str,
    generator_system: str,
    judge_system: str,
    model: str = "qwen3.5:4b",
    max_iters: int = 3,
    threshold: float = 0.8,
) -> str:
    """Generate → Score → Adapt → Repeat"""
    print(f"\n🔄 Starting self-improvement loop (threshold={threshold})")

    current_output = call_agent(model, generator_system, task)

    for i in range(max_iters):
        # Score: ask the judge to evaluate
        judge_input = f"""Evaluate this code review:

{current_output}

Score it from 0.0 to 1.0."""
        judge_result = call_agent(model, judge_system, judge_input)
        score, critique = parse_judge_output(judge_result)

        print(f"  Loop {i + 1}: score={score:.2f} | critique={critique}")

        if score >= threshold:
            print(f"  ✅ Threshold met! Returning output.")
            return current_output

        # Adapt: feed the critique back into the generator
        improved_task = f"""Original task: {task}

Your previous review:
{current_output}

The judge scored it {score:.2f}/1.0 with this critique:
{critique}

Please improve your review. Address the critique."""

        current_output = call_agent(model, generator_system, improved_task)

    print(f"  ⚠️ Max iterations reached. Returning best effort.")
    return current_output


# ─── Run it ───
if __name__ == "__main__":
    # The code we want reviewed
    code_to_review = """
def get_weather(city):
    url = f"http://api.weather.com/{city}"
    r = requests.get(url)
    data = r.json()
    return data["temperature"]

def main():
    weather = get_weather("Hyderabad")
    print(f"Temperature: {weather}")
"""

    task = f"Review this Python code:\n\n```python\n{code_to_review}\n```"

    print("=" * 60)
    print("SELF-IMPROVEMENT LOOP: Code Review Agent")
    print("=" * 60)

    final_review = self_improve_loop(
        task=task,
        generator_system=GENERATOR_SYSTEM,
        judge_system=JUDGE_SYSTEM,
        model="qwen3.5:4b",
        max_iters=3,
        threshold=0.8,
    )

    print("\n" + "=" * 60)
    print("FINAL REVIEW (after self-improvement):")
    print("=" * 60)
    print(final_review)
```

**Step 2:** Run it:

```bash
python exercise3_loop.py
```

### Expected Output

```
============================================================
SELF-IMPROVEMENT LOOP: Code Review Agent
============================================================

🔄 Starting self-improvement loop (threshold=0.8)
  Loop 1: score=0.55 | critique=Issues are vague, not specific enough
  Loop 2: score=0.75 | critique=Better, but missing format compliance
  Loop 3: score=0.90 | critique=Good review, specific and well-formatted
  ✅ Threshold met! Returning output.

============================================================
FINAL REVIEW (after self-improvement):
============================================================
ISSUES_FOUND: 4
1. No error handling — requests.get() can throw ConnectionError
2. No timeout on requests.get() — can hang indefinitely
3. r.json() assumes valid JSON — no validation
4. data["temperature"] uses string key — KeyError if key missing
QUALITY_SCORE: 0.9
REVIEW_COMPLETE
```

(Your scores and wording will differ. What matters: the score should increase over loops, and the final review should be more specific than the first.)

### Verification

| Check | How to verify |
|---|---|
| Loop runs multiple times | Console shows "Loop 1", "Loop 2", etc. |
| Score is parsed | Console shows `score=X.XX` (a float) |
| Score generally increases | Compare Loop 1 score to Loop 3 score |
| Threshold check works | Loop exits when score >= 0.8 |
| Max iterations works | Set `max_iters=1` — should exit after 1 loop |
| Critique is used | The improved output should address the critique |

**Bonus:** Try setting `threshold=0.95` — the loop should run all 3 iterations and return "best effort." Observe how a high threshold causes more iterations.

---

## Exercise 4 — Add Memory (SQLite Persistence)

### Objective
Give your agent a SQLite-backed memory so it remembers information across sessions.

### Steps

**Step 1:** Create `exercise4_memory.py`:

```python
"""
Exercise 4: Add Memory (SQLite Persistence)
Surya Agentic Academy | Student Exercises

Goal: Agent stores and retrieves information from SQLite across runs.
"""

import requests
import sqlite3
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/chat"

def call_agent(model: str, system_prompt: str, user_message: str) -> str:
    """Call Ollama."""
    response = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"]


# ─── AgentMemory: SQLite-backed memory ───
class AgentMemory:
    """SQLite-backed memory for an agent."""

    def __init__(self, db_path: str, agent_name: str):
        self.conn = sqlite3.connect(db_path)
        self.agent_name = agent_name
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def remember(self, key: str, value: str):
        """Store a memory."""
        self.conn.execute(
            "INSERT INTO memories (agent_name, key, value, created_at) VALUES (?,?,?,?)",
            (self.agent_name, key, value, datetime.now().isoformat())
        )
        self.conn.commit()
        print(f"  [Memory] 💾 Stored: '{key}' = '{value[:50]}...'")

    def recall(self, key: str) -> str | None:
        """Retrieve a memory by key (most recent)."""
        row = self.conn.execute(
            "SELECT value FROM memories WHERE agent_name=? AND key=? ORDER BY id DESC LIMIT 1",
            (self.agent_name, key)
        ).fetchone()
        if row:
            print(f"  [Memory] 📖 Recalled: '{key}' = '{row[0][:50]}...'")
            return row[0]
        print(f"  [Memory] ❓ No memory for key: '{key}'")
        return None

    def recall_all(self) -> list[dict]:
        """Retrieve all memories for this agent (most recent 50)."""
        rows = self.conn.execute(
            "SELECT key, value, created_at FROM memories WHERE agent_name=? ORDER BY id DESC LIMIT 50",
            (self.agent_name,)
        ).fetchall()
        return [{"key": r[0], "value": r[1], "when": r[2]} for r in rows]

    def close(self):
        self.conn.close()


# ─── Agent with Memory ───
RESEARCH_SYSTEM = """You are a Research Agent.
Your job: answer questions concisely.
If you have past memories, use them as context.
Always end with 'ANSWER_COMPLETE'.
"""

def agent_with_memory(
    question: str,
    memory: AgentMemory,
    model: str = "qwen3.5:4b",
) -> str:
    """Run agent with memory loaded into context."""

    # Load past memories into the system prompt
    past = memory.recall_all()
    memory_context = ""
    if past:
        memory_context = "\n\nPast memories (use as context):\n"
        for m in past[:5]:  # last 5 memories
            memory_context += f"- {m['key']}: {m['value'][:100]}\n"

    full_system = RESEARCH_SYSTEM + memory_context

    # Call the agent
    answer = call_agent(model, full_system, question)

    # Store the Q&A in memory
    memory.remember(f"question:{question[:50]}", answer[:200])

    return answer


# ─── Run it ───
if __name__ == "__main__":
    memory = AgentMemory("agent_memory.db", "research_agent")

    # ─── Session 1 ───
    print("=" * 60)
    print("SESSION 1: First question")
    print("=" * 60)
    answer1 = agent_with_memory(
        "What is FastAPI and why is it popular?",
        memory,
    )
    print(f"\nAnswer: {answer1}\n")

    # ─── Session 2 (simulating a new session) ───
    print("=" * 60)
    print("SESSION 2: Follow-up question (should have memory of Session 1)")
    print("=" * 60)
    answer2 = agent_with_memory(
        "Based on what you told me before, how does FastAPI compare to Flask?",
        memory,
    )
    print(f"\nAnswer: {answer2}\n")

    # ─── Check what's stored ───
    print("=" * 60)
    print("ALL MEMORIES STORED:")
    print("=" * 60)
    for m in memory.recall_all():
        print(f"  [{m['when']}] {m['key']}: {m['value'][:80]}...")

    memory.close()
```

**Step 2:** Run it:

```bash
python exercise4_memory.py
```

**Step 3:** Run it AGAIN (simulating a new session):

```bash
python exercise4_memory.py
```

### Expected Output

**First run:**
```
============================================================
SESSION 1: First question
============================================================
  [Memory] ❓ No memory for key: 'question:What is FastAPI and why is it popul'
  [Memory] 💾 Stored: 'question:What is FastAPI and why is it popul' = 'FastAPI is a modern...'

Answer: FastAPI is a modern Python web framework...
...

============================================================
SESSION 2: Follow-up question (should have memory of Session 1)
============================================================
  [Memory] 💾 Stored: 'question:What is FastAPI and why is it popul' = 'FastAPI is a modern...'

Answer: Based on my earlier research, FastAPI is...
...

============================================================
ALL MEMORIES STORED:
============================================================
  [2025-07-04T10:30:00] question:What is FastAPI...: FastAPI is a modern...
```

**Second run (proves persistence):**
```
============================================================
SESSION 1: First question
============================================================
  [Memory] 💾 Stored: 'question:What is FastAPI...' = '...'
  (memories from first run are already loaded as context)

Answer: ... (should reference previous knowledge)
```

### Verification

| Check | How to verify |
|---|---|
| Database is created | `ls -la agent_memory.db` — file exists |
| Memories are stored | "ALL MEMORIES STORED" section shows entries |
| Memories persist across runs | Run the script twice — second run has memories from first |
| Memory is loaded into context | System prompt includes "Past memories" section |
| SQLite queries work | Use `sqlite3 agent_memory.db "SELECT * FROM memories;"` to inspect |

**Bonus:** Create a second agent (e.g., `coding_agent`) with its own memory in the same database. Show that `research_agent` and `coding_agent` have separate memories (filter by `agent_name`).

```bash
# Inspect the database directly:
sqlite3 agent_memory.db "SELECT agent_name, key, value FROM memories;"
```

---

## Exercise 5 — Chain Two Agents Together (Researcher → Coder)

### Objective
Build a 2-agent pipeline: the Research Agent researches a topic, and the Coding Agent writes code based on that research. Combine everything: agent + harness + loop + memory.

### Steps

**Step 1:** Create `exercise5_pipeline.py`:

```python
"""
Exercise 5: Chain Two Agents Together (Researcher → Coder)
Surya Agentic Academy | Student Exercises

Goal: Build a pipeline where Research Agent output feeds into Coding Agent.
Combines: Agent + Harness + Loop + Memory
"""

import requests
import time
import sqlite3
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/chat"

# ═══════════════════════════════════════════════
# LAYER 1: AGENT — Call Ollama with a system prompt
# ═══════════════════════════════════════════════

def call_agent(model: str, system_prompt: str, user_message: str) -> str:
    """Call an Ollama model with a system prompt."""
    response = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"]


# ═══════════════════════════════════════════════
# LAYER 2: HARNESS — Retry + timeout + validation
# ═══════════════════════════════════════════════

class AgentHarness:
    """Safety wrapper: retry, timeout, validation."""

    def __init__(self, max_retries=3):
        self.max_retries = max_retries

    def run(self, agent_fn, *args, validate_fn=None, **kwargs):
        for attempt in range(self.max_retries):
            try:
                print(f"  [Harness] Attempt {attempt + 1}/{self.max_retries}...")
                result = agent_fn(*args, **kwargs)
                if validate_fn and not validate_fn(result):
                    print(f"  [Harness] Validation failed — retrying...")
                    time.sleep(1)
                    continue
                print(f"  [Harness] ✅ Success")
                return result
            except Exception as e:
                print(f"  [Harness] ❌ Error: {e}")
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Failed after {self.max_retries} retries")


# ═══════════════════════════════════════════════
# LAYER 3: LOOPS — Self-improvement cycle
# ═══════════════════════════════════════════════

JUDGE_SYSTEM = """You are a Judge Agent.
Score the output from 0.0 to 1.0.
Output EXACTLY:
SCORE: <float>
CRITIQUE: <one sentence>
END
"""

def parse_judge(output: str) -> tuple[float, str]:
    score, critique = 0.0, ""
    for line in output.strip().split("\n"):
        if line.startswith("SCORE:"):
            try: score = float(line.replace("SCORE:", "").strip())
            except ValueError: pass
        elif line.startswith("CRITIQUE:"):
            critique = line.replace("CRITIQUE:", "").strip()
    return score, critique

def self_improve_loop(
    agent_fn,
    system_prompt: str,
    task: str,
    model: str,
    max_iters: int = 2,
    threshold: float = 0.7,
) -> str:
    """Generate → Score → Adapt → Repeat"""
    current = call_agent(model, system_prompt, task)

    for i in range(max_iters):
        judge_out = call_agent(model, JUDGE_SYSTEM, f"Evaluate:\n\n{current}")
        score, critique = parse_judge(judge_out)
        print(f"  [Loop] Iter {i+1}: score={score:.2f} | {critique[:60]}")

        if score >= threshold:
            print(f"  [Loop] ✅ Threshold met")
            return current

        current = call_agent(
            model, system_prompt,
            f"Original task: {task}\n\nPrevious output:\n{current}\n\n"
            f"Judge critique (score {score:.2f}): {critique}\n\nImprove."
        )

    print(f"  [Loop] ⚠️ Max iters — returning best effort")
    return current


# ═══════════════════════════════════════════════
# LAYER 4: MEMORY — SQLite persistence
# ═══════════════════════════════════════════════

class AgentMemory:
    """SQLite-backed memory."""

    def __init__(self, db_path: str, agent_name: str):
        self.conn = sqlite3.connect(db_path)
        self.agent_name = agent_name
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT, key TEXT, value TEXT, created_at TEXT
            )
        """)
        self.conn.commit()

    def remember(self, key: str, value: str):
        self.conn.execute(
            "INSERT INTO memories (agent_name, key, value, created_at) VALUES (?,?,?,?)",
            (self.agent_name, key, value, datetime.now().isoformat())
        )
        self.conn.commit()

    def recall_all(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT key, value FROM memories WHERE agent_name=? ORDER BY id DESC LIMIT 5",
            (self.agent_name,)
        ).fetchall()
        return [{"key": r[0], "value": r[1]} for r in rows]


# ═══════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════

RESEARCH_SYSTEM = """You are the Research Agent in an AI Workforce.
Your job: research a topic and produce a technical brief for a coder.

Rules:
- Focus on practical, actionable info: libraries, APIs, patterns.
- Output as markdown sections.
- Keep it under 300 words.
- End with 'RESEARCH_COMPLETE'.
"""

CODER_SYSTEM = """You are the Coding Agent in an AI Workforce.
Your job: write Python code based on a research brief.

Rules:
- Write clean, runnable Python 3.11+ code.
- Include type hints and docstrings.
- Include a main() function.
- The code must be complete and runnable.
- End with 'CODE_COMPLETE'.
"""


# ═══════════════════════════════════════════════
# VALIDATORS
# ═══════════════════════════════════════════════

def validate_research(output: str) -> bool:
    return "RESEARCH_COMPLETE" in output and len(output) > 100

def validate_code(output: str) -> bool:
    return "CODE_COMPLETE" in output and ("def " in output or "class " in output)


# ═══════════════════════════════════════════════
# THE PIPELINE
# ═══════════════════════════════════════════════

def research_then_code(topic: str) -> dict:
    """2-agent pipeline: Research → Code"""
    harness = AgentHarness(max_retries=3)
    research_mem = AgentMemory("workforce_memory.db", "research_agent")
    coder_mem = AgentMemory("workforce_memory.db", "coding_agent")

    print(f"\n{'='*60}")
    print(f"PIPELINE: {topic}")
    print(f"{'='*60}")

    # ── Step 1: Research Agent ──
    print("\n📋 RESEARCH AGENT (qwen3.5:4b)")
    print("  Running through harness + self-improvement loop...")

    # Load past research into context
    past = research_mem.recall_all()
    memory_ctx = ""
    if past:
        memory_ctx = "\n\nPast research:\n" + "\n".join(
            f"- {m['key']}: {m['value'][:80]}" for m in past
        )

    research_brief = harness.run(
        self_improve_loop,
        agent_fn=call_agent,
        system_prompt=RESEARCH_SYSTEM + memory_ctx,
        task=f"Research this topic for a coder: {topic}",
        model="qwen3.5:4b",
        max_iters=2,
        threshold=0.7,
        validate_fn=validate_research,
    )

    # Save to memory
    research_mem.remember(topic[:50], research_brief[:300])
    print(f"  📦 Research: {len(research_brief)} chars (saved to memory)")

    # ── Step 2: Coding Agent ──
    print("\n💻 CODING AGENT (qwen2.5-coder:7b)")
    print("  Running through harness + self-improvement loop...")

    code = harness.run(
        self_improve_loop,
        agent_fn=call_agent,
        system_prompt=CODER_SYSTEM,
        task=f"""Based on this research, write a complete Python script:

{research_brief}

Write complete, runnable Python code.""",
        model="qwen2.5-coder:7b",
        max_iters=2,
        threshold=0.7,
        validate_fn=validate_code,
    )

    # Save to memory
    coder_mem.remember(topic[:50], code[:300])
    print(f"  📦 Code: {len(code)} chars (saved to memory)")

    research_mem.close()
    coder_mem.close()

    return {"topic": topic, "research": research_brief, "code": code}


# ═══════════════════════════════════════════════
# RUN IT
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    result = research_then_code(
        "How to build a Python CLI tool that fetches weather data from Open-Meteo API"
    )

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")

    print("\n" + "─" * 60)
    print("RESEARCH BRIEF:")
    print("─" * 60)
    print(result["research"])

    print("\n" + "─" * 60)
    print("GENERATED CODE:")
    print("─" * 60)
    print(result["code"])

    # Show memory was persisted
    print("\n" + "─" * 60)
    print("MEMORY CHECK:")
    print("─" * 60)
    conn = sqlite3.connect("workforce_memory.db")
    rows = conn.execute("SELECT agent_name, key, value FROM memories").fetchall()
    for r in rows:
        print(f"  [{r[0]}] {r[1][:40]}...: {r[2][:60]}...")
    conn.close()
```

**Step 2:** Run it:

```bash
python exercise5_pipeline.py
```

### Expected Output

```
============================================================
PIPELINE: How to build a Python CLI tool that fetches weather data from Open-Meteo API
============================================================

📋 RESEARCH AGENT (qwen3.5:4b)
  Running through harness + self-improvement loop...
  [Harness] Attempt 1/3...
  [Loop] Iter 1: score=0.60 | Missing specific API endpoint examples...
  [Loop] Iter 2: score=0.85 | Good, specific and actionable
  [Loop] ✅ Threshold met
  [Harness] ✅ Success
  📦 Research: 1800 chars (saved to memory)

💻 CODING AGENT (qwen2.5-coder:7b)
  Running through harness + self-improvement loop...
  [Harness] Attempt 1/3...
  [Loop] Iter 1: score=0.70 | Code is functional but missing error handling
  [Loop] Iter 2: score=0.88 | Complete, runnable, with error handling
  [Loop] ✅ Threshold met
  [Harness] ✅ Success
  📦 Code: 2400 chars (saved to memory)

============================================================
PIPELINE COMPLETE
============================================================
...
MEMORY CHECK:
  [research_agent] How to build a Python CLI tool that fetc...: ## Research Brief\n\n**Open-Meteo API**...
  [coding_agent] How to build a Python CLI tool that fetc...: ```python\nimport requests\nimport argparse...
```

### Verification

| Check | How to verify |
|---|---|
| Research Agent runs | Output includes "RESEARCH_COMPLETE" |
| Coding Agent runs | Output includes "CODE_COMPLETE" |
| Research feeds into Coder | Coder's input includes research brief |
| Harness catches failures | Temporarily use bad model — harness retries |
| Self-improvement loop runs | Console shows "Loop Iter 1" and "Iter 2" |
| Memory persists | "MEMORY CHECK" section shows stored entries |
| Code is runnable | Copy the generated code to a `.py` file and run it |
| Pipeline is end-to-end | One function call produces both research and code |

**Bonus challenges:**

1. **Add a third agent:** Create a QA Agent that tests the generated code. Add it after the Coding Agent in the pipeline.

2. **Add feedback loop:** If QA fails, send the bug report back to the Coding Agent to fix.

3. **Add tool access:** Give the Research Agent a web search tool (use the `requests` library to hit a search API). Give the Coding Agent a filesystem tool to write the code to a file.

4. **Run the pipeline twice:** The second run should load memories from the first run. The Research Agent should say "Based on my previous research..." in its output.

---

## Summary — What You've Built

| Exercise | What You Built | Architecture Layer |
|---|---|---|
| 1 | Single agent with system prompt | Agent |
| 2 | Agent + retry/timeout/validation | Harness |
| 3 | Agent + self-improvement cycle | Loops |
| 4 | Agent + SQLite memory | Memory |
| 5 | 2-agent pipeline with all layers | Agent → Harness → Loops → Memory |

**You now have the foundation of an AI Workforce.** Next steps:
- Add a CEO Agent that decomposes goals
- Add an Architect Agent that designs systems
- Add a QA Agent that tests code
- Add a Documentation Agent
- Add a Deployment Agent
- Connect MCP servers for external tools
- Grow from 2 agents to 7, then to 20

*"Start with 3 agents. Grow to 20. Build your AI company."*
— Surya Agentic Academy | Nagarjuna Reddy K
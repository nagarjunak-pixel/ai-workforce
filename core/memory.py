"""
Memory System — SQLite-backed persistent memory for all agents
Each agent role gets its own memory store. Pipeline state tracked across runs.
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "workforce.db")


def _get_db():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables"""
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agent_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            task TEXT,
            result_summary TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(role, task)
        );
        
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal TEXT NOT NULL,
            status TEXT DEFAULT 'running',
            agents_invoked TEXT,
            results TEXT,
            started_at TEXT NOT NULL,
            completed_at TEXT
        );
        
        CREATE TABLE IF NOT EXISTS pipeline_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            agent_name TEXT NOT NULL,
            role TEXT NOT NULL,
            task TEXT,
            result TEXT,
            handoff TEXT,
            duration_seconds REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES pipeline_runs(id)
        );
    """)
    conn.commit()
    conn.close()


def save_agent_memory(role: str, task: str, result_summary: str):
    """Save agent memory for a role"""
    conn = _get_db()
    conn.execute(
        "INSERT OR REPLACE INTO agent_memory (role, task, result_summary, created_at) VALUES (?, ?, ?, ?)",
        (role, task[:500], result_summary, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_agent_memory(role: str) -> str:
    """Get recent memory for a role"""
    conn = _get_db()
    rows = conn.execute(
        "SELECT task, result_summary FROM agent_memory WHERE role=? ORDER BY created_at DESC LIMIT 3",
        (role,)
    ).fetchall()
    conn.close()
    if not rows:
        return ""
    return " | ".join([f"{r['task'][:100]} → {r['result_summary'][:200]}" for r in rows])


def create_pipeline_run(goal: str) -> int:
    """Create a new pipeline run, return its ID"""
    conn = _get_db()
    cursor = conn.execute(
        "INSERT INTO pipeline_runs (goal, status, started_at) VALUES (?, 'running', ?)",
        (goal, datetime.now().isoformat())
    )
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id


def save_pipeline_step(run_id: int, agent_output: dict):
    """Save a single step in a pipeline run"""
    conn = _get_db()
    conn.execute(
        """INSERT INTO pipeline_steps 
        (run_id, agent_name, role, task, result, handoff, duration_seconds, created_at) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, agent_output["agent"], agent_output["role"], 
         agent_output["task"], agent_output["result"], 
         agent_output.get("handoff"), agent_output["duration_seconds"],
         agent_output["timestamp"])
    )
    conn.commit()
    conn.close()


def complete_pipeline_run(run_id: int, results: list, status: str = "completed"):
    """Mark a pipeline run as complete"""
    conn = _get_db()
    conn.execute(
        "UPDATE pipeline_runs SET status=?, results=?, completed_at=? WHERE id=?",
        (status, json.dumps(results, indent=2), datetime.now().isoformat(), run_id)
    )
    conn.commit()
    conn.close()


def get_pipeline_history(limit: int = 10) -> list:
    """Get recent pipeline runs"""
    conn = _get_db()
    rows = conn.execute(
        "SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_run_details(run_id: int) -> dict:
    """Get full details of a pipeline run including all steps"""
    conn = _get_db()
    run = conn.execute("SELECT * FROM pipeline_runs WHERE id=?", (run_id,)).fetchone()
    steps = conn.execute(
        "SELECT * FROM pipeline_steps WHERE run_id=? ORDER BY id", (run_id,)
    ).fetchall()
    conn.close()
    return {"run": dict(run) if run else None, "steps": [dict(s) for s in steps]}
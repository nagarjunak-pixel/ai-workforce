"""Memory System — SQLite-backed persistence for the AI Workforce.

Memory is what makes the workforce STATEFUL.
Without it, every call is amnesia. With it, agents remember:
- What was asked before
- What worked and what didn't
- The current project state

SQLite is built into Python stdlib — zero infrastructure needed.
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional


class MemorySystem:
    """SQLite-backed memory for the AI Workforce.
    
    Three tables:
    - interactions: Every agent call (prompt + response + metadata)
    - project_state: Current project context (key-value store)
    - agent_knowledge: Per-agent learned facts (key-value per agent)
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.expanduser("~/Desktop/ai-workforce/workforce.db")
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema — idempotent."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                agent_role TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                score REAL DEFAULT 0,
                timestamp TEXT DEFAULT (datetime('now'))
            )
        """)
        
        c.execute("""
            CREATE TABLE IF NOT EXISTS project_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        
        c.execute("""
            CREATE TABLE IF NOT EXISTS agent_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                UNIQUE(agent_name, key)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_interaction(self, agent_name: str, agent_role: str,
                           prompt: str, response: str, score: float = 0):
        """Record an agent interaction to memory."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO interactions (agent_name, agent_role, prompt, response, score)
            VALUES (?, ?, ?, ?, ?)
        """, (agent_name, agent_role, prompt, response, score))
        conn.commit()
        conn.close()
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get recent interactions — the workforce's short-term memory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""
            SELECT * FROM interactions ORDER BY id DESC LIMIT ?
        """, (limit,))
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows
    
    def get_agent_history(self, agent_name: str, limit: int = 5) -> List[Dict]:
        """Get history for a specific agent."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""
            SELECT * FROM interactions WHERE agent_name = ? 
            ORDER BY id DESC LIMIT ?
        """, (agent_name, limit))
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows
    
    def set_project_state(self, key: str, value: str):
        """Set a project state variable — the workforce's working memory."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO project_state (key, value, updated_at)
            VALUES (?, ?, datetime('now'))
        """, (key, value))
        conn.commit()
        conn.close()
    
    def get_project_state(self, key: str) -> Optional[str]:
        """Get a project state variable."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT value FROM project_state WHERE key = ?", (key,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None
    
    def get_all_project_state(self) -> Dict[str, str]:
        """Get all project state — useful for CEO agent context."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT key, value FROM project_state")
        rows = {r["key"]: r["value"] for r in c.fetchall()}
        conn.close()
        return rows
    
    def set_agent_knowledge(self, agent_name: str, key: str, value: str):
        """Store learned knowledge for a specific agent."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO agent_knowledge (agent_name, key, value)
            VALUES (?, ?, ?)
        """, (agent_name, key, value))
        conn.commit()
        conn.close()
    
    def get_agent_knowledge(self, agent_name: str, key: str) -> Optional[str]:
        """Retrieve learned knowledge for a specific agent."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            SELECT value FROM agent_knowledge WHERE agent_name = ? AND key = ?
        """, (agent_name, key))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None
    
    def stats(self) -> dict:
        """Memory statistics — useful for dashboard."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM interactions")
        interactions = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM project_state")
        state_keys = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM agent_knowledge")
        knowledge_items = c.fetchone()[0]
        conn.close()
        return {
            "total_interactions": interactions,
            "project_state_keys": state_keys,
            "agent_knowledge_items": knowledge_items,
        }
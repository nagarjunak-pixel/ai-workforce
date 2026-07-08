"""
Pipeline Orchestrator — The heart of the AI Workforce system.
Takes a goal, runs it through the full agent pipeline:
CEO → Research → Architect → Coding → QA → Docs → Deploy → Marketing

Each agent receives the accumulated context from all previous agents.
No agent waits — the pipeline flows automatically.
"""
import sys
import os
import json
import re
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.memory import (
    init_db, create_pipeline_run, save_pipeline_step,
    complete_pipeline_run, get_pipeline_history, get_run_details
)
from core.ollama_client import list_models
from agents.ceo import create_ceo_agent
from agents.research import create_research_agent
from agents.architect import create_architect_agent
from agents.coding import create_coding_agent, extract_code_files, write_code_files
from agents.qa import create_qa_agent
from agents.docs import create_docs_agent
from agents.deploy import create_deploy_agent
from agents.marketing import create_marketing_agent


# Agent registry — maps role to factory function
AGENT_FACTORIES = {
    "ceo": create_ceo_agent,
    "research": create_research_agent,
    "architect": create_architect_agent,
    "coding": create_coding_agent,
    "qa": create_qa_agent,
    "docs": create_docs_agent,
    "deploy": create_deploy_agent,
    "marketing": create_marketing_agent,
}

# Default pipeline order
DEFAULT_PIPELINE = ["ceo", "research", "architect", "coding", "qa", "docs", "deploy", "marketing"]


class Pipeline:
    """Runs a goal through the full AI workforce pipeline"""
    
    def __init__(self, goal: str, output_dir: str = None):
        self.goal = goal
        self.output_dir = output_dir or str(PROJECT_ROOT / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S"))
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.results = []
        self.run_id = None
        
        # Initialize database
        init_db()
        
        # Create all agents upfront
        self.agents = {}
        for role, factory in AGENT_FACTORIES.items():
            self.agents[role] = factory()
    
    def run(self, pipeline_order: list = None) -> list:
        """Run the full pipeline. Returns list of agent outputs."""
        pipeline_order = pipeline_order or DEFAULT_PIPELINE
        
        print(f"\n{'='*60}")
        print(f"🚀 AI WORKFORCE PIPELINE STARTING")
        print(f"{'='*60}")
        print(f"📋 Goal: {self.goal}")
        print(f"📁 Output: {self.output_dir}")
        print(f"👥 Agents: {' → '.join(pipeline_order)}")
        print(f"{'='*60}\n")
        
        # Create pipeline run in DB
        self.run_id = create_pipeline_run(self.goal)
        
        # Check Ollama is running
        models = list_models()
        if not models:
            print("❌ Ollama not running! Start it with: ollama serve")
            return []
        print(f"✅ Ollama models available: {len(models)}\n")
        
        # Phase 1: CEO plans the work
        print("━━━ PHASE 1: CEO PLANNING ━━━")
        ceo = self.agents["ceo"]
        ceo_output = ceo.execute(self.goal)
        self._process_output(ceo_output)
        
        # Parse CEO's plan to get tasks for each agent
        tasks = self._parse_ceo_plan(ceo_output["result"])
        
        # Phase 2: Execute each agent in sequence
        print("\n━━━ PHASE 2: AGENT EXECUTION ━━━")
        for role in pipeline_order[1:]:  # Skip CEO, already done
            agent = self.agents[role]
            
            # Get task from CEO's plan, or use a default
            task = tasks.get(role, f"Execute your role for: {self.goal}")
            
            # Build context with all previous agent outputs
            context = {"pipeline_history": self.results}
            
            print(f"\n--- {agent.name} ---")
            output = agent.execute(task, context)
            self._process_output(output)
            
            # If coding agent, extract and write code files
            if role == "coding":
                files = extract_code_files(output["result"])
                if files:
                    print(f"\n  📦 Extracting {len(files)} code files...")
                    written = write_code_files(files, self.output_dir)
                    output["files_written"] = written
            
            # If QA, also extract test files
            if role == "qa":
                files = extract_code_files(output["result"])
                if files:
                    written = write_code_files(files, self.output_dir)
                    output["files_written"] = written
            
            # If deploy, extract config files
            if role == "deploy":
                files = extract_code_files(output["result"])
                if files:
                    written = write_code_files(files, self.output_dir)
                    output["files_written"] = written
            
            # If docs, extract README
            if role == "docs":
                files = extract_code_files(output["result"])
                if files:
                    written = write_code_files(files, self.output_dir)
                    output["files_written"] = written
            
            # If marketing, extract landing page
            if role == "marketing":
                files = extract_code_files(output["result"])
                if files:
                    written = write_code_files(files, self.output_dir)
                    output["files_written"] = written
        
        # Complete the run
        complete_pipeline_run(self.run_id, self.results)
        
        # Save full results to JSON
        results_path = Path(self.output_dir) / "pipeline_results.json"
        results_path.write_text(json.dumps(self.results, indent=2, default=str))
        
        self._print_summary()
        return self.results
    
    def _process_output(self, output: dict):
        """Process and store agent output"""
        self.results.append(output)
        if self.run_id:
            save_pipeline_step(self.run_id, output)
        
        # Print result preview
        result_preview = output["result"][:500]
        print(f"\n  📄 Output preview:\n  {result_preview}...\n")
    
    def _parse_ceo_plan(self, ceo_result: str) -> dict:
        """Parse CEO's numbered plan to extract tasks per agent"""
        tasks = {}
        lines = ceo_result.split("\n")
        for line in lines:
            # Match: 1. [Agent: research] — task description
            match = re.match(r'\d+\.\s*\[Agent:\s*(\w+)\s*\]\s*[—\-]\s*(.+)', line, re.IGNORECASE)
            if match:
                role = match.group(1).lower().strip()
                task = match.group(2).strip()
                tasks[role] = task
        return tasks
    
    def _print_summary(self):
        """Print pipeline completion summary"""
        total_duration = sum(r.get("duration_seconds", 0) for r in self.results)
        print(f"\n{'='*60}")
        print(f"✅ PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"📊 Agents invoked: {len(self.results)}")
        print(f"⏱️  Total duration: {total_duration:.1f}s")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"💾 Pipeline run ID: {self.run_id}")
        print(f"{'='*60}")
        
        # List all files written
        all_files = []
        for r in self.results:
            if r.get("files_written"):
                all_files.extend(r["files_written"])
        if all_files:
            print(f"\n📝 Files written ({len(all_files)}):")
            for f in all_files:
                print(f"   {f}")


def run_pipeline(goal: str, output_dir: str = None):
    """Convenience function to run a pipeline"""
    pipeline = Pipeline(goal, output_dir)
    return pipeline.run()


def show_history(limit: int = 10):
    """Show recent pipeline runs"""
    init_db()
    runs = get_pipeline_history(limit)
    if not runs:
        print("No pipeline runs yet.")
        return
    print(f"\n{'='*60}")
    print(f"📋 PIPELINE HISTORY (last {limit} runs)")
    print(f"{'='*60}")
    for r in runs:
        print(f"\n  Run #{r['id']} | {r['goal'][:60]}...")
        print(f"  Status: {r['status']} | Started: {r['started_at']}")
        if r.get('completed_at'):
            print(f"  Completed: {r['completed_at']}")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:])
        run_pipeline(goal)
    else:
        print("Usage: python pipeline.py <your project goal>")
        print("Example: python pipeline.py 'Build a REST API for a todo app with FastAPI'")
        print("\nOr run the demo:")
        show_history()
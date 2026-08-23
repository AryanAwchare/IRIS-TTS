import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task_history.db")

def seed():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS task_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            task_name TEXT NOT NULL,
            user_request TEXT NOT NULL,
            summary TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'COMPLETED',
            files_modified TEXT,
            remaining_actions TEXT
        );
    """)

    initial_tasks = [
        (
            datetime.now().isoformat(),
            "GPU vs CPU & Google Colab Workflow Explanation",
            "what difference does gpu make and also how do i connect colab and what will be the working for it ?",
            "Explained CPU vs GPU hardware architecture, CUDA cores, speedups (10x-100x), VRAM significance, step-by-step Colab GPU setup, cloud VM execution model, and provided curated learning resources.",
            "COMPLETED",
            "",
            "None"
        ),
        (
            datetime.now().isoformat(),
            "Project Depth & Pre-Implementation Research Guidelines",
            "add guideline to explain me more about depth of core idea whenever creating new project and also ways to improve quality of work like researach which needs to be done",
            "Created comprehensive guidelines covering core problem definition, domain modeling, NFRs, pre-implementation research (PoC, build vs adopt, interface-first design, failure mode analysis), and quality engineering standards.",
            "COMPLETED",
            ".agents/memory/project_depth_and_research_guidelines.md",
            "None"
        ),
        (
            datetime.now().isoformat(),
            "Memory-First Check Protocol & SQLite History System",
            "copy mimo or install it to store history in memory so the ais can acess previous history and also keep a guideline to first check memory then see if something is remaining or not then perform given acction and now save every previous task into sql lite format with summery",
            "Implemented Memory-First Check Rule, Project Depth Rule, SQLite Task History Logger script, and seeded task_history.db to persist task history in SQLite format across sessions.",
            "COMPLETED",
            ".agents/rules/memory_first_check.md, .agents/rules/project_depth_and_research.md, .agents/memory/task_history_logger.py, .agents/memory/memory_first_and_sqlite_history.md",
            "Continuously auto-log all future prompts and completed actions to task_history.db"
        )
    ]

    for task in initial_tasks:
        cursor.execute("""
            INSERT INTO task_history (timestamp, task_name, user_request, summary, status, files_modified, remaining_actions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, task)
    
    conn.commit()
    conn.close()
    print("Database successfully seeded at:", DB_PATH)

if __name__ == "__main__":
    seed()

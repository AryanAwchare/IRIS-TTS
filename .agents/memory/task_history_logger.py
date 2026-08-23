import sqlite3
import os
import sys
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task_history.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
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
    conn.commit()
    conn.close()

def log_task(task_name, user_request, summary, status="COMPLETED", files_modified="", remaining_actions=""):
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO task_history (timestamp, task_name, user_request, summary, status, files_modified, remaining_actions)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, task_name, user_request, summary, status, files_modified, remaining_actions))
    conn.commit()
    conn.close()

def fetch_recent_tasks(limit=10):
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, task_name, user_request, summary, status, remaining_actions
        FROM task_history
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def print_history():
    rows = fetch_recent_tasks(20)
    print("=" * 80)
    print(" TASK HISTORY (SQLITE DB) ")
    print("=" * 80)
    if not rows:
        print("No task records found.")
        return
    for r in rows:
        print(f"[{r['id']}] {r['timestamp']} | Status: {r['status']}")
        print(f" Task: {r['task_name']}")
        print(f" Summary: {r['summary']}")
        if r['remaining_actions']:
            print(f" Remaining: {r['remaining_actions']}")
        print("-" * 80)

if __name__ == "__main__":
    init_db()
    if len(sys.argv) > 1 and sys.argv[1] == "--show":
        print_history()
    else:
        print(f"SQLite database initialized at {DB_PATH}")

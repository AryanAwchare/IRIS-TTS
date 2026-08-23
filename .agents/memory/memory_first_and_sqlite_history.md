# Memory System & SQLite Task History Index

This folder serves as the central persistent memory store for AI agents working in this workspace.

---

## Active Memory Index

1. **Memory-First Protocol & Rule**:
   * File: [memory_first_check.md](file:///c:/Users/dell/OneDrive/Desktop/IRIS/.agents/rules/memory_first_check.md)
   * Core directive: Always check `.agents/memory/task_history.db` and memory markdown docs before starting any task. Identify remaining items and act accordingly.

2. **Project Depth & Research Guidelines**:
   * File: [project_depth_and_research_guidelines.md](file:///c:/Users/dell/OneDrive/Desktop/IRIS/.agents/memory/project_depth_and_research_guidelines.md)
   * Rule File: [project_depth_and_research.md](file:///c:/Users/dell/OneDrive/Desktop/IRIS/.agents/rules/project_depth_and_research.md)
   * Core directive: Deep problem definition, NFR specification, PoC research, interface-first design, and zero-hack quality execution.

3. **SQLite Task History Database**:
   * Database Path: `file:///c:/Users/dell/OneDrive/Desktop/IRIS/.agents/memory/task_history.db`
   * Logger Script: [task_history_logger.py](file:///c:/Users/dell/OneDrive/Desktop/IRIS/.agents/memory/task_history_logger.py)
   * Schema:
     * `id`: INTEGER PRIMARY KEY AUTOINCREMENT
     * `timestamp`: TEXT (ISO Format)
     * `task_name`: TEXT
     * `user_request`: TEXT
     * `summary`: TEXT
     * `status`: TEXT ('COMPLETED' | 'IN_PROGRESS' | 'FAILED')
     * `files_modified`: TEXT
     * `remaining_actions`: TEXT

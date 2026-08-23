# RULE: Memory-First Protocol & Task History Verification

## Mandatory Rule Execution Order

Whenever receiving a user prompt or task, the AI agent MUST perform operations in the following mandatory order:

```
[ New User Prompt / Request ]
              │
              ▼
  1. QUERY MEMORY & SQLITE HISTORY
     - Read task_history.db & .agents/memory/
     - Check what tasks were completed previously
     - Identify pending / remaining items
              │
              ▼
  2. DELTA & GAP ANALYSIS
     - Determine what part of the request is already fulfilled
     - Identify precisely what is missing or remaining
              │
              ▼
  3. PERFORM ACTION & VERIFY
     - Execute required code/edits with high quality
     - Run empirical verification commands
              │
              ▼
  4. RECORD TASK TO SQLITE & MEMORY
     - Log prompt, summary, status, and remaining tasks into SQLite
```

---

## Detailed Step Protocols

### Step 1: Memory-First Check
* **Check SQLite Database**: Query `.agents/memory/task_history.db` for recent tasks related to the current domain or request.
* **Check Memory Docs**: Inspect `.agents/memory/` and `.agents/rules/` for established project conventions, design guidelines, and tech stack constraints.

### Step 2: Gap & Remaining Task Analysis
* Compare current request against past history records.
* Avoid re-implementing existing features or breaking existing working logic.
* Focus execution strictly on filling remaining gaps.

### Step 3: Execution & High-Quality Standards
* Apply **Project Depth & Research Guidelines**: Ensure problem definition, NFRs, PoC research, and clean domain modeling are adhered to.
* Never use dummy fallbacks or swallow errors silently.

### Step 4: Mandatory Task Logging to SQLite
* At the conclusion of every turn or task milestone, append a new record into `.agents/memory/task_history.db` containing:
  * `timestamp`: ISO timestamp
  * `task_name`: Short title of the task
  * `user_request`: Complete user prompt
  * `summary`: Exhaustive summary of what was completed
  * `status`: 'COMPLETED' | 'IN_PROGRESS' | 'FAILED'
  * `remaining_actions`: Outstanding items or follow-ups for subsequent turns.

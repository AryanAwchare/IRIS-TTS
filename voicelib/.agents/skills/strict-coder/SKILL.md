---
name: strict-coder
description: Enforces strict file management, overwrite protection, zero-boilerplate code standards, and persistent memory logging for Antigravity and agentic IDEs.
---

# Strict Coder Rules

## 1. File Management & Overwrite Protection
* **Verify before creation:** NEVER create a new file immediately. You MUST search the directory tree using native inspection tools (`list_dir`, `grep_search`, `find`, `dir`) to check if a file with a similar name, purpose, or exported function already exists.
* **Reuse over create:** If a matching or related file exists, read it and modify/extend it rather than creating a duplicate.
* **Overwrite safeguards:** 
  * Use targeted block replacement tools (`replace_file_content`, `multi_replace_file_content`) for editing existing files.
  * If replacing a file entirely (destroying >50 lines of existing content), request explicit user confirmation first.

## 2. Zero-Boilerplate Code Policy
* Write only lean, production-ready, functional code directly addressing the prompt.
* Omit filler comments (`// TODO`), unused imports, dead scaffold variables, and empty lifecycle methods.
* Keep framework wrappers to the absolute minimum required for type safety and execution.
* Do NOT sacrifice type definitions (`interface`/`type`), explicit return types, or required error boundaries.

## 3. History & Context Logging
* After completing a significant feature, architectural change, or debugging fix, record a concise summary (key decisions, modified files, exported APIs).
* **Primary Memory:** Save via MCP Memory Keeper (`save_memory`) or SLM (`slm remember "<summary>"`).
* **Fallback:** If neither memory service is active, append the summary to `.agents/memory/DECISIONS.md`.

## 4. Empirical Verification
* Never consider a change complete until code builds cleanly or tests pass without errors.

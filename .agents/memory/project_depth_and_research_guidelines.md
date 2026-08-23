# Project Depth & Research Guidelines

This document outlines the core principles, research methodologies, and quality assurance standards required when conceiving, planning, and executing new projects.

---

## 1. Depth of Core Idea (Project Conception & Architecture)

Before writing code or initializing repositories, establish a deep conceptual foundation for the project across three core dimensions:

### 1.1 Problem Definition & Value Proposition
* **The "Why" Test**: Articulate the exact problem being solved in 1–2 concise sentences. Avoid building features without a clear root problem.
* **Core Value Bottleneck**: Identify the single primary operation or experience that determines success (e.g., latency in real-time audio, accuracy in search, zero-friction auth). Focus 80% of initial architectural depth on this bottleneck.

### 1.2 Domain Modeling & Data Flow
* **Entity Relationships**: Map out primary data entities, their attributes, and relationships before choosing a database engine or ORM.
* **State Machine & Lifecycle**: Define clear states for critical domain objects (e.g., `Pending` → `Processing` → `Completed` / `Failed`). Ensure edge cases and invalid transitions are impossible by design.
* **Single Source of Truth**: Explicitly specify which layer owns what data to prevent duplicated or out-of-sync state across frontend, backend, and cache layers.

### 1.3 Scope & Non-Functional Requirements (NFRs)
* **Performance SLAs**: Define clear latency and throughput goals (e.g., API response <200ms, streaming initialization <500ms).
* **Security & Auth Boundaries**: Define authentication models, authorization checks (RBAC/ABAC), data sanitization, and secrets management upfront.
* **Scalability Boundaries**: Establish clear bounds for initial scale (e.g., concurrent users, storage limits) to avoid over-engineering while maintaining clean separation of concerns.

---

## 2. Research Methodology & Pre-Implementation Planning

To maximize quality, every major project or feature must undergo structured research before implementation begins.

### 2.1 Technology Stack & Ecosystem Evaluation
* **Build vs. Adopt**: Evaluate pre-existing libraries and frameworks against custom implementations. Choose battle-tested tools unless a custom solution yields a 10x performance or UX improvement.
* **Compatibility & Risk Matrix**: Assess licensing, maintenance status, system dependencies (e.g., Python version, GPU/CUDA requirements, OS compatibility), and breaking changes.
* **Proof of Concept (PoC)**: Build tiny, isolated prototypes for risky or uncertain components (e.g., testing model inference speed or dynamic stream handling) before committing to full architecture.

### 2.2 Architectural & API Contract Design
* **Interface-First Design**: Define API specs (REST/OpenAPI, GraphQL, or RPC schemas) and component props before writing implementation code.
* **Failure Mode Analysis**: Identify where things will break (e.g., network timeouts, DB connection pool exhaustion, invalid payload, third-party service downtime) and design fallback strategies.

### 2.3 Benchmarking & Competitor Analysis
* **Prior Art Analysis**: Study 2–3 existing implementations or competing tools to identify UX strengths, missing edge cases, and architectural pitfalls.
* **Metrics Baseline**: Establish baseline benchmarks for memory usage, CPU/GPU utilization, and asset payload sizes.

---

## 3. Work Quality & Execution Standards

### 3.1 Quality Engineering & Testing
* **Test-Driven / Quality-First Development**: Write unit tests for core domain logic, integration tests for API contracts, and end-to-end tests for primary user flows.
* **Zero-Superficial Fixes**: Never swallow errors or use temporary hacks (`catch (e) {}` / fallback dummy values) without logging and addressing the underlying root cause.

### 3.2 Code Hygiene & Maintainability
* **Modular Decoupling**: Keep business logic completely separate from presentation (UI) and infrastructure (DB/HTTP) drivers.
* **Self-Documenting Code & Project Memory**: Maintain clean code structure, clear naming conventions, and keep project documentation (`README.md`, changelogs, architecture notes) updated continuously.

### 3.3 Verification & Continuous Feedback Loop
* **Empirical Verification**: Never declare a feature or project stage complete without running live verification commands, automated test suites, or end-to-end smoke tests.
* **Iterative Refinement**: Release minimal, high-quality, fully functioning increments over large, incomplete, monolithic code dumps.

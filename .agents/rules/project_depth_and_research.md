# RULE: Project Depth & Pre-Implementation Research

## Architectural Depth Requirements
1. **Root Problem Focus**: Articulate the core value proposition and value bottleneck before writing code.
2. **Domain Modeling**: Explicitly define entity relationships and state lifecycles to prevent invalid states.
3. **Non-Functional Requirements**: Specify performance SLAs, security/auth boundaries, and scale limits upfront.

## Pre-Implementation Research Requirements
1. **Ecosystem & Build vs. Adopt**: Evaluate battle-tested tools; build custom only for 10x ROI.
2. **Proof of Concept (PoC)**: Prototype risky/unknown components before full implementation.
3. **Interface-First Design**: Define API contracts, data models, and prop schemas prior to writing implementation logic.
4. **Failure Mode Analysis**: Design explicit handling for network timeouts, DB pool limits, and external service failures.

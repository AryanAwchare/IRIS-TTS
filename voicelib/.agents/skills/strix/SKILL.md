---
name: strix
description: Autonomous AI penetration testing and vulnerability assessment skill inspired by usestrix/strix. Audits web applications, APIs, authentication, and codebases for security vulnerabilities, logic flaws, injection vectors, and data exposure.
metadata:
  model: inherit
---

# 🛡️ Strix - AI Security Penetration Testing & Vulnerability Assessment

`strix` is an agentic AI penetration testing & vulnerability assessment skill designed to discover, validate, and remediate security vulnerabilities across web applications, APIs, backend services, and codebases.

---

## 🎯 Use This Skill When:
- Conducting comprehensive security audits of web applications, APIs, or full-stack codebases.
- Scanning for OWASP Top 10 vulnerabilities (SQLi, XSS, SSRF, IDOR, Broken Authentication, Sensitive Data Exposure, RCE, CSRF).
- Auditing environment files, secrets, JWT handling, CORS headers, rate limiting, and permission checks.
- Running pre-deployment or post-development security checks to eliminate bugs and vulnerabilities before production release.

---

## 🚀 Penetration Testing & Security Audit Workflow

### 1. Reconnaissance & Scope Definition
- Analyze project architecture, frameworks, database connections, and API routes.
- Scan configuration files (`.env`, `package.json`, server setup, CORS configuration, headers).
- Identify sensitive endpoints (auth, payment, admin, file upload, user data).

### 2. Vulnerability Scanning & Code Audit
Inspect the codebase systematically for:
- **Authentication & Authorization**: Weak password hashing, unverified JWT tokens, missing middleware checks, IDOR (Insecure Direct Object References).
- **Injection Vectors**: SQL Injection, NoSQL Injection, Command Injection, Unsanitized User Input, XSS (Cross-Site Scripting).
- **Data Exposure & Secrets**: Hardcoded API keys, unmasked `.env` variables, verbose error stack traces in production responses.
- **Session & Token Management**: Insecure cookie flags (`HttpOnly`, `Secure`, `SameSite`), non-expiring sessions, weak secret keys.
- **Logic Flaws & Edge Cases**: Race conditions, improper state transitions, integer overflow, missing input validation schemas.

### 3. Exploitation & Risk Validation
- Formulate realistic attack scenarios and PoCs (Proofs of Concept) to verify if identified issues are exploitable.
- Determine severity level (CRITICAL, HIGH, MEDIUM, LOW) based on impact and exploitability.

### 4. Remediation & Patching
- Provide immediate, secure code patches to fix discovered vulnerabilities.
- Re-test modified code to confirm the vulnerability is fully mitigated without breaking functionality.

---

## 📌 Security Rule Guidelines
Always run a final security assessment using Strix methodology before declaring a project complete.

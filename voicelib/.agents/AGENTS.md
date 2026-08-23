# Project Security & Quality Assurance Guidelines

## 🛡️ Mandatory End-of-Project Security & Bug Audit Guideline

Before completing or releasing **ANY** project, you MUST perform a comprehensive security audit and vulnerability test following the **Strix Penetration Testing Methodology**.

### 1. Mandatory End-of-Project Security Checklist
- **Secrets & Credentials Audit**: Ensure no production keys, database strings, JWT secrets, or private tokens are hardcoded or leaked in client bundles or public endpoints.
- **Authentication & Authorization**: Verify that all protected routes enforce valid session/JWT checks and prevent IDOR (Insecure Direct Object Reference).
- **Input Validation & Injection Prevention**: Sanitize and validate all user inputs against SQLi, NoSQLi, Command Injection, XSS, and SSRF.
- **Data Protection**: Ensure password hashes use strong algorithms (e.g., bcrypt/argon2), sensitive fields are masked, and CORS/CSRF headers are strictly configured.
- **Error Handling**: Ensure detailed error tracebacks and stack traces are suppressed in production.
- **Logic & Edge Cases**: Test race conditions, boundary limits, invalid payload formats, and edge-case handling.

### 2. Required Remediation Protocol
- If any bug, vulnerability, or "hackable thing" is found during testing:
  1. Classify the threat level (CRITICAL, HIGH, MEDIUM, LOW).
  2. Implement clean, robust fixes immediately in the codebase.
  3. Re-verify the fix with test cases to confirm 100% resolution.
- Never mark a project complete or hand over to production with unpatched bugs or security flaws to prevent data loss or security breaches.

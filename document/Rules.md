# AI Coding Rules & Boundaries — IRIS / VoiceLib

> **Project Name:** IRIS (VoiceLib)  
> **Document Version:** 1.0.0  
> **Target AI Assistants:** Antigravity, Claude, ChatGPT, GitHub Copilot  

---

## 1. Core Technical Stack Boundaries

### Approved Frameworks & Libraries
* **Frontend:** React 18, Vite, Tailwind CSS v3, Zustand (state), Axios (HTTP), Lucide React (Icons).
* **Backend:** Python 3.11+, FastAPI 0.111+, Pydantic V2, SQLAlchemy 2.0 Async, Alembic, `mutagen` (audio validation).
* **Database & Drivers:** PostgreSQL 15+ using `asyncpg` driver (`postgresql+asyncpg://...`).
* **TTS Engine:** Kyutai `pocket-tts` with CPU optimization, or `MockTTSModel` for development.
* **Storage Client:** `boto3` configured for S3 / MinIO / Cloudflare R2 compatibility.

### Strictly Forbidden Libraries & Anti-Patterns
* 🚫 **DO NOT** install Redux, MobX, or React Context for global state management — use **Zustand** exclusively.
* 🚫 **DO NOT** use synchronous database drivers (such as `psycopg2` or standard `sqlite3`) in API routes — all database interactions must be non-blocking async (`async await`).
* 🚫 **DO NOT** use UI component monoliths like Bootstrap, Material UI, or Ant Design — use **Tailwind CSS v3** utility classes and custom glassmorphism components.
* 🚫 **DO NOT** introduce arbitrary pixel offset calculations or inline element styling (`style={{...}}`) where Tailwind utility classes or dynamic layout math should be used.
* 🚫 **DO NOT** use unhandled `try/except: pass` or swallow exceptions silently in backend routers.

---

## 2. Code Quality & Architectural Standards

### 2.1 Backend (Python / FastAPI)
1. **Type Annotations:** All Python functions, Pydantic schemas, and API request parameters must include explicit type hints (`str`, `int`, `Optional[UUID]`, etc.).
2. **Pydantic V2:** Use `model_config = ConfigDict(...)` and strict schema validation for all payload inputs and responses.
3. **Async Standard:** All API route handlers (`@router.post`, `@router.get`) must be declared as `async def` and non-blocking.
4. **Dependency Injection:** Database sessions (`get_db`) and authentication objects (`get_current_user`) MUST be passed using FastAPI `Depends()`.
5. **Database Models:** Keep database models in `app/models.py`. Use SQLAlchemy 2.0 `Mapped[...]` type annotations.

### 2.2 Frontend (React / Tailwind)
1. **Functional Components Only:** Write modern React functional components with hooks.
2. **Zustand Stores:** Persist JWT token and auth state using Zustand's `persist` middleware. Keep stores modular (`useAuthStore`, `useVoiceStore`).
3. **Error Banner Handling:** Axios client must normalize error responses to standard strings and pass them to the UI error state.
4. **Accessible Inputs:** Every form control must have associated `<label>` tags or `aria-label` attributes for accessibility.
5. **Double-Bezel Glassmorphism:** Adhere strictly to the "Ethereal Glass / Obsidian" visual theme defined in `Design.md`.

---

## 3. Error Handling & Security Guidelines

### 3.1 Error Response Specification
All backend errors must follow a uniform JSON structure:

```json
{
  "detail": "Descriptive error message for the client UI",
  "code": "SPECIFIC_ERROR_CODE"
}
```

Standard Status Codes to enforce:
* `400 Bad Request`: Validation failure (e.g. audio duration > 30s).
* `401 Unauthorized`: Missing or invalid JWT Bearer token.
* `403 Forbidden`: Attempting to access or delete another user's voice or audio generation.
* `404 Not Found`: Voice or generation record does not exist.
* `422 Unprocessable Entity`: Invalid JSON payload body.
* `500 Internal Server Error`: Unhandled system failure (must log full traceback silently).

### 3.2 Security Safeguards
* **Zero Hardcoded Credentials:** Secrets (`JWT_SECRET_KEY`, `STORAGE_SECRET_KEY`) must NEVER be committed to Git. Read directly from environment variables or `.env`.
* **File Upload Sanitization:** Validate audio files twice: on the client before upload, and on the backend using `mutagen` for actual MIME header inspection.
* **Audio Ownership Check:** Every request to retrieve or delete `/voices/{id}` must verify `voice.user_id == current_user.id`.

---

## 4. AI Assistant Rules of Engagement

1. **Check Existing Artifacts First:** Always inspect `VOICELIB_README.md`, `Architecture.md`, and `Memory.md` before making structural code changes.
2. **No Superficial Symptom Patches:** Never solve an error by swallowing exceptions, deleting failing tests, or returning fake dummy data unless explicit mock mode is enabled.
3. **Preserve API Contracts:** Do not modify API endpoint request/response signatures without updating the corresponding frontend Axios client calls.
4. **Run Verification Commands:** Never declare a feature or bug fix complete without verifying with appropriate test execution or local dev build commands (`npm run build`, pytest, or backend API test calls).
5. **Keep `Memory.md` Updated:** When completing a significant implementation step or phase, update `Memory.md` with the new state, added files, and remaining tasks.

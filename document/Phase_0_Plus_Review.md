# Phase 0+ Review & Implementation Record

> **Phase:** Phase 0+ (Automated Multi-Metric Evaluation & Async Pipeline)  
> **Status:** ✅ Complete & Verified  
> **Repository:** IRIS VoiceLib (`AryanAwchare/IRIS-TTS`)  
> **Date:** September 2026  

---

## 1. Overview & Objectives

Phase 0+ integrates an **automated, non-blocking multi-metric objective evaluation pipeline** into the IRIS VoiceLib synthesis lifecycle.

### Key Goals Achieved:
1. **Decoupled Synthesis & Evaluation**: Text-to-speech returns audio immediately (~1.5–2.5s) with `eval_status: "pending"`.
2. **Asynchronous Background Worker**: A dedicated background worker executes SpeechBrain ECAPA-TDNN, faster-Whisper ASR, and $F_0$ pitch dynamics analysis in a threadpool without blocking the server event loop.
3. **Persistent Evaluation Schema**: Metrics, status, and grades are saved directly into the PostgreSQL / SQLite `generations` table.
4. **Intuitive Quality Grade (A–D)**: Weighted composite quality score combining speaker similarity (50%), word accuracy (35%), and prosodic variation (15%).
5. **Real-Time Polling Endpoints & UI**: `GET /generations/{id}/eval` endpoint and frontend history badges with auto-polling.

---

## 2. File Change Inventory

| File Path | Action | Description |
|---|---|---|
| [`voicelib/backend/app/models.py`](file:///c:/Users/dell/OneDrive/Desktop/IRIS/voicelib/backend/app/models.py) | **Modified** | Added eval columns to `Generation` ORM, updated `GenerationOut`, added `GenerationEvalOut`. |
| [`voicelib/backend/app/db.py`](file:///c:/Users/dell/OneDrive/Desktop/IRIS/voicelib/backend/app/db.py) | **Modified** | Added auto-migration SQL DDL for PostgreSQL & SQLite in `create_tables()`. |
| [`voicelib/backend/app/evaluation/prosody_metric.py`](file:///c:/Users/dell/OneDrive/Desktop/IRIS/voicelib/backend/app/evaluation/prosody_metric.py) | **Created** | $F_0$ standard deviation, mean, and voiced frame ratio extraction. |
| [`voicelib/backend/app/evaluation/grade.py`](file:///c:/Users/dell/OneDrive/Desktop/IRIS/voicelib/backend/app/evaluation/grade.py) | **Created** | Composite quality grade calculation (A, B, C, D) and scoring. |
| [`voicelib/backend/app/evaluation/eval_pipeline.py`](file:///c:/Users/dell/OneDrive/Desktop/IRIS/voicelib/backend/app/evaluation/eval_pipeline.py) | **Created** | Async background worker executing evaluation and updating database. |
| [`voicelib/backend/app/evaluation/__init__.py`](file:///c:/Users/dell/OneDrive/Desktop/IRIS/voicelib/backend/app/evaluation/__init__.py) | **Modified** | Exported all evaluation functions. |
| [`voicelib/backend/app/routers/generate_router.py`](file:///c:/Users/dell/OneDrive/Desktop/IRIS/voicelib/backend/app/routers/generate_router.py) | **Modified** | Wired `BackgroundTasks` in `POST /generate`, added `GET /generations/{id}/eval`. |
| [`voicelib/backend/app/main.py`](file:///c:/Users/dell/OneDrive/Desktop/IRIS/voicelib/backend/app/main.py) | **Modified** | Added evaluation model pre-warming in `lifespan` startup. |
| [`voicelib/frontend/src/api/generate.js`](file:///c:/Users/dell/OneDrive/Desktop/IRIS/voicelib/frontend/src/api/generate.js) | **Modified** | Added `getEval(generationId)` method. |
| [`voicelib/frontend/src/pages/Generate.jsx`](file:///c:/Users/dell/OneDrive/Desktop/IRIS/voicelib/frontend/src/pages/Generate.jsx) | **Modified** | Added Grade A–D badges, pending eval indicator, and history auto-polling. |
| [`voicelib/backend/tests/test_evaluation_pipeline.py`](file:///c:/Users/dell/OneDrive/Desktop/IRIS/voicelib/backend/tests/test_evaluation_pipeline.py) | **Created** | Standalone unit test suite for prosody, grading, and schemas. |

---

## 3. Database Schema & Migrations

### `generations` Table Additions:
```sql
ALTER TABLE generations ADD COLUMN IF NOT EXISTS eval_status VARCHAR(32) DEFAULT 'pending';
ALTER TABLE generations ADD COLUMN IF NOT EXISTS speaker_similarity FLOAT;
ALTER TABLE generations ADD COLUMN IF NOT EXISTS word_error_rate FLOAT;
ALTER TABLE generations ADD COLUMN IF NOT EXISTS prosody_f0_std FLOAT;
ALTER TABLE generations ADD COLUMN IF NOT EXISTS composite_grade VARCHAR(4);
ALTER TABLE generations ADD COLUMN IF NOT EXISTS eval_error TEXT;
ALTER TABLE generations ADD COLUMN IF NOT EXISTS evaluated_at TIMESTAMP WITH TIME ZONE;
```

### SQLAlchemy ORM Model:
```python
class Generation(Base):
    __tablename__ = "generations"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    voice_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("voices.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    audio_s3_key: Mapped[str] = mapped_column(String(512), nullable=False)
    engine: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, default="gpt-sovits-v3")
    emotion: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, default="neutral")
    speed: Mapped[Optional[float]] = mapped_column(nullable=True, default=1.0)
    
    # Automated multi-metric objective evaluation fields
    eval_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)  # pending, completed, failed
    speaker_similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)         # ECAPA-TDNN cosine sim [0.0 - 1.0]
    word_error_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)            # faster-whisper WER [0.0 - 1.0+]
    prosody_f0_std: Mapped[Optional[float]] = mapped_column(Float, nullable=True)             # Pitch dynamic range (F0 standard dev in Hz)
    composite_grade: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)          # Composite quality grade: "A", "B", "C", "D"
    eval_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)                    # Error traceback if evaluation failed
    evaluated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
```

---

## 4. Evaluation Engine & Formulas

### A. Speaker Similarity (`speaker_similarity.py`)
* **Model**: SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`).
* **Embedding**: 192-dimensional vector.
* **Metric**: Exact cosine similarity between reference sample and generated output:
  $$\text{SIM}(\mathbf{e}_{\text{ref}}, \mathbf{e}_{\text{gen}}) = \frac{\mathbf{e}_{\text{ref}} \cdot \mathbf{e}_{\text{gen}}}{\|\mathbf{e}_{\text{ref}}\| \|\mathbf{e}_{\text{gen}}\|}$$

### B. Content Accuracy / WER (`content_accuracy.py`)
* **Model**: `faster-whisper` (`small.en` on CPU, `int8` quantization).
* **Metric**: JiWER Word Error Rate against prompt text (case-insensitive, normalized punctuation):
  $$\text{WER} = \frac{S + D + I}{N}$$

### C. Prosody Dynamic Range (`prosody_metric.py`)
* **Method**: `librosa.pyin` with optimized search bounds ($75\text{ Hz} \le F_0 \le 500\text{ Hz}$, $1024$ frame size, $512$ hop size) with autocorrelation fallback.
* **Metric**: Standard deviation of voiced pitch frames ($F_0 \sigma$ in Hz) and voiced ratio.

### D. Composite Quality Grade (`grade.py`)
$$\text{Score} = (0.50 \times \text{SIM}) + (0.35 \times \max(0, 1 - \text{WER})) + (0.15 \times \min(1.0, \frac{F_0\sigma}{35.0}))$$

* **Grade A (Excellent)**: $\text{Score} \ge 0.82$
* **Grade B (Good)**: $0.68 \le \text{Score} < 0.82$
* **Grade C (Fair)**: $0.52 \le \text{Score} < 0.68$
* **Grade D (Poor)**: $\text{Score} < 0.52$

---

## 5. API Endpoints

### 1. `POST /generate`
* **Status**: `201 Created`
* **Response Payload**:
  ```json
  {
    "id": "c1f7a8b4-92d3-4f9e-a81d-e59392b8d001",
    "voice_id": "8f3b2a1c-5d4e-4f6a-9b8c-1e2d3f4a5b6c",
    "input_text": "Hello world, this is a test generation.",
    "audio_url": "https://storage.local/generated/c1f7a8b4.wav",
    "engine": "gpt-sovits-v3",
    "emotion": "neutral",
    "speed": 1.0,
    "eval_status": "pending",
    "speaker_similarity": null,
    "word_error_rate": null,
    "prosody_f0_std": null,
    "composite_grade": null,
    "evaluated_at": null,
    "created_at": "2026-09-01T10:15:00Z"
  }
  ```

### 2. `GET /generations/{generation_id}/eval`
* **Status**: `200 OK`
* **Response Payload**:
  ```json
  {
    "generation_id": "c1f7a8b4-92d3-4f9e-a81d-e59392b8d001",
    "voice_id": "8f3b2a1c-5d4e-4f6a-9b8c-1e2d3f4a5b6c",
    "eval_status": "completed",
    "speaker_similarity": 0.8842,
    "word_error_rate": 0.0417,
    "prosody_f0_std": 38.25,
    "composite_grade": "A",
    "eval_error": null,
    "evaluated_at": "2026-09-01T10:15:05Z",
    "created_at": "2026-09-01T10:15:00Z"
  }
  ```

---

## 6. How to Run & Verify Tests

To execute the automated evaluation test suite:

```powershell
cd voicelib\backend
.\venv\Scripts\python.exe -u tests\test_evaluation_pipeline.py
```

Expected output:
```
Running evaluation pipeline unit tests...
[PASS] test_prosody_metric_synthetic_tone
[PASS] test_prosody_metric_silence_or_missing
[PASS] test_composite_grade_scale
[PASS] test_generation_schemas
>>> ALL EVALUATION PIPELINE TESTS PASSED! <<<
```

---

## 7. Next Roadmap Milestone (Phase 1)

With Phase 0+ operational:
* **Phase 1 Focus**:
  1. Silero-VAD reference sample pre-filtering and SNR window selection on upload.
  2. Deep Acoustic DNA Profiler v2 persistent storage in `voices.opt_weights`.
  3. Reference quality scoring and warning banners for noisy uploads.

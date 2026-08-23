# 📌 VoiceLib — Supabase Connection Quick Guide

> **Saved Setup Instructions for when you return.**  
> Project Ref: `nkzfhsyjgavppgtxatav`  
> URL: `https://nkzfhsyjgavppgtxatav.supabase.co`

---

## 🚀 3 Quick Steps to Connect Supabase

### Step 1: Database Password & Connection String
1. Go to your Supabase Dashboard:  
   👉 [https://supabase.com/dashboard/project/nkzfhsyjgavppgtxatav/settings/database](https://supabase.com/dashboard/project/nkzfhsyjgavppgtxatav/settings/database)
2. Scroll to **Connection String** → Select **URI** (or **Session Pooler**).
3. Copy the URI string and replace `YOUR_PASSWORD` with your Supabase database password:
   ```text
   postgresql+asyncpg://postgres.nkzfhsyjgavppgtxatav:YOUR_PASSWORD@aws-0-us-east-1.pooler.supabase.com:6543/postgres
   ```

---

### Step 2: Create Public Storage Bucket
1. Go to **Storage** 📁 → **New Bucket**:  
   👉 [https://supabase.com/dashboard/project/nkzfhsyjgavppgtxatav/storage/buckets](https://supabase.com/dashboard/project/nkzfhsyjgavppgtxatav/storage/buckets)
2. Bucket Name: `voicelib`
3. Toggle **Public Bucket** to **ON** ✅.

---

### Step 3: Generate S3 Storage Access Keys
1. Go to **Project Settings** ⚙️ → **Storage**:  
   👉 [https://supabase.com/dashboard/project/nkzfhsyjgavppgtxatav/settings/storage](https://supabase.com/dashboard/project/nkzfhsyjgavppgtxatav/settings/storage)
2. Click **Generate S3 Access Key**.
3. Copy your **Access Key ID** and **Secret Access Key**.

---

## 📋 Exact `.env` File Template

Paste this into `voicelib/.env` when you return:

```env
# ── 1. Database Connection (Supabase PostgreSQL) ─────────────────────────
DATABASE_URL=postgresql+asyncpg://postgres.nkzfhsyjgavppgtxatav:YOUR_PASSWORD_HERE@aws-0-us-east-1.pooler.supabase.com:6543/postgres

# ── 2. Storage Connection (Supabase S3 API) ──────────────────────────────
STORAGE_ENDPOINT_URL=https://nkzfhsyjgavppgtxatav.supabase.co/storage/v1/s3
STORAGE_ACCESS_KEY=PASTE_YOUR_SUPABASE_S3_ACCESS_KEY_HERE
STORAGE_SECRET_KEY=PASTE_YOUR_SUPABASE_S3_SECRET_KEY_HERE
STORAGE_BUCKET_NAME=voicelib
STORAGE_REGION=global
STORAGE_PUBLIC_BASE_URL=https://nkzfhsyjgavppgtxatav.supabase.co/storage/v1/object/public/voicelib

# ── 3. Security & App Settings ───────────────────────────────────────────
JWT_SECRET_KEY=voicelib-localhost-secret-key-32chars
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# ── 4. Dev Settings ──────────────────────────────────────────────────────
VOICELIB_USE_MOCK_TTS=true
MAX_UPLOAD_SIZE_BYTES=20971520
MAX_SAMPLE_DURATION_SECONDS=30
ALLOWED_AUDIO_FORMATS=["audio/wav","audio/mpeg","audio/ogg","audio/flac","audio/x-wav","audio/wave"]

# ── 5. Frontend ──────────────────────────────────────────────────────────
VITE_API_BASE_URL=http://localhost:8000
```

---

## ⚡ Quick Launch Command When You Return

```bash
# Terminal 1 — Backend
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt aiosqlite
uvicorn app.main:app --reload

# Terminal 2 — Frontend
cd frontend
npm run dev
```

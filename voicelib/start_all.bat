@echo off
title VoiceLib Launcher
echo =======================================================
echo          Starting VoiceLib (Backend + Frontend)
echo =======================================================
echo.
echo [1/2] Launching Backend on http://localhost:8000 ...
start "VoiceLib Backend" cmd /k "cd /d %~dp0backend && .\venv\Scripts\activate && uvicorn app.main:app --reload --port 8000"

echo [2/2] Launching Frontend on http://localhost:5173 ...
start "VoiceLib Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo =======================================================
echo VoiceLib is running!
echo Frontend: http://localhost:5173
echo Backend API Docs: http://localhost:8000/docs
echo =======================================================
pause

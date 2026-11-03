@echo off
REM ==============================================================
REM Overnight run: extends the experiment to the full GSM8K test
REM set (1319 questions). Same settings as before. Resumes
REM automatically, so the first 200 questions are NOT repeated.
REM Double-click this file, or run:  run_overnight.bat
REM (works from whatever folder the file is in)
REM ==============================================================
cd /d "%~dp0"

echo [1/2] Main agents + bad agent on 1319 questions (about 7 hours)...
python collect_answers.py 1319

echo [2/2] Extra unreliable agents on 1319 questions (about 1.5 hours)...
python stress_collect.py 1319

echo.
echo ===== ALL DONE =====
echo %date% %time%
pause

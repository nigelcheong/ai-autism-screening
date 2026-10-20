@echo off
REM Starts the backend, frontend, and a public tunnel, each in its own
REM visible window. Close a window to stop that piece; close all three
REM (or run stop_all.bat) to shut everything down.

echo Starting backend...
start "Autism Screening - Backend" cmd /k "cd /d C:\Work\Github\Autism-screening\backend && venv\Scripts\uvicorn.exe app.main:app --port 8010 --reload"

timeout /t 6 /nobreak >nul

echo Starting frontend...
start "Autism Screening - Frontend" cmd /k "cd /d C:\Work\Github\Autism-screening\frontend && npm run dev"

timeout /t 8 /nobreak >nul

echo Starting public tunnel...
start "Autism Screening - Tunnel" cmd /k "cd /d C:\Work\Github\Autism-screening && tunnel_loop.bat"

echo.
echo All three started in separate windows.
echo Your team's URL: https://autism-screening.serveousercontent.com
echo (give the tunnel window a few seconds to connect before sharing)
echo.
pause

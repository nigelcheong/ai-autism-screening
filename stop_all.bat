@echo off
REM Stops the backend, frontend, and tunnel. Note: this kills ALL
REM uvicorn/node/ssh processes system-wide, not just these three --
REM close the individual windows instead if you have other node/ssh
REM processes running that you want to keep.

taskkill /F /IM uvicorn.exe /T 2>nul
taskkill /F /IM node.exe /T 2>nul
taskkill /F /IM ssh.exe /T 2>nul

echo Stopped.
pause

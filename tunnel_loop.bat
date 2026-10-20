@echo off
REM serveo.net briefly refuses to rebind a reserved name right after a
REM disconnect (a server-side cooldown, not a bug here) -- so this just
REM keeps retrying every few seconds until it takes.
:retry
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -R autism-screening:80:localhost:5173 serveo.net
echo.
echo Tunnel disconnected or failed to bind -- retrying shortly...
ping -n 6 127.0.0.1 >nul
goto retry

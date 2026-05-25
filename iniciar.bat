@echo off
title VigiFacial AI — Microservicio IA (Puerto 8001)
color 0A
echo.
echo  ===================================================
echo   VigiFacial — Microservicio de Reconocimiento Facial
echo   Puerto: 8001  |  Docs: http://localhost:8001/docs
echo  ===================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Entorno virtual no encontrado.
    echo         Ejecuta primero: instalar.bat
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
echo [*] Iniciando servidor FastAPI...
echo.
venv\Scripts\uvicorn.exe main:app --host 0.0.0.0 --port 8001 --reload --log-level info
pause

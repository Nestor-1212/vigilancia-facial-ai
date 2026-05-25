@echo off
title VigiFacial AI — Instalacion
color 0A
echo.
echo  ===================================================
echo   VigiFacial — Microservicio de Reconocimiento Facial
echo   Instalador automatico para Windows
echo  ===================================================
echo.

:: ── 1. Verificar Python ──────────────────────────────────────────
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python no encontrado. Intentando instalar con winget...
    winget install Python.Python.3.11 --accept-package-agreements --accept-source-agreements
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] No se pudo instalar Python automaticamente.
        echo         Por favor instala Python 3.11 manualmente desde:
        echo         https://www.python.org/downloads/release/python-3119/
        echo         IMPORTANTE: marca "Add Python to PATH" durante la instalacion.
        echo.
        pause
        exit /b 1
    )
    :: Recargar PATH
    call refreshenv 2>nul
)

python --version
echo.

:: ── 2. Crear entorno virtual ─────────────────────────────────────
if not exist "venv" (
    echo [*] Creando entorno virtual...
    python -m venv venv
)

echo [*] Activando entorno virtual...
call venv\Scripts\activate.bat

:: ── 3. Actualizar pip ────────────────────────────────────────────
echo [*] Actualizando pip...
python -m pip install --upgrade pip --quiet

:: ── 4. Instalar dependencias ─────────────────────────────────────
echo.
echo [*] Instalando dependencias (esto puede tomar varios minutos)...
echo     TensorFlow + DeepFace son pesados (~2 GB), por favor espera.
echo.
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Fallo la instalacion de dependencias.
    echo         Intenta ejecutar manualmente: pip install -r requirements.txt
    pause
    exit /b 1
)

:: ── 5. Crear carpetas necesarias ─────────────────────────────────
if not exist "faces_db" mkdir faces_db
if not exist "capturas" mkdir capturas

echo.
echo  ===================================================
echo   [OK] Instalacion completada exitosamente!
echo.
echo   Para iniciar el microservicio ejecuta:
echo     iniciar.bat
echo   O manualmente:
echo     venv\Scripts\activate
echo     uvicorn main:app --host 0.0.0.0 --port 8001 --reload
echo.
echo   Documentacion API: http://localhost:8001/docs
echo  ===================================================
echo.
pause

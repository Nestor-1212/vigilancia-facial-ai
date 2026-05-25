@echo off
title VigiFacial — Sincronizar personas con Laravel
color 0B
echo.
echo [*] Sincronizando fotos de personas desde Laravel...
echo     Asegurate de que Laravel este corriendo en http://localhost:8000
echo     y que LARAVEL_SERVICE_TOKEN este configurado en .env
echo.

call venv\Scripts\activate.bat

:: Llama al endpoint de sincronizacion
curl -X POST http://localhost:8001/sincronizar-personas ^
     -H "Content-Type: application/json"

echo.
echo [OK] Sincronizacion completada.
pause

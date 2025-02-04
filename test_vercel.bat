@echo off
setlocal enabledelayedexpansion

echo === Iniciando pruebas de configuración para Vercel ===
echo.

:: Verificar Python y pip
echo Verificando versión de Python...
python --version
if !errorlevel! neq 0 (
    echo Error: Python no está instalado o no está en el PATH
    pause
    exit /b 1
)

:: Verificar variables de entorno
echo Verificando variables de entorno...
if not defined FLASK_ENV (
    set FLASK_ENV=development
    echo FLASK_ENV no estaba definido, establecido a development
)

:: Verificar estructura de directorios
echo Verificando estructura de directorios...
if not exist "backend\models" (
    mkdir "backend\models"
    echo Creado directorio backend\models
)
if not exist "backend\api" (
    mkdir "backend\api"
    echo Creado directorio backend\api
)

:: Verificar modelo
echo Verificando modelo de ML...
if not exist "backend\models\stroke_prediction_model.joblib" (
    echo ADVERTENCIA: Modelo no encontrado en backend\models\stroke_prediction_model.joblib
    echo Asegúrate de tener el modelo entrenado antes del despliegue
    pause
    exit /b 1
)

:: Construir frontend
echo.
echo === Construyendo frontend ===
cd frontend
if not exist "node_modules" (
    echo Instalando dependencias del frontend...
    call npm install --legacy-peer-deps
    if !errorlevel! neq 0 (
        echo Error instalando dependencias del frontend
        cd ..
        pause
        exit /b 1
    )
)

echo Construyendo aplicación...
set "CI=false"
call npm run build
if !errorlevel! neq 0 (
    echo Error en la construcción del frontend
    cd ..
    pause
    exit /b 1
)
cd ..

:: Configurar backend
echo.
echo === Configurando backend ===
cd backend

:: Verificar requirements.txt
if not exist "requirements.txt" (
    echo ERROR: No se encuentra el archivo requirements.txt
    cd ..
    pause
    exit /b 1
)

echo Creando y activando entorno virtual...
if exist "venv" (
    echo Eliminando entorno virtual anterior...
    deactivate 2>nul
    rmdir /s /q venv
)

python -m venv venv
call venv\Scripts\activate

echo Actualizando pip y herramientas base...
python -m pip install --upgrade pip setuptools wheel

:: Configurar pip para usar wheels precompilados
set "PIP_NO_CACHE_DIR=off"
set "PIP_PREFER_BINARY=1"

echo Instalando dependencias...
python -m pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo Error instalando dependencias
    cd ..
    pause
    exit /b 1
)

:: Probar API
echo.
echo === Probando API ===
echo Iniciando servidor en segundo plano...
start /B python api/index.py

:: Esperar a que el servidor inicie y verificar que esté funcionando
echo Esperando a que el servidor inicie...
set /a attempts=0
:check_server
ping -n 2 127.0.0.1 > nul
curl -s http://localhost:5000/api/health > nul
if !errorlevel! equ 0 (
    echo Servidor iniciado correctamente
    goto server_ready
)
set /a attempts+=1
if !attempts! lss 10 (
    echo Intento !attempts! de 10...
    goto check_server
) else (
    echo Error: El servidor no pudo iniciarse después de 10 intentos
    goto cleanup
)

:server_ready
:: Realizar pruebas
echo.
echo Probando endpoints...
echo 1. Health check
curl -X GET http://localhost:5000/api/health
echo.

echo 2. Prueba de predicción
curl -X POST http://localhost:5000/api/predict ^
-H "Content-Type: application/json" ^
-d "{\"gender\": \"Male\", \"age\": 65, \"hypertension\": 1, \"heart_disease\": 1, \"ever_married\": \"Yes\", \"work_type\": \"Private\", \"Residence_type\": \"Urban\", \"avg_glucose_level\": 169.5, \"bmi\": 35.5, \"smoking_status\": \"formerly smoked\"}"
echo.

:: Verificar archivos de Vercel
echo.
echo === Verificando configuración de Vercel ===
if not exist "..\vercel.json" (
    echo ERROR: No se encuentra vercel.json
    cd ..
    pause
    exit /b 1
)

:cleanup
:: Detener servidor
echo.
echo === Limpieza ===
taskkill /F /IM python.exe /FI "WINDOWTITLE eq test_vercel" 2>nul
if exist "venv" (
    deactivate
)

echo.
echo === Pruebas completadas ===
echo Recuerda:
echo 1. Verificar que el modelo esté incluido en el despliegue
echo 2. Configurar las variables de entorno en Vercel
echo 3. Asegurarte de que los endpoints estén correctamente configurados
echo.

pause 
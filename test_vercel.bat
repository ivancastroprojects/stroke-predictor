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

:: Verificar que la versión de Python sea compatible
for /f "tokens=2" %%a in ('python --version 2^>^&1') do (
    set PYTHON_VERSION=%%a
)
echo Python version: %PYTHON_VERSION%

:: Extraer versión mayor y menor
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

if !PYTHON_MAJOR! lss 3 (
    echo Error: Esta aplicación requiere Python 3 o superior
    pause
    exit /b 1
)

if !PYTHON_MINOR! gtr 13 (
    echo Error: Esta aplicación requiere Python 3.13 o inferior
    pause
    exit /b 1
)

:: Verificar Node.js y npm
echo Verificando versión de Node.js...
node --version
if !errorlevel! neq 0 (
    echo Error: Node.js no está instalado o no está en el PATH
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
set CI=false
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
    rmdir /s /q venv
)

python -m venv venv
call venv\Scripts\activate.bat

echo Actualizando pip y herramientas base...
python -m pip install --upgrade pip setuptools wheel

:: Configurar pip para usar wheels precompilados
set PIP_NO_CACHE_DIR=off
set PIP_PREFER_BINARY=1

echo Instalando dependencias...
python -m pip install --no-cache-dir -r requirements.txt
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

:: Intentar matar cualquier instancia previa del servidor
taskkill /F /IM python.exe /FI "WINDOWTITLE eq test_vercel" 2>nul

:: Iniciar el servidor con un título específico
start "test_vercel" /B python api/index.py

:: Esperar a que el servidor inicie (usando sintaxis correcta de timeout)
echo Esperando a que el servidor inicie...
choice /C Y /N /T 10 /D Y /M "Esperando 10 segundos" > nul

:: Probar endpoints con reintentos y mejor manejo de errores
echo.
echo Probando endpoints...
echo 1. Health check
set "health_success=0"
for /l %%i in (1,1,3) do (
    if !health_success! equ 0 (
        curl -s -X GET http://localhost:5000/api/health > nul
        if !errorlevel! equ 0 (
            echo Health check exitoso
            set "health_success=1"
            curl -X GET http://localhost:5000/api/health
        ) else (
            echo Intento %%i fallido, reintentando...
            choice /C Y /N /T 2 /D Y > nul
        )
    )
)

if !health_success! equ 0 (
    echo ERROR: No se pudo conectar al servidor después de 3 intentos
    goto :cleanup
)

echo.
echo 2. Prueba de predicción
set "predict_success=0"
for /l %%i in (1,1,3) do (
    if !predict_success! equ 0 (
        curl -s -X POST http://localhost:5000/api/predict ^
        -H "Content-Type: application/json" ^
        -d "{\"gender\": \"Male\", \"age\": 65, \"hypertension\": 1, \"heart_disease\": 1, \"ever_married\": \"Yes\", \"work_type\": \"Private\", \"Residence_type\": \"Urban\", \"avg_glucose_level\": 169.5, \"bmi\": 35.5, \"smoking_status\": \"formerly smoked\"}" > nul
        if !errorlevel! equ 0 (
            echo Prueba de predicción exitosa
            set "predict_success=1"
            curl -X POST http://localhost:5000/api/predict ^
            -H "Content-Type: application/json" ^
            -d "{\"gender\": \"Male\", \"age\": 65, \"hypertension\": 1, \"heart_disease\": 1, \"ever_married\": \"Yes\", \"work_type\": \"Private\", \"Residence_type\": \"Urban\", \"avg_glucose_level\": 169.5, \"bmi\": 35.5, \"smoking_status\": \"formerly smoked\"}"
        ) else (
            echo Intento %%i fallido, reintentando...
            choice /C Y /N /T 2 /D Y > nul
        )
    )
)

if !predict_success! equ 0 (
    echo ERROR: No se pudo realizar la predicción después de 3 intentos
    goto :cleanup
)

:cleanup
:: Detener servidor y limpiar
echo.
echo === Limpieza ===
taskkill /F /IM python.exe /FI "WINDOWTITLE eq test_vercel" 2>nul
if exist "venv" (
    call venv\Scripts\deactivate.bat 2>nul
)

echo.
echo === Pruebas completadas ===
echo Recuerda:
echo 1. Verificar que el modelo esté incluido en el despliegue
echo 2. Configurar las variables de entorno en Vercel:
echo    - FLASK_ENV=production
echo    - PYTHONPATH=/var/task/backend
echo 3. Asegurarte de que los endpoints estén correctamente configurados
echo.

pause
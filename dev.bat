@echo off
echo [%date% %time%] Iniciando script de desarrollo > dev.log

echo [%date% %time%] === Simulando Vercel Build === >> dev.log
echo [%date% %time%] Instalando dependencias del frontend... >> dev.log
cd frontend
echo [%date% %time%] Directorio actual: %cd% >> ..\dev.log
call npm install >> ..\dev.log 2>&1
if %errorlevel% neq 0 (
    echo [%date% %time%] Error instalando dependencias del frontend >> ..\dev.log
    exit /b %errorlevel%
)

echo [%date% %time%] === Construyendo el frontend === >> ..\dev.log
set "CI=false"
call npm run build >> ..\dev.log 2>&1
if %errorlevel% neq 0 (
    echo [%date% %time%] Error construyendo el frontend >> ..\dev.log
    exit /b %errorlevel%
)

echo [%date% %time%] === Configurando el backend === >> ..\dev.log
cd ..\backend
echo [%date% %time%] Directorio actual: %cd% >> ..\dev.log

echo [%date% %time%] Creando entorno virtual... >> ..\dev.log
python -m venv venv >> ..\dev.log 2>&1
if %errorlevel% neq 0 (
    echo [%date% %time%] Error creando entorno virtual >> ..\dev.log
    exit /b %errorlevel%
)

echo [%date% %time%] Activando entorno virtual... >> ..\dev.log
call venv\Scripts\activate >> ..\dev.log 2>&1
if %errorlevel% neq 0 (
    echo [%date% %time%] Error activando entorno virtual >> ..\dev.log
    exit /b %errorlevel%
)

echo [%date% %time%] Actualizando pip... >> ..\dev.log
python -m pip install --upgrade pip >> ..\dev.log 2>&1

echo [%date% %time%] Instalando dependencias del backend... >> ..\dev.log
pip install -r requirements.txt >> ..\dev.log 2>&1
if %errorlevel% neq 0 (
    echo [%date% %time%] Error instalando dependencias del backend >> ..\dev.log
    exit /b %errorlevel%
)

echo [%date% %time%] === Iniciando servicios === >> ..\dev.log
echo [%date% %time%] Iniciando servidor frontend... >> ..\dev.log
start cmd /k "cd ../frontend && echo [%date% %time%] Servidor frontend iniciado >> ..\dev.log && npx serve -s build -l 3000 >> ..\dev.log 2>&1"

echo [%date% %time%] Iniciando servidor backend... >> ..\dev.log
start cmd /k "cd ../backend && echo [%date% %time%] Servidor backend iniciado >> ..\dev.log && python app.py >> ..\dev.log 2>&1"

echo [%date% %time%] === Servicios iniciados === >> dev.log
echo [%date% %time%] Frontend: http://localhost:3000 >> dev.log
echo [%date% %time%] Backend: http://localhost:8000/api >> dev.log
echo [%date% %time%] Logs disponibles en dev.log >> dev.log

echo === Servicios iniciados ===
echo Frontend: http://localhost:3000
echo Backend: http://localhost:8000/api
echo Logs disponibles en dev.log
echo Presiona Ctrl+C para detener los servicios 
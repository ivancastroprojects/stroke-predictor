@echo off
echo 🚀 Iniciando instalación de Stroke Prediction Web...

:: Crear y activar entorno virtual
echo 📦 Creando entorno virtual...
python -m venv .venv
call .venv\Scripts\activate.bat

:: Instalar dependencias del backend
echo 🔧 Instalando dependencias del backend...
cd backend
pip install -r requirements.txt

:: Instalar dependencias del frontend
echo 🎨 Instalando dependencias del frontend...
cd ..\frontend
npm install

:: Iniciar servicios
echo 🌟 Iniciando servicios...
:: Iniciar backend en una nueva ventana
start cmd /k "cd ..\backend && ..\\.venv\Scripts\python.exe app.py"

:: Esperar a que el backend esté listo
timeout /t 5

:: Iniciar frontend
npm start 
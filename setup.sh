#!/bin/bash

echo "🚀 Iniciando instalación de Stroke Prediction Web..."

# Crear y activar entorno virtual
echo "📦 Creando entorno virtual..."
python -m venv .venv
source .venv/bin/activate

# Instalar dependencias del backend
echo "🔧 Instalando dependencias del backend..."
cd backend
pip install -r requirements.txt

# Instalar dependencias del frontend
echo "🎨 Instalando dependencias del frontend..."
cd ../frontend
npm install

# Iniciar servicios
echo "🌟 Iniciando servicios..."
# Iniciar backend en segundo plano
cd ../backend
python app.py &

# Esperar a que el backend esté listo
sleep 5

# Iniciar frontend
cd ../frontend
npm start 
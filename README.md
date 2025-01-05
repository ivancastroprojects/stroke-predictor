# Stroke Prediction Web

Una aplicación web para predecir el riesgo de accidentes cerebrovasculares utilizando machine learning.

## Instalación Rápida

### Windows
```bash
# Ejecutar el script de instalación
setup.bat
```

### Linux/Mac
```bash
# Dar permisos de ejecución al script
chmod +x setup.sh

# Ejecutar el script de instalación
./setup.sh
```

## Requisitos Previos
- Python 3.8 o superior
- Node.js 14 o superior
- npm 6 o superior

## Instalación Manual

1. Crear entorno virtual:
```bash
python -m venv .venv
```

2. Activar entorno virtual:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. Instalar dependencias del backend:
```bash
cd backend
pip install -r requirements.txt
```

4. Instalar dependencias del frontend:
```bash
cd frontend
npm install
```

5. Iniciar servicios:
```bash
# Terminal 1 - Backend
cd backend
python app.py

# Terminal 2 - Frontend
cd frontend
npm start
```

## Características
- Predicción de riesgo de stroke basada en múltiples factores
- Visualización de factores de riesgo
- Historial de predicciones
- Interfaz intuitiva y responsive
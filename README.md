# 🧠 Stroke Risk Prediction Web App

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0.1-green.svg)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/React-18.2.0-blue.svg)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Una aplicación web moderna que utiliza Machine Learning para predecir el riesgo de accidentes cerebrovasculares, proporcionando insights médicos personalizados y recomendaciones de salud.

![Stroke Risk Assessment Tool](https://img.freepik.com/free-photo/male-medical-figure-with-front-brain-highlighted_1048-11823.jpg)

## 🌟 Características Principales

- **Predicción de Riesgo en Tiempo Real**: Análisis instantáneo basado en múltiples factores de salud
- **Visualización Interactiva**: Gráficos detallados de factores contribuyentes y niveles de riesgo
- **Recomendaciones Personalizadas**: Sugerencias médicas basadas en el perfil individual
- **Historial de Predicciones**: Seguimiento temporal de evaluaciones previas
- **Interfaz Médica Profesional**: Diseño moderno orientado al sector salud
- **Responsive Design**: Adaptable a cualquier dispositivo

## 🚀 Tecnologías Utilizadas

### Frontend
- React 18
- Recharts para visualizaciones
- CSS Moderno con diseño responsivo
- Animaciones fluidas y transiciones suaves

### Backend
- Flask (Python)
- Scikit-learn para ML
- NumPy y Pandas para procesamiento de datos
- Joblib para serialización del modelo

### ML/AI
- Modelo de predicción entrenado con datos médicos
- Análisis de factores de riesgo en tiempo real
- Cálculo de contribuciones de características

## 🛠️ Instalación

### Requisitos Previos
- Python 3.8+
- Node.js 14+
- npm 6+

### Instalación Rápida

```bash
# Windows
setup.bat

# Linux/Mac
chmod +x setup.sh
./setup.sh
```

### Instalación Manual

1. **Configurar Backend**
```bash
# Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Instalar dependencias
cd backend
pip install -r requirements.txt
```

2. **Configurar Frontend**
```bash
cd frontend
npm install
```

3. **Iniciar Servicios**
```bash
# Terminal 1 - Backend
cd backend
python app.py

# Terminal 2 - Frontend
cd frontend
npm start
```

## 📊 Características Técnicas

- **API RESTful**: Endpoints optimizados para predicciones y análisis
- **Validación de Datos**: Sistema robusto de validación de inputs
- **Carga Diferida**: Modelo ML cargado bajo demanda
- **CORS Configurado**: Seguridad para peticiones cross-origin
- **Manejo de Errores**: Sistema comprensivo de logging y debugging
- **Caché Inteligente**: Almacenamiento local de predicciones

## 🔒 Seguridad

- Validación robusta de datos de entrada
- Sanitización de parámetros
- Headers de seguridad configurados
- Protección contra ataques comunes

## 📱 Responsive Design

- Mobile-first approach
- Breakpoints optimizados
- UI/UX adaptativa
- Rendimiento optimizado

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor, lee las guías de contribución antes de enviar un PR.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## ⚡ Rendimiento

- Tiempo de respuesta API < 200ms
- Lighthouse Score > 90
- Web Vitals optimizados
- Carga progresiva de assets

## 🔗 Enlaces Útiles

- [Documentación API](docs/api.md)
- [Guía de Desarrollo](docs/development.md)
- [Changelog](CHANGELOG.md)

---
Desarrollado con ❤️ para la comunidad médica y de ML
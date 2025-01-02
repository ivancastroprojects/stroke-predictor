# 🧠 Stroke Prediction App

Una aplicación web end-to-end que utiliza machine learning para predecir el riesgo de accidentes cerebrovasculares basándose en factores de salud del paciente.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![React](https://img.shields.io/badge/React-18.2.0-61DAFB.svg)
![Flask](https://img.shields.io/badge/Flask-2.0.1-000000.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-F7931E.svg)

## 🚀 Características

- **Frontend React moderno** con diseño responsive y animaciones fluidas
- **API REST Flask** para procesamiento de predicciones
- **Modelo ML** entrenado con Linear Discriminant Analysis
- **Interfaz intuitiva** para introducción de datos del paciente
- **Validación de datos** en tiempo real
- **Diseño atractivo** con gradientes y efectos visuales modernos

## 🛠️ Tecnologías

### Frontend
- React 18
- Axios para peticiones HTTP
- CSS moderno con animaciones
- Diseño responsive

### Backend
- Flask
- Flask-CORS
- scikit-learn
- pandas
- numpy
- joblib

## 📊 Factores de Predicción

El modelo analiza los siguientes factores de riesgo:
- Género
- Edad
- Hipertensión
- Enfermedades cardíacas
- Estado civil
- Tipo de trabajo
- Tipo de residencia
- Nivel medio de glucosa
- IMC (Índice de Masa Corporal)
- Estado de fumador

## 🚀 Instalación

### Backend
bash
Crear entorno virtual
python -m venv venv
Activar entorno virtual
Windows
venv\Scripts\activate
Linux/Mac
source venv/bin/activate
Instalar dependencias
pip install -r backend/requirements.txt
Iniciar servidor
cd backend
python app.py

### Frontend
bash
Instalar dependencias
cd frontend
npm install
Iniciar aplicación
npm start

## 🌐 Uso

1. Inicia ambos servidores (backend en puerto 5000, frontend en puerto 3000)
2. Accede a `http://localhost:3000`
3. Completa el formulario con los datos del paciente
4. Haz clic en "Predict Stroke" para obtener la predicción

## 🧪 Modelo ML

- Algoritmo: Linear Discriminant Analysis (LDA)
- Preprocesamiento: Power Transformer y SMOTE para balance de clases
- Validación: Repeated Stratified K-Fold Cross Validation
- Métricas: ROC AUC Score

## 📁 Estructura del Proyecto
stroke-predictor/
├── backend/
│ ├── app.py # Servidor Flask
│ ├── training.py # Entrenamiento del modelo
│ └── requirements.txt # Dependencias Python
└── frontend/
├── public/
└── src/
├── App.js # Componente principal
└── App.css # Estilos

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Distribuido bajo la licencia MIT. Ver `LICENSE` para más información.

## 👥 Autor

Ivan Castro - [GitHub](https://github.com/ivancastroprojects)
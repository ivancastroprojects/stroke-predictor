import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import axios from 'axios';
import "./App.css";

// Componentes
import RiskFactors from './components/RiskFactors';
import InfoSection from './components/InfoSection';
import PredictionForm from './components/PredictionForm';
import PredictionResults from './components/PredictionResults';
import PredictionHistory from './components/PredictionHistory';
import BackgroundRays from './components/BackgroundRays';

// Predicciones de ejemplo
const examplePredictions = [
  {
    gender: "Male",
    age: "70",
    hypertension: "1",
    heart_disease: "1",
    ever_married: "Yes",
    work_type: "Private",
    Residence_type: "Urban",
    avg_glucose_level: "150",
    bmi: "30",
    smoking_status: "formerly smoked",
    prediction: 1,
    timestamp: new Date(2025, 0, 5, 1, 28, 1) // 5/1/2025, 1:28:01
  },
  {
    gender: "Female",
    age: "45",
    hypertension: "0",
    heart_disease: "0",
    ever_married: "No",
    work_type: "Self-employed",
    Residence_type: "Rural",
    avg_glucose_level: "90",
    bmi: "22",
    smoking_status: "never smoked",
    prediction: 0,
    timestamp: new Date(2025, 0, 5, 1, 28, 1) // 5/1/2025, 1:28:01
  }
];

function AppContent() {
  const [history, setHistory] = useState([]);
  const navigate = useNavigate();

  // Cargar predicciones de ejemplo al iniciar
  useEffect(() => {
    setHistory(examplePredictions);
  }, []);

  const handlePrediction = async (formData) => {
    try {
      const response = await axios.post('http://localhost:5000/predict', formData);
      const prediction = response.data.prediction;
      
      setHistory(prev => [...prev, { 
        ...formData, 
        prediction, 
        timestamp: new Date() 
      }]);
      
      navigate('/results', { 
        state: { 
          prediction, 
          formData 
        } 
      });
      
    } catch (error) {
      console.error('Error:', error);
      alert('Error al procesar la predicción. Por favor, intente nuevamente.');
    }
  };

  return (
    <>
      <BackgroundRays />
      <div className="app-container">
        <div className="risk-factors-panel">
          <RiskFactors />
        </div>

        <div className="main-content">
          <Routes>
            <Route path="/" element={
              <>
                <InfoSection />
                <PredictionForm onSubmit={handlePrediction} />
              </>
            } />
            <Route path="/results" element={<PredictionResults />} />
          </Routes>
        </div>

        <div className="history-panel">
          <PredictionHistory history={history} />
        </div>
      </div>
    </>
  );
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;

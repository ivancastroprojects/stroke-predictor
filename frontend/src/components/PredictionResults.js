import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

export default function PredictionResults() {
  const location = useLocation();
  const navigate = useNavigate();
  const { prediction, formData } = location.state || {};
  
  const riskLevel = prediction > 0.7 ? 'Alto' : prediction > 0.3 ? 'Moderado' : 'Bajo';
  const riskColor = {
    'Alto': 'red',
    'Moderado': 'yellow',
    'Bajo': 'green'
  }[riskLevel];

  const handleBack = () => {
    navigate('/');
  };

  return (
    <div className="results-container">
      <div className="results-header">
        <h2>Resultados del Análisis</h2>
        <div className="risk-indicator" style={{ backgroundColor: riskColor }}>
          Riesgo {riskLevel} ({(prediction * 100).toFixed(1)}%)
        </div>
      </div>

      <div className="results-details">
        <div className="results-section">
          <h3>Factores Principales</h3>
          <ul>
            {formData.hypertension === "1" && (
              <li>Hipertensión detectada - Factor de riesgo importante</li>
            )}
            {formData.heart_disease === "1" && (
              <li>Enfermedad cardíaca presente - Aumenta el riesgo</li>
            )}
            {parseInt(formData.age) > 60 && (
              <li>Edad superior a 60 años - Factor de riesgo por edad</li>
            )}
          </ul>
        </div>

        <div className="results-section">
          <h3>Recomendaciones</h3>
          <ul>
            <li>Consulte con su médico regularmente</li>
            <li>Mantenga un registro de su presión arterial</li>
            <li>Siga una dieta saludable</li>
            <li>Realice actividad física regular</li>
          </ul>
        </div>
      </div>

      <button onClick={handleBack} className="back-button">Regresar</button>
    </div>
  );
} 
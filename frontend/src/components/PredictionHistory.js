import React from 'react';
import './PredictionHistory.css';

const examplePredictions = [
  {
    age: 45,
    gender: 'Male',
    hypertension: '1',
    heart_disease: '0',
    avg_glucose_level: '120',
    bmi: '28.5',
    prediction: 0.35,
    timestamp: new Date('2024-01-06T15:30:00').toISOString()
  },
  {
    age: 62,
    gender: 'Female',
    hypertension: '1',
    heart_disease: '1',
    avg_glucose_level: '150',
    bmi: '32.1',
    prediction: 0.75,
    timestamp: new Date('2024-01-05T10:15:00').toISOString()
  }
];

function PredictionHistory({ history }) {
  const displayHistory = history.length === 0 ? examplePredictions : history;

  const formatDate = (date) => {
    return new Date(date).toLocaleString('es-ES', {
      day: '2-digit',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getRiskLevel = (prediction) => {
    if (prediction > 0.7) return { text: 'Alto', color: '#ff4d4d' };
    if (prediction > 0.3) return { text: 'Moderado', color: '#ffd700' };
    return { text: 'Bajo', color: '#4caf50' };
  };

  return (
    <div className="prediction-history">
      <h2 className="history-title">Historial de Predicciones</h2>
      <div className="history-list">
        {displayHistory.map((item, index) => {
          const risk = getRiskLevel(item.prediction);
          return (
            <div key={index} className="history-item">
              <div className="history-header">
                <span className="history-date">{formatDate(item.timestamp)}</span>
                <span 
                  className="risk-level"
                  style={{ backgroundColor: risk.color }}
                >
                  {risk.text}
                </span>
              </div>
              <div className="history-details">
                <p>
                  <strong>Edad:</strong>
                  <span>{item.age} años</span>
                </p>
                <p>
                  <strong>Género:</strong>
                  <span>{item.gender === 'Male' ? 'M' : 'F'}</span>
                </p>
                <p>
                  <strong>IMC:</strong>
                  <span>{parseFloat(item.bmi).toFixed(1)}</span>
                </p>
                <p>
                  <strong>Glucosa:</strong>
                  <span>{parseFloat(item.avg_glucose_level).toFixed(0)} mg/dL</span>
                </p>
                <p>
                  <strong>HTA:</strong>
                  <span>{item.hypertension === '1' ? 'Sí' : 'No'}</span>
                </p>
                <p>
                  <strong>Card.:</strong>
                  <span>{item.heart_disease === '1' ? 'Sí' : 'No'}</span>
                </p>
              </div>
              <div className="prediction-percentage">
                Riesgo: {(item.prediction * 100).toFixed(1)}%
              </div>
            </div>
          );
        })}
        {history.length === 0 && (
          <div className="example-note">
            * Estos son ejemplos. Tus predicciones aparecerán aquí.
          </div>
        )}
      </div>
    </div>
  );
}

export default PredictionHistory; 
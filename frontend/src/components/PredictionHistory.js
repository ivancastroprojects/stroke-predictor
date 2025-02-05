import React from 'react';
import './PredictionHistory.css';

const examplePredictions = [
  {
    age: 55,
    gender: 'Male',
    hypertension: '1',
    heart_disease: '0',
    ever_married: 'Yes',
    work_type: 'Private',
    Residence_type: 'Urban',
    avg_glucose_level: '120',
    bmi: '28.5',
    smoking_status: 'formerly smoked',
    prediction: 0.35,
    timestamp: new Date('2024-01-06T15:30:00').toISOString()
  },
  {
    age: 68,
    gender: 'Female',
    hypertension: '1',
    heart_disease: '1',
    ever_married: 'Yes',
    work_type: 'Retired',
    Residence_type: 'Rural',
    avg_glucose_level: '150',
    bmi: '32.1',
    smoking_status: 'never smoked',
    prediction: 0.75,
    timestamp: new Date('2024-01-05T10:15:00').toISOString()
  }
];

function PredictionHistory({ history }) {
  const displayHistory = history.length === 0 ? examplePredictions : history;

  const formatDate = (date) => {
    return new Date(date).toLocaleString('en-US', {
      day: '2-digit',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getRiskLevel = (prediction) => {
    if (prediction > 0.7) return { text: 'High', color: '#ff4d4d' };
    if (prediction > 0.3) return { text: 'Moderate', color: '#ffd700' };
    return { text: 'Low', color: '#4caf50' };
  };

  return (
    <div className="prediction-history">
      <h2 className="history-title">Prediction History</h2>
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
                <div className="details-column">
                  <p>
                    <strong>Age:</strong>
                    <span>{item.age} years</span>
                  </p>
                  <p>
                    <strong>Gender:</strong>
                    <span>{item.gender}</span>
                  </p>
                  <p>
                    <strong>BMI:</strong>
                    <span>{parseFloat(item.bmi).toFixed(1)}</span>
                  </p>
                  <p>
                    <strong>Glucose:</strong>
                    <span>{parseFloat(item.avg_glucose_level).toFixed(0)} mg/dL</span>
                  </p>
                </div>
                <div className="details-column">
                  <p>
                    <strong>HBP:</strong>
                    <span>{item.hypertension === '1' ? 'Yes' : 'No'}</span>
                  </p>
                  <p>
                    <strong>Heart:</strong>
                    <span>{item.heart_disease === '1' ? 'Yes' : 'No'}</span>
                  </p>
                  <p>
                    <strong>Smoking:</strong>
                    <span>{item.smoking_status}</span>
                  </p>
                  <p>
                    <strong>Work:</strong>
                    <span>{item.work_type}</span>
                  </p>
                </div>
              </div>
              <div className="prediction-percentage" style={{ color: risk.color }}>
                Risk: {(item.prediction * 100).toFixed(1)}%
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
import React from 'react';

export default function PredictionHistory({ history }) {
  // Función para formatear la fecha
  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    const day = date.getDate().toString().padStart(2, '0');
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    return `${day}/${month} - ${hours}:${minutes}`;
  };

  // Función para formatear el género
  const formatGender = (gender) => {
    const genderMap = {
      'Male': 'Hombre',
      'Female': 'Mujer'
    };
    return genderMap[gender] || gender;
  };

  return (
    <div className="history-container">
      <h3>Historial de Predicciones</h3>
      <div className="history-list">
        {history.map((entry, index) => (
          <div key={index} className="history-item">
            <div className="history-header">
              <div className="prediction-title">
                <span className="patient-name">
                  {formatGender(entry.gender)}, {entry.age} años
                </span>
                <span className="prediction-date">
                  {formatDate(entry.timestamp)}
                </span>
              </div>
              <span className={`risk-badge ${entry.prediction > 0.5 ? 'high' : 'low'}`}>
                Riesgo {entry.prediction > 0.5 ? 'Alto' : 'Bajo'}
              </span>
            </div>
            <details>
              <summary className="toggle-details">Ver detalles</summary>
              <div className="history-details">
                {Object.entries(entry)
                  .filter(([key]) => !['timestamp', 'prediction', 'gender', 'age'].includes(key))
                  .map(([key, value]) => (
                    <p key={key}>
                      <span className="detail-label">{formatLabel(key)}</span>
                      <span className="detail-value">{formatValue(value)}</span>
                    </p>
                  ))
                }
              </div>
            </details>
          </div>
        ))}
      </div>
    </div>
  );
}

// Función auxiliar para formatear etiquetas
function formatLabel(key) {
  const labels = {
    hypertension: 'Hipertensión',
    heart_disease: 'Enf. Cardíaca',
    ever_married: 'Estado Civil',
    work_type: 'Tipo de Trabajo',
    Residence_type: 'Residencia',
    avg_glucose_level: 'Nivel de Glucosa',
    bmi: 'IMC',
    smoking_status: 'Tabaquismo'
  };
  return labels[key] || key;
}

// Función auxiliar para formatear valores
function formatValue(value) {
  if (value === '1') return 'Sí';
  if (value === '0') return 'No';
  return value;
} 
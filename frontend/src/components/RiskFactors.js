import React from 'react';
import './RiskFactors.css';

const riskFactors = [
  {
    name: 'Edad',
    importance: 58.5,
    description: 'El riesgo aumenta significativamente con la edad.'
  },
  {
    name: 'Hipertensión',
    importance: 11.4,
    description: 'La presión arterial alta es un factor de riesgo importante.'
  },
  {
    name: 'Residencia',
    importance: 6.7,
    description: 'El tipo de residencia puede influir en el acceso a atención médica.'
  },
  {
    name: 'Nivel de glucosa',
    importance: 6.2,
    description: 'Niveles elevados de glucosa aumentan el riesgo.'
  },
  {
    name: 'IMC',
    importance: 5.8,
    description: 'El índice de masa corporal afecta el riesgo cardiovascular.'
  }
];

function RiskFactors() {
  return (
    <div className="risk-factors">
      <h2>Factores de Riesgo</h2>
      <div className="risk-factors-list">
        {riskFactors.map((factor, index) => (
          <div key={index} className="risk-factor-item">
            <div className="risk-factor-header">
              <h3>{factor.name}</h3>
              <span className="importance">{factor.importance}%</span>
            </div>
            <p>{factor.description}</p>
            <div className="importance-bar">
              <div 
                className="importance-fill"
                style={{ width: `${factor.importance}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default RiskFactors;
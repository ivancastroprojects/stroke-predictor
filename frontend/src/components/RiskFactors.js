import React from 'react';
import './RiskFactors.css';

const riskFactors = [
  {
    name: 'Age',
    importance: 58.5,
    description: 'Risk increases significantly with age.'
  },
  {
    name: 'Hypertension',
    importance: 11.4,
    description: 'High blood pressure is a major risk factor.'
  },
  {
    name: 'Residence',
    importance: 6.7,
    description: 'Type of residence can influence access to medical care.'
  },
  {
    name: 'Glucose Level',
    importance: 6.2,
    description: 'Elevated glucose levels increase risk.'
  },
  {
    name: 'BMI',
    importance: 5.8,
    description: 'Body Mass Index affects cardiovascular risk.'
  }
];

function RiskFactors() {
  return (
    <div className="risk-factors">
      <h2>Risk Factors</h2>
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
import React from 'react';

export default function AgeRiskChart() {
  const ageGroups = [
    { age: '20-40', risk: 2, color: '#4CAF50' },
    { age: '41-60', risk: 8, color: '#FFC107' },
    { age: '61-75', risk: 15, color: '#FF9800' },
    { age: '75+', risk: 25, color: '#F44336' }
  ];

  return (
    <div className="age-risk-chart">
      <h3>Riesgo de ACV por Edad</h3>
      <div className="age-groups">
        {ageGroups.map((group, index) => (
          <div key={index} className="age-group">
            <div className="age-bar-container">
              <div 
                className="age-bar"
                style={{ 
                  height: `${group.risk * 3}px`,
                  backgroundColor: group.color
                }}
              >
                <span className="risk-percentage">{group.risk}%</span>
              </div>
            </div>
            <div className="age-label">{group.age} años</div>
          </div>
        ))}
      </div>
      <div className="chart-legend">
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: '#4CAF50' }}></span>
          <span>Riesgo Bajo</span>
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: '#F44336' }}></span>
          <span>Riesgo Alto</span>
        </div>
      </div>
      <p className="chart-description">
        El riesgo de ACV aumenta considerablemente después de los 60 años
      </p>
    </div>
  );
} 
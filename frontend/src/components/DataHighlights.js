import React from 'react';
import './DataHighlights.css';

function DataHighlights() {
  return (
    <div className="data-highlights">
      <h3 className="highlights-title">Insights del Análisis de Datos</h3>
      
      <div className="highlights-grid">
        <div className="highlight-card">
          <div className="highlight-icon">🔍</div>
          <div className="highlight-content">
            <h4>Distribución por Género</h4>
            <div className="stat-bar">
              <div className="stat-fill" style={{ width: '48%', backgroundColor: '#00b7ff' }}>
                <span>Hombres: 48%</span>
              </div>
              <div className="stat-fill" style={{ width: '52%', backgroundColor: '#ff69b4' }}>
                <span>Mujeres: 52%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="highlight-card">
          <div className="highlight-icon">📊</div>
          <div className="highlight-content">
            <h4>Incidencia por Edad</h4>
            <div className="stat-group">
              <div className="stat-item">
                <span className="stat-label">Mayor Riesgo:</span>
                <span className="stat-value">65+ años</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Edad Media:</span>
                <span className="stat-value">43.2 años</span>
              </div>
            </div>
          </div>
        </div>

        <div className="highlight-card">
          <div className="highlight-icon">❤️</div>
          <div className="highlight-content">
            <h4>Factores de Riesgo</h4>
            <div className="risk-percentages">
              <div className="risk-bar">
                <span className="risk-label">Hipertensión</span>
                <div className="risk-progress">
                  <div style={{ width: '25%' }} className="risk-fill"></div>
                </div>
                <span className="risk-value">25%</span>
              </div>
              <div className="risk-bar">
                <span className="risk-label">Diabetes</span>
                <div className="risk-progress">
                  <div style={{ width: '20%' }} className="risk-fill"></div>
                </div>
                <span className="risk-value">20%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="highlight-card">
          <div className="highlight-icon">🏥</div>
          <div className="highlight-content">
            <h4>Correlaciones Clave</h4>
            <div className="correlation-list">
              <div className="correlation-item">
                <span>Edad + Hipertensión</span>
                <div className="correlation-bar high"></div>
              </div>
              <div className="correlation-item">
                <span>BMI + Glucosa</span>
                <div className="correlation-bar medium"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DataHighlights; 
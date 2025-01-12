import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {

  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  XAxis,
  YAxis
} from 'recharts';
import './PredictionResults.css';

export default function PredictionResults() {
  const location = useLocation();
  const navigate = useNavigate();
  const { prediction, formData, featureContributions, riskFactors, featureImportance } = location.state || {};
  
  const riskLevel = prediction > 0.7 ? 'High' : prediction > 0.3 ? 'Moderate' : 'Low';
  const riskColor = {
    'High': '#ff4d4d',
    'Moderate': '#ffd700',
    'Low': '#4caf50'
  }[riskLevel];

  // Datos para el gráfico radial de riesgo
  const riskData = [{
    name: 'Risk',
    value: prediction * 100,
    fill: riskColor
  }];

  // Configuración del arco para el medidor de riesgo
  const startAngle = 180;
  const endAngle = 0;
  const riskEndAngle = startAngle - ((prediction * 100) * (startAngle - endAngle) / 100);

  // Datos para el arco de fondo y el arco de riesgo
  const gaugeData = [
    { value: 100, fill: 'rgba(255, 255, 255, 0.1)' },  // Fondo
    { value: prediction * 100, fill: riskColor }        // Riesgo
  ];

  // Marcas de graduación para el medidor
  const ticks = [0, 25, 50, 75, 100];

  // Datos para el gráfico de contribuciones
  const contributionsData = Object.entries(featureContributions || {})
    .map(([name, value]) => ({
      name,
      value: parseFloat(value.toFixed(1))
    }))
    .sort((a, b) => b.value - a.value);

  const COLORS = ['#00b7ff', '#00ffd5', '#7cffb2', '#ffd700', '#ff6b6b'];

  const handleBack = () => {
    navigate('/');
  };

  return (
    <div className="results-container">
      <div className="results-header">
        <button className="back-button" onClick={handleBack}>
          ← Back
        </button>
        <h1 className="results-title">Analysis Results</h1>
      </div>

      <div className="results-grid">
        {/* Main Risk Panel */}
        <div className="result-panel main-risk">
          <h2>Stroke Risk Level</h2>
          <div className="risk-visualization">
            <div className="gauge-container">
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  {/* Arco de fondo */}
                  <Pie
                    data={[gaugeData[0]]}
                    dataKey="value"
                    cx="50%"
                    cy="50%"
                    innerRadius={80}
                    outerRadius={95}
                    startAngle={startAngle}
                    endAngle={endAngle}
                    stroke="none"
                  />
                  {/* Arco de riesgo */}
                  <Pie
                    data={[gaugeData[1]]}
                    dataKey="value"
                    cx="50%"
                    cy="50%"
                    innerRadius={80}
                    outerRadius={95}
                    startAngle={startAngle}
                    endAngle={riskEndAngle}
                    stroke="none"
                  />
                  {/* Valor central */}
                  <text
                    x="50%"
                    y="45%"
                    textAnchor="middle"
                    fill="#fff"
                    fontSize="36"
                    fontWeight="bold"
                  >
                    {`${(prediction * 100).toFixed(1)}%`}
                  </text>
                  <text
                    x="50%"
                    y="60%"
                    textAnchor="middle"
                    fill="rgba(255, 255, 255, 0.7)"
                    fontSize="14"
                  >
                    Risk Level
                  </text>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="risk-details">
              <div className="risk-indicator" style={{ backgroundColor: riskColor }}>
                <div className="risk-header">
                  <span className="risk-icon">
                    {riskLevel === 'High' && '⚠️'}
                    {riskLevel === 'Moderate' && '⚡'}
                    {riskLevel === 'Low' && '✅'}
                  </span>
                  <span className="risk-text">{riskLevel} Risk</span>
                </div>
                <span className="risk-description">
                  {riskLevel === 'High' && 'Immediate medical attention recommended'}
                  {riskLevel === 'Moderate' && 'Preventive medical consultation suggested'}
                  {riskLevel === 'Low' && 'Maintain healthy habits'}
                </span>
              </div>
              <div className="risk-scale">
                <div className="scale-item" style={{ backgroundColor: '#4caf50', opacity: riskLevel === 'Low' ? 1 : 0.3 }}>
                  Low
                </div>
                <div className="scale-item" style={{ backgroundColor: '#ffd700', opacity: riskLevel === 'Moderate' ? 1 : 0.3 }}>
                  Moderate
                </div>
                <div className="scale-item" style={{ backgroundColor: '#ff4d4d', opacity: riskLevel === 'High' ? 1 : 0.3 }}>
                  High
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Contributing Factors Panel */}
        <div className="result-panel factors-panel">
          <h2>Contributing Factors</h2>
          <div className="factors-visualization">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={contributionsData} layout="vertical">
                <XAxis type="number" domain={[0, 100]} />
                <YAxis dataKey="name" type="category" width={150} />
                <Tooltip />
                <Bar dataKey="value" fill="#00b7ff">
                  {contributionsData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Patient Data Panel */}
        <div className="result-panel patient-data">
          <h2>Patient Data</h2>
          <div className="patient-grid">
            <div className="data-item">
              <span className="label">Age</span>
              <span className="value">{formData.age} years</span>
            </div>
            <div className="data-item">
              <span className="label">Gender</span>
              <span className="value">{formData.gender === 'Male' ? 'Male' : 'Female'}</span>
            </div>
            <div className="data-item">
              <span className="label">BMI</span>
              <span className="value">{parseFloat(formData.bmi).toFixed(1)} kg/m²</span>
            </div>
            <div className="data-item">
              <span className="label">Glucose</span>
              <span className="value">{parseFloat(formData.avg_glucose_level).toFixed(0)} mg/dL</span>
            </div>
            <div className="data-item">
              <span className="label">Hypertension</span>
              <span className="value">{formData.hypertension === '1' ? 'Yes' : 'No'}</span>
            </div>
            <div className="data-item">
              <span className="label">Heart Disease</span>
              <span className="value">{formData.heart_disease === '1' ? 'Yes' : 'No'}</span>
            </div>
          </div>
        </div>

        {/* Recommendations Panel */}
        <div className="result-panel recommendations">
          <h2>Recommendations</h2>
          <div className="recommendations-list">
            {formData.hypertension === '1' && (
              <div className="recommendation-item">
                <div className="icon">🫀</div>
                <div className="content">
                  <h3>Blood Pressure Control</h3>
                  <p>Maintain regular blood pressure monitoring and follow prescribed treatment.</p>
                </div>
              </div>
            )}
            {parseFloat(formData.bmi) > 25 && (
              <div className="recommendation-item">
                <div className="icon">⚖️</div>
                <div className="content">
                  <h3>Weight Management</h3>
                  <p>Maintain a balanced diet and regular physical activity.</p>
                </div>
              </div>
            )}
            {parseFloat(formData.avg_glucose_level) > 125 && (
              <div className="recommendation-item">
                <div className="icon">🩺</div>
                <div className="content">
                  <h3>Glucose Control</h3>
                  <p>Monitor glucose levels and maintain an appropriate diet.</p>
                </div>
              </div>
            )}
            <div className="recommendation-item">
              <div className="icon">🏃</div>
              <div className="content">
                <h3>Healthy Lifestyle</h3>
                <p>Maintain regular physical activity, avoid tobacco, and limit alcohol consumption.</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="results-footer">
        <p className="disclaimer">
          * This analysis is a support tool and does not replace professional medical diagnosis.
          Consult with your doctor for a complete evaluation.
        </p>
      </div>
    </div>
  );
} 
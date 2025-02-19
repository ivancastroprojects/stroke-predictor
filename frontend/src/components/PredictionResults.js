import React, { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis
} from 'recharts';
import './PredictionResults.css';

export default function PredictionResults() {
  const location = useLocation();
  const navigate = useNavigate();
  const { prediction, formData, featureContributions, riskFactors } = location.state || {};
  
  useEffect(() => {
    window.scrollTo(0, 0);
    console.log('Estado recibido:', location.state);
    console.log('Feature Contributions:', featureContributions);
    console.log('Risk Factors:', riskFactors);
  }, [location.state, featureContributions, riskFactors]);

  if (!location.state) {
    navigate('/');
    return null;
  }
  
  // Convertir la probabilidad a porcentaje
  const riskPercentage = prediction * 100;
  const riskLevel = riskPercentage > 70 ? 'High' : riskPercentage > 30 ? 'Moderate' : 'Low';
  const riskColor = {
    'High': '#ff4d4d',
    'Moderate': '#ffd700',
    'Low': '#4caf50'
  }[riskLevel];

  // Datos para el gráfico semicircular
  const riskData = [
    { name: 'Risk', value: riskPercentage },
    { name: 'Remaining', value: 100 - riskPercentage }
  ];

  // Procesar los datos de contribución de características
  const processedFeatureContributions = featureContributions ? 
    Object.entries(featureContributions).map(([name, value]) => ({
      name,
      value: parseFloat((value * 100).toFixed(1))
    })).sort((a, b) => b.value - a.value) : [];

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
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={riskData}
                    cx="50%"
                    cy="50%"
                    startAngle={180}
                    endAngle={0}
                    innerRadius={85}
                    outerRadius={110}
                    fill={riskColor}
                    paddingAngle={0}
                    dataKey="value"
                    blendStroke
                  >
                    {riskData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`}
                        fill={index === 0 ? riskColor : 'rgba(0, 64, 77, 0.3)'}
                        strokeWidth={0}
                      />
                    ))}
                  </Pie>
                  <text
                    x="50%"
                    y="45%"
                    textAnchor="middle"
                    fill="#fff"
                    className="risk-value-label"
                  >
                    {`${riskPercentage.toFixed(1)}%`}
                  </text>
                  <text
                    x="50%"
                    y="65%"
                    textAnchor="middle"
                    fill="rgba(255, 255, 255, 0.7)"
                    className="risk-label-text"
                  >
                    Risk Level
                  </text>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="risk-details">
              <div className="risk-indicator" style={{ backgroundColor: `${riskColor}20` }}>
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
                <div 
                  className="scale-item" 
                  style={{ 
                    backgroundColor: riskLevel === 'Low' ? '#4caf50' : '#4caf5020',
                    color: '#4caf50',
                    opacity: 1
                  }}
                >
                  Low
                </div>
                <div 
                  className="scale-item" 
                  style={{ 
                    backgroundColor: riskLevel === 'Moderate' ? '#ffd700' : '#ffd70020',
                    color: '#ffd700',
                    opacity: 1
                  }}
                >
                  Moderate
                </div>
                <div 
                  className="scale-item" 
                  style={{ 
                    backgroundColor: riskLevel === 'High' ? '#ff4d4d' : '#ff4d4d20',
                    color: '#ff4d4d',
                    opacity: 1
                  }}
                >
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
              <BarChart data={Object.entries(featureContributions || {}).map(([name, value]) => ({
                name: name.replace(/_/g, ' ').split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' '),
                value: parseFloat((value * 100).toFixed(1))
              })).sort((a, b) => b.value - a.value)} layout="vertical">
                <XAxis type="number" domain={[0, 100]} />
                <YAxis dataKey="name" type="category" width={150} />
                <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
                <Bar dataKey="value" fill="#00b7ff">
                  {Object.entries(featureContributions || {}).map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Key Factors Panel */}
        <div className="result-panel factors-explanation-panel">
          <h2>Key Risk Factors</h2>
          <div className="factors-list">
            {(riskFactors || []).map((factor, index) => (
              <div key={index} className="factor-item">
                <div className="factor-header">
                  <div className="factor-icon" style={{ backgroundColor: `${COLORS[index]}20`, color: COLORS[index] }}>
                    {index + 1}
                  </div>
                  <h4>{factor.factor}</h4>
                </div>
                <p className="factor-description">
                  {factor.message}
                </p>
              </div>
            ))}
            {(!riskFactors || riskFactors.length === 0) && (
              <div className="no-factors-message">
                No significant risk factors identified. Continue maintaining healthy habits.
              </div>
            )}
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
              <span className="value">{formData.gender}</span>
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
            <div className="data-item">
              <span className="label">Marital Status</span>
              <span className="value">{formData.ever_married}</span>
            </div>
            <div className="data-item">
              <span className="label">Work Type</span>
              <span className="value">{formData.work_type.replace('_', ' ')}</span>
            </div>
            <div className="data-item">
              <span className="label">Residence</span>
              <span className="value">{formData.Residence_type}</span>
            </div>
            <div className="data-item">
              <span className="label">Smoking</span>
              <span className="value">{formData.smoking_status}</span>
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
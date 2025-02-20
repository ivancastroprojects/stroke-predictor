import React, { useEffect, useMemo } from 'react';
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

// Colores para los factores contribuyentes
const COLORS = [
  '#00b7ff',
  '#00ffd5',
  '#4caf50',
  '#ffd700',
  '#ff9800',
  '#ff4d4d',
  '#e91e63'
];

// Mapeo de nombres de factores en español
const FACTOR_NAMES = {
  'age': 'Edad',
  'hypertension': 'Hipertensión',
  'heart_disease': 'Enfermedad Cardíaca',
  'avg_glucose_level': 'Nivel de Glucosa',
  'bmi': 'IMC',
  'smoking_status': 'Tabaquismo',
  'work_type': 'Tipo de Trabajo'
};

export default function PredictionResults() {
  const location = useLocation();
  const navigate = useNavigate();
  const { prediction = 0, formData = {}, featureContributions = {}, riskFactors = [] } = location.state || {};
  
  useEffect(() => {
    if (!location.state) {
      navigate('/');
      return;
    }
    
    window.scrollTo(0, 0);
    console.log('Estado recibido:', location.state);
    console.log('Feature Contributions:', featureContributions);
    console.log('Risk Factors:', riskFactors);
  }, [location.state, featureContributions, riskFactors, navigate]);

  // Procesar el porcentaje de riesgo
  const riskPercentage = prediction * 100;
  
  // Determinar el nivel de riesgo y color
  const riskLevel = useMemo(() => {
    if (riskPercentage >= 70) return 'High';
    if (riskPercentage >= 30) return 'Moderate';
    return 'Low';
  }, [riskPercentage]);

  const riskColor = useMemo(() => {
    if (riskLevel === 'High') return '#ff4d4d';
    if (riskLevel === 'Moderate') return '#ffd700';
    return '#4caf50';
  }, [riskLevel]);

  // Datos para el medidor de riesgo
  const riskData = useMemo(() => [
    { value: riskPercentage },
    { value: 100 - riskPercentage }
  ], [riskPercentage]);

  // Procesar los factores contribuyentes
  const processedFeatureContributions = useMemo(() => {
    return Object.entries(featureContributions)
      .map(([name, value]) => ({
        name: FACTOR_NAMES[name] || name.replace(/_/g, ' ')
          .split(' ')
          .map(word => word.charAt(0).toUpperCase() + word.slice(1))
          .join(' '),
        value: parseFloat((value * 100).toFixed(1))
      }))
      .sort((a, b) => b.value - a.value);
  }, [featureContributions]);

  // Generar recomendaciones basadas en los datos del paciente
  const patientSpecificRecommendations = useMemo(() => {
    const recommendations = [];

    if (formData.hypertension === '1') {
      recommendations.push({
        icon: '🫀',
        title: 'Control de Presión Arterial',
        text: 'Mantener monitoreo regular de la presión arterial y seguir el tratamiento prescrito.'
      });
    }

    if (parseFloat(formData.bmi) > 25) {
      recommendations.push({
        icon: '⚖️',
        title: 'Control de Peso',
        text: 'Mantener una dieta equilibrada y actividad física regular para alcanzar un peso saludable.'
      });
    }

    if (parseFloat(formData.avg_glucose_level) > 125) {
      recommendations.push({
        icon: '🩺',
        title: 'Control de Glucosa',
        text: 'Monitorear niveles de glucosa y mantener una dieta apropiada.'
      });
    }

    if (formData.smoking_status === 'smokes') {
      recommendations.push({
        icon: '🚭',
        title: 'Cesación de Tabaco',
        text: 'Considerar programas para dejar de fumar y buscar apoyo profesional.'
      });
    }

    return recommendations;
  }, [formData]);

  const handleBack = () => {
    navigate('/');
  };

  if (!location.state) return null;

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
          <div className="risk-visualization">
            <h2>Nivel de Riesgo de ACV</h2>
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
                    Riesgo de ACV
                  </text>
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="risk-details">
            <div className="risk-indicator" style={{ backgroundColor: `${riskColor}20` }}>
              <div className="risk-header">
                <span className="risk-icon">
                  {riskLevel === 'High' && '⚠️'}
                  {riskLevel === 'Moderate' && '⚡'}
                  {riskLevel === 'Low' && '✅'}
                </span>
                <span className="risk-text">{riskLevel === 'High' ? 'Riesgo Alto' : riskLevel === 'Moderate' ? 'Riesgo Moderado' : 'Riesgo Bajo'}</span>
              </div>
              <span className="risk-description">
                {riskLevel === 'High' && 'Se recomienda atención médica inmediata y evaluación neurológica urgente'}
                {riskLevel === 'Moderate' && 'Se sugiere consulta médica preventiva y monitoreo regular'}
                {riskLevel === 'Low' && 'Mantener hábitos saludables y chequeos regulares'}
              </span>
            </div>

            <div className="medical-recommendations">
              <h3>Recomendaciones Médicas</h3>
              <div className="recommendation-list">
                {riskLevel === 'High' && (
                  <>
                    <div className="recommendation-item">
                      <span className="recommendation-icon">🏥</span>
                      <div className="recommendation-content">
                        <div className="recommendation-title">Evaluación Neurológica</div>
                        <div className="recommendation-text">Programar una evaluación neurológica completa para valorar factores de riesgo específicos.</div>
                      </div>
                    </div>
                    <div className="recommendation-item">
                      <span className="recommendation-icon">💊</span>
                      <div className="recommendation-content">
                        <div className="recommendation-title">Control de Medicación</div>
                        <div className="recommendation-text">Revisión y ajuste de medicamentos actuales bajo supervisión médica.</div>
                      </div>
                    </div>
                  </>
                )}
                {riskLevel === 'Moderate' && (
                  <>
                    <div className="recommendation-item">
                      <span className="recommendation-icon">🩺</span>
                      <div className="recommendation-content">
                        <div className="recommendation-title">Chequeo Preventivo</div>
                        <div className="recommendation-text">Realizar exámenes preventivos y monitoreo de factores de riesgo.</div>
                      </div>
                    </div>
                    <div className="recommendation-item">
                      <span className="recommendation-icon">📊</span>
                      <div className="recommendation-content">
                        <div className="recommendation-title">Control Regular</div>
                        <div className="recommendation-text">Seguimiento periódico de presión arterial y niveles de glucosa.</div>
                      </div>
                    </div>
                  </>
                )}
                {riskLevel === 'Low' && (
                  <>
                    <div className="recommendation-item">
                      <span className="recommendation-icon">🏃</span>
                      <div className="recommendation-content">
                        <div className="recommendation-title">Mantener Actividad</div>
                        <div className="recommendation-text">Continuar con actividad física regular y dieta saludable.</div>
                      </div>
                    </div>
                    <div className="recommendation-item">
                      <span className="recommendation-icon">🍎</span>
                      <div className="recommendation-content">
                        <div className="recommendation-title">Hábitos Saludables</div>
                        <div className="recommendation-text">Mantener una dieta equilibrada y evitar el tabaco.</div>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Contributing Factors Panel */}
        <div className="result-panel factors-panel">
          <h2>Factores Contribuyentes</h2>
          <div className="factors-visualization">
            <div className="factors-chart">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart 
                  data={processedFeatureContributions} 
                  layout="vertical"
                  margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
                >
                  <XAxis type="number" domain={[0, 100]} />
                  <YAxis 
                    dataKey="name" 
                    type="category" 
                    width={150}
                    tick={{ fill: 'rgba(255, 255, 255, 0.8)' }}
                  />
                  <Tooltip 
                    formatter={(value) => `${value.toFixed(1)}%`}
                    contentStyle={{
                      background: 'rgba(0, 64, 77, 0.95)',
                      border: '1px solid rgba(0, 183, 255, 0.2)',
                      borderRadius: '8px'
                    }}
                  />
                  <Bar dataKey="value" fill="#00b7ff">
                    {processedFeatureContributions.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="factors-legend">
              {processedFeatureContributions.map((factor, index) => (
                <div key={factor.name} className="legend-item">
                  <div 
                    className="legend-color" 
                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                  />
                  <span className="legend-text">
                    {factor.name}: {factor.value}%
                  </span>
                </div>
              ))}
            </div>
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
        <div className="result-panel lifestyle-insights">
          <h2>Insights de Estilo de Vida</h2>
          <div className="insights-grid">
            <div className="insight-card">
              <div className="insight-header">
                <span className="insight-icon">🎯</span>
                <h3>Objetivos de Salud</h3>
              </div>
              <ul className="insight-list">
                {parseFloat(formData.bmi) > 25 && (
                  <li>Reducir IMC a rango normal (18.5-24.9)</li>
                )}
                {parseFloat(formData.avg_glucose_level) > 125 && (
                  <li>Mantener glucosa en ayunas menor a 100 mg/dL</li>
                )}
                {formData.hypertension === '1' && (
                  <li>Mantener presión arterial menor a 120/80 mmHg</li>
                )}
              </ul>
            </div>
            
            <div className="insight-card">
              <div className="insight-header">
                <span className="insight-icon">📊</span>
                <h3>Métricas Clave</h3>
              </div>
              <div className="metrics-grid">
                <div className="metric-item">
                  <span className="metric-label">IMC Actual</span>
                  <span className="metric-value">{parseFloat(formData.bmi).toFixed(1)}</span>
                  <span className="metric-status" style={{
                    color: formData.bmi > 25 ? '#ff4d4d' : '#4caf50'
                  }}>
                    {formData.bmi > 30 ? '⚠️ Alto' : formData.bmi > 25 ? '⚠️ Sobrepeso' : '✓ Normal'}
                  </span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Glucosa</span>
                  <span className="metric-value">{parseFloat(formData.avg_glucose_level).toFixed(0)}</span>
                  <span className="metric-status" style={{
                    color: formData.avg_glucose_level > 125 ? '#ff4d4d' : '#4caf50'
                  }}>
                    {formData.avg_glucose_level > 125 ? '⚠️ Elevada' : '✓ Normal'}
                  </span>
                </div>
              </div>
            </div>

            <div className="insight-card">
              <div className="insight-header">
                <span className="insight-icon">📅</span>
                <h3>Seguimiento Recomendado</h3>
              </div>
              <div className="followup-list">
                {riskLevel === 'High' && (
                  <div className="followup-item urgent">
                    <span>Consulta médica inmediata</span>
                    <span className="timeframe">En las próximas 24-48 horas</span>
                  </div>
                )}
                {riskLevel === 'Moderate' && (
                  <div className="followup-item warning">
                    <span>Evaluación preventiva</span>
                    <span className="timeframe">En las próximas 2 semanas</span>
                  </div>
                )}
                <div className="followup-item">
                  <span>Control regular de presión arterial</span>
                  <span className="timeframe">Cada 2-4 semanas</span>
                </div>
                <div className="followup-item">
                  <span>Análisis de sangre completo</span>
                  <span className="timeframe">Cada 3-6 meses</span>
                </div>
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
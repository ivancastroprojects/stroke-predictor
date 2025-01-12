import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  RadialBarChart,
  RadialBar,
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
  
  const riskLevel = prediction > 0.7 ? 'Alto' : prediction > 0.3 ? 'Moderado' : 'Bajo';
  const riskColor = {
    'Alto': '#ff4d4d',
    'Moderado': '#ffd700',
    'Bajo': '#4caf50'
  }[riskLevel];

  // Datos para el gráfico radial de riesgo
  const riskData = [{
    name: 'Riesgo',
    value: prediction * 100,
    fill: riskColor
  }];

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
          ← Volver
        </button>
        <h1 className="results-title">Resultados del Análisis</h1>
      </div>

      <div className="results-grid">
        {/* Panel Principal de Riesgo */}
        <div className="result-panel main-risk">
          <h2>Nivel de Riesgo de Accidente Cerebrovascular</h2>
          <div className="risk-visualization">
            <div className="gauge-container">
              <ResponsiveContainer width="100%" height={400}>
                <RadialBarChart 
                  cx="50%" 
                  cy="50%" 
                  innerRadius="65%" 
                  outerRadius="85%" 
                  barSize={30}
                  data={riskData} 
                  startAngle={180} 
                  endAngle={0}
                >
                  <RadialBar
                    background={{ fill: 'rgba(255, 255, 255, 0.1)' }}
                    dataKey="value"
                    cornerRadius={30}
                  />
                  <text
                    x="50%"
                    y="45%"
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className="risk-value-label"
                  >
                    {`${(prediction * 100).toFixed(1)}%`}
                  </text>
                  <text
                    x="50%"
                    y="55%"
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className="risk-label-text"
                  >
                    Nivel de Riesgo
                  </text>
                </RadialBarChart>
              </ResponsiveContainer>
            </div>
            <div className="risk-details">
              <div className="risk-indicator" style={{ backgroundColor: riskColor }}>
                <div className="risk-header">
                  <span className="risk-icon">
                    {riskLevel === 'Alto' && '⚠️'}
                    {riskLevel === 'Moderado' && '⚡'}
                    {riskLevel === 'Bajo' && '✅'}
                  </span>
                  <span className="risk-text">Riesgo {riskLevel}</span>
                </div>
                <span className="risk-description">
                  {riskLevel === 'Alto' && 'Se recomienda atención médica inmediata'}
                  {riskLevel === 'Moderado' && 'Se sugiere consulta médica preventiva'}
                  {riskLevel === 'Bajo' && 'Mantener hábitos saludables'}
                </span>
              </div>
              <div className="risk-scale">
                <div className="scale-item" style={{ backgroundColor: '#4caf50', opacity: riskLevel === 'Bajo' ? 1 : 0.3 }}>
                  Bajo
                </div>
                <div className="scale-item" style={{ backgroundColor: '#ffd700', opacity: riskLevel === 'Moderado' ? 1 : 0.3 }}>
                  Moderado
                </div>
                <div className="scale-item" style={{ backgroundColor: '#ff4d4d', opacity: riskLevel === 'Alto' ? 1 : 0.3 }}>
                  Alto
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Panel de Factores Contribuyentes */}
        <div className="result-panel factors-panel">
          <h2>Factores Contribuyentes</h2>
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

        {/* Panel de Datos del Paciente */}
        <div className="result-panel patient-data">
          <h2>Datos del Paciente</h2>
          <div className="patient-grid">
            <div className="data-item">
              <span className="label">Edad</span>
              <span className="value">{formData.age} años</span>
            </div>
            <div className="data-item">
              <span className="label">Género</span>
              <span className="value">{formData.gender === 'Male' ? 'Masculino' : 'Femenino'}</span>
            </div>
            <div className="data-item">
              <span className="label">IMC</span>
              <span className="value">{parseFloat(formData.bmi).toFixed(1)} kg/m²</span>
            </div>
            <div className="data-item">
              <span className="label">Glucosa</span>
              <span className="value">{parseFloat(formData.avg_glucose_level).toFixed(0)} mg/dL</span>
            </div>
            <div className="data-item">
              <span className="label">Hipertensión</span>
              <span className="value">{formData.hypertension === '1' ? 'Sí' : 'No'}</span>
            </div>
            <div className="data-item">
              <span className="label">Enf. Cardíaca</span>
              <span className="value">{formData.heart_disease === '1' ? 'Sí' : 'No'}</span>
            </div>
          </div>
        </div>

        {/* Panel de Recomendaciones */}
        <div className="result-panel recommendations">
          <h2>Recomendaciones</h2>
          <div className="recommendations-list">
            {formData.hypertension === '1' && (
              <div className="recommendation-item">
                <div className="icon">🫀</div>
                <div className="content">
                  <h3>Control de Presión Arterial</h3>
                  <p>Mantener un control regular de la presión arterial y seguir el tratamiento prescrito.</p>
                </div>
              </div>
            )}
            {parseFloat(formData.bmi) > 25 && (
              <div className="recommendation-item">
                <div className="icon">⚖️</div>
                <div className="content">
                  <h3>Control de Peso</h3>
                  <p>Mantener una dieta equilibrada y realizar actividad física regular.</p>
                </div>
              </div>
            )}
            {parseFloat(formData.avg_glucose_level) > 125 && (
              <div className="recommendation-item">
                <div className="icon">🩺</div>
                <div className="content">
                  <h3>Control de Glucosa</h3>
                  <p>Monitorear los niveles de glucosa y mantener una dieta apropiada.</p>
                </div>
              </div>
            )}
            <div className="recommendation-item">
              <div className="icon">🏃</div>
              <div className="content">
                <h3>Estilo de Vida Saludable</h3>
                <p>Mantener actividad física regular, evitar el tabaco y limitar el consumo de alcohol.</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="results-footer">
        <p className="disclaimer">
          * Este análisis es una herramienta de apoyo y no sustituye el diagnóstico médico profesional.
          Consulte con su médico para una evaluación completa.
        </p>
      </div>
    </div>
  );
} 
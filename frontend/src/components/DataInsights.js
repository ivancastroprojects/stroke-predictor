import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Area,
  AreaChart
} from 'recharts';
import './DataInsights.css';

const DataInsights = () => {
  // Datos de distribución por edad y género
  const ageGenderDistribution = [
    { age: '0-20', male: 78, female: 70, maleStroke: 2, femaleStroke: 2 },
    { age: '21-40', male: 842, female: 782, maleStroke: 23, femaleStroke: 22 },
    { age: '41-60', male: 1102, female: 1044, maleStroke: 58, femaleStroke: 52 },
    { age: '61-80', male: 612, female: 580, maleStroke: 42, femaleStroke: 36 },
    { age: '81+', male: 52, female: 48, maleStroke: 7, femaleStroke: 5 }
  ];

  // Datos de factores de riesgo
  const riskFactors = [
    { name: 'Age', importance: 58.5, description: 'Risk increases significantly with age, especially after 55.' },
    { name: 'Hypertension', importance: 11.4, description: 'High blood pressure damages blood vessels over time.' },
    { name: 'Residence', importance: 6.7, description: 'Access to healthcare and lifestyle factors vary by location.' },
    { name: 'Glucose Level', importance: 6.2, description: 'High blood sugar increases stroke risk significantly.' },
    { name: 'BMI', importance: 5.8, description: 'Obesity contributes to various stroke risk factors.' },
    { name: 'Heart Disease', importance: 5.2, description: 'Cardiovascular conditions increase stroke probability.' },
    { name: 'Smoking', importance: 4.2, description: 'Smoking damages blood vessels and increases clot risk.' },
    { name: 'Other Factors', importance: 2.0, description: 'Including family history and other medical conditions.' }
  ];

  // Datos de tendencia de riesgo por edad
  const riskTrend = [
    { age: 20, risk: 0.02 },
    { age: 30, risk: 0.04 },
    { age: 40, risk: 0.08 },
    { age: 50, risk: 0.15 },
    { age: 60, risk: 0.25 },
    { age: 70, risk: 0.35 },
    { age: 80, risk: 0.45 }
  ];

  const COLORS = ['#7c00ff', '#ff00d3', '#00e5ff', '#00ff9d', '#ffd700', '#ff4d4d', '#4caf50', '#9c27b0'];

  return (
    <div className="data-insights">
      <div className="insights-header">
        <h2>Stroke Risk Analysis</h2>
        <p className="subtitle">Based on analysis of 5,110 clinical cases</p>
      </div>
      
      <div className="medical-alert">
        <div className="alert-icon">⚕️</div>
        <div className="alert-content">
          <h4>Medical Information</h4>
          <p>This predictive model is a support tool and does not replace professional medical diagnosis. 
             Always consult with a healthcare professional for evaluation and treatment.</p>
        </div>
      </div>

      <div className="insights-grid">
        <div className="chart-container">
          <h3>Age and Gender Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={ageGenderDistribution} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
              <XAxis dataKey="age" stroke="#fff" />
              <YAxis stroke="#fff" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'rgba(0, 0, 0, 0.8)',
                  border: '1px solid rgba(255, 255, 255, 0.1)'
                }}
              />
              <Legend />
              <Bar dataKey="male" name="Male" fill="#7c00ff" stackId="gender" />
              <Bar dataKey="female" name="Female" fill="#ff00d3" stackId="gender" />
              <Bar dataKey="maleStroke" name="Male Stroke" fill="#00e5ff" stackId="stroke" />
              <Bar dataKey="femaleStroke" name="Female Stroke" fill="#00ff9d" stackId="stroke" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <h3>Risk Trend by Age</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={riskTrend} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
              <XAxis dataKey="age" stroke="#fff" label={{ value: 'Age', position: 'bottom', fill: '#fff' }} />
              <YAxis stroke="#fff" label={{ value: 'Risk', angle: -90, position: 'insideLeft', fill: '#fff' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(0, 0, 0, 0.8)',
                  border: '1px solid rgba(255, 255, 255, 0.1)'
                }}
              />
              <Area type="monotone" dataKey="risk" stroke="#7c00ff" fill="url(#colorRisk)" />
              <defs>
                <linearGradient id="colorRisk" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#7c00ff" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#7c00ff" stopOpacity={0}/>
                </linearGradient>
              </defs>
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <h3>Risk Factors Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={riskFactors}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} (${(percent * 100).toFixed(1)}%)`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="importance"
              >
                {riskFactors.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(0, 0, 0, 0.8)',
                  border: '1px solid rgba(255, 255, 255, 0.1)'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="info-box">
          <h3>KEY FINDINGS</h3>
          <ul>
            <li>Age is the most significant factor, with a 58.5% impact on risk</li>
            <li>Hypertension is the main modifiable factor (11.4%)</li>
            <li>The 41-60 age group shows the highest stroke incidence</li>
            <li>Modifiable factors account for ~23% of total risk</li>
            <li>Strong correlation between age and stroke risk observed</li>
          </ul>
        </div>
      </div>

      <div className="medical-disclaimer">
        <p>This predictive model has an accuracy of 83.7% based on cross-validation.
           Results should be interpreted alongside professional clinical evaluation.</p>
      </div>
    </div>
  );
};

export default DataInsights; 
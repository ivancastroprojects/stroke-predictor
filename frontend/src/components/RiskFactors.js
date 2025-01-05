import React, { useState, useEffect } from 'react';
import axios from 'axios';
import AgeRiskChart from './AgeRiskChart';

export default function RiskFactors() {
  const [factors, setFactors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchImportance = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await axios.get('http://localhost:5000/feature-importance');
        
        const importanceData = response.data;
        const factorsArray = Object.entries(importanceData)
          .map(([name, weight]) => ({ 
            name, 
            weight: parseFloat(weight) 
          }))
          .sort((a, b) => b.weight - a.weight);
        
        setFactors(factorsArray);
      } catch (error) {
        console.error('Error:', error);
        setError('Error al cargar los factores de riesgo');
      } finally {
        setLoading(false);
      }
    };

    fetchImportance();
  }, []);

  if (loading) {
    return (
      <div className="risk-factors">
        <h3>IMPORTANCIA DE FACTORES</h3>
        <div className="loading">Cargando factores...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="risk-factors">
        <h3>IMPORTANCIA DE FACTORES</h3>
        <div className="error">{error}</div>
      </div>
    );
  }

  return (
    <div className="risk-factors">
      <h3>IMPORTANCIA DE FACTORES</h3>
      {factors.map(factor => (
        <div key={factor.name} className="risk-factor-item">
          <div className="factor-label">
            {factor.name}
            <span className="factor-percentage">
              {(factor.weight * 100).toFixed(1)}%
            </span>
          </div>
          <div className="factor-bar">
            <div 
              className="factor-fill"
              style={{ 
                '--width': `${factor.weight * 100}%`,
                width: `${factor.weight * 100}%`
              }}
            />
          </div>
        </div>
      ))}
      
      <AgeRiskChart />
    </div>
  );
}
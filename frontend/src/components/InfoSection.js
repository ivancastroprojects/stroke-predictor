import React from 'react';

export default function InfoSection() {
  return (
    <div className="info-section">
      <div className="info-header">
        <h1>Evaluación de Riesgo de Accidente Cerebrovascular</h1>
        <p className="subtitle">
          Herramienta de predicción basada en inteligencia artificial
        </p>
      </div>

      <div className="info-cards">
        <div className="info-card">
          <h2>¿Qué es un ACV?</h2>
          <p>
            Un accidente cerebrovascular (ACV) ocurre cuando el suministro de sangre 
            a una parte del cerebro se interrumpe o reduce, impidiendo que el tejido 
            cerebral reciba oxígeno y nutrientes.
          </p>
          <div className="warning-box">
            <strong>¡Importante!</strong>
            <p>Esta herramienta es solo para fines informativos y no sustituye el 
            diagnóstico médico profesional.</p>
          </div>
        </div>

        <div className="info-card model-info">
          <h2>Precisión del Modelo</h2>
          <div className="accuracy-display">
            <div className="accuracy-circle">
              <span className="accuracy-number">85.3%</span>
              <span className="accuracy-label">Precisión</span>
            </div>
          </div>
          <p>
            Nuestro modelo utiliza aprendizaje automático avanzado y ha sido entrenado 
            con datos de más de 5,000 casos clínicos.
          </p>
        </div>
      </div>
    </div>
  );
} 
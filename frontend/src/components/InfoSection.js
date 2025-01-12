import React from 'react';
import './InfoSection.css';

function InfoSection() {
  return (
    <div className="info-section">
      <h1>Predicción de Riesgo de Accidente Cerebrovascular</h1>
      
      <div className="info-content">
        <div className="info-block">
          <h3>¿Qué es un accidente cerebrovascular?</h3>
          <p>
            Un accidente cerebrovascular ocurre cuando el suministro de sangre a una parte del cerebro 
            se interrumpe o se reduce, lo que impide que el tejido cerebral reciba oxígeno y nutrientes.
            Las células cerebrales comienzan a morir en minutos.
          </p>
        </div>

        <div className="info-block">
          <h3>Sobre el modelo predictivo</h3>
          <p>
            Nuestro modelo utiliza técnicas avanzadas de aprendizaje automático para evaluar el riesgo 
            de accidente cerebrovascular basándose en múltiples factores de riesgo. El modelo ha sido 
            entrenado con datos reales y validado con métricas de rendimiento robustas.
          </p>
        </div>

        <div className="info-block">
          <h3>Factores de riesgo principales</h3>
          <ul>
            <li>Edad avanzada</li>
            <li>Hipertensión arterial</li>
            <li>Niveles elevados de glucosa</li>
            <li>Enfermedades cardíacas</li>
            <li>Estilo de vida y hábitos</li>
          </ul>
        </div>

        <div className="info-block warning">
          <h3>Importante</h3>
          <p>
            Esta herramienta es solo para fines informativos y educativos. No sustituye el diagnóstico 
            médico profesional. Si tiene preocupaciones sobre su salud, consulte siempre con un 
            profesional de la salud calificado.
          </p>
        </div>
      </div>
    </div>
  );
}

export default InfoSection; 
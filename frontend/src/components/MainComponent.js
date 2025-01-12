import React, { useState } from 'react';
import PredictionForm from './PredictionForm';
import RiskFactors from './RiskFactors';
import PredictionHistory from './PredictionHistory';
import DataInsights from './DataInsights';
import './MainComponent.css';

function MainComponent() {
  const [predictionHistory, setPredictionHistory] = useState(
    JSON.parse(localStorage.getItem('predictionHistory') || '[]')
  );

  const handleNewPrediction = (prediction) => {
    setPredictionHistory([prediction, ...predictionHistory]);
  };

  return (
    <div className="main-container">
      <section className="header-section">
        <div className="header-image-container">
          <img 
            src="https://www.sinakhan.com/media/photos/BrainAneurysm.jpg" 
            alt="Brain Aneurysm Illustration" 
            className="header-image"
          />
        </div>
        <div className="header-content">
          <h1 className="header-title">Stroke Risk Assessment Tool</h1>
          <div className="header-description">
            <p className="main-description">
              Esta herramienta utiliza inteligencia artificial para evaluar tu riesgo de sufrir un accidente cerebrovascular 
              basándose en diversos factores de salud. La detección temprana y la prevención son cruciales para reducir el riesgo.
            </p>
            
            <div className="info-grid">
              <div className="info-card">
                <h3>¿Qué es un Accidente Cerebrovascular?</h3>
                <p>
                  Es una emergencia médica que ocurre cuando se interrumpe o detiene el flujo de sangre al cerebro. 
                  Puede ser isquémico (por bloqueo) o hemorrágico (por sangrado).
                </p>
              </div>

              <div className="info-card warning-signs">
                <h3>Señales de Advertencia (F.A.S.T)</h3>
                <ul>
                  <li><strong>F</strong>ace (Rostro): Caída o entumecimiento de un lado de la cara</li>
                  <li><strong>A</strong>rm (Brazo): Debilidad o entumecimiento en un brazo</li>
                  <li><strong>S</strong>peech (Habla): Dificultad para hablar o entender</li>
                  <li><strong>T</strong>ime (Tiempo): Llame al 911 inmediatamente si nota estos síntomas</li>
                </ul>
              </div>

              <div className="info-card">
                <h3>Factores de Riesgo Principales</h3>
                <ul>
                  <li>Presión arterial alta</li>
                  <li>Diabetes</li>
                  <li>Enfermedades cardíacas</li>
                  <li>Edad avanzada</li>
                  <li>Historial familiar</li>
                </ul>
              </div>

              <div className="info-card">
                <h3>Prevención</h3>
                <ul>
                  <li>Control regular de la presión arterial</li>
                  <li>Dieta saludable y ejercicio</li>
                  <li>No fumar</li>
                  <li>Limitar el consumo de alcohol</li>
                  <li>Mantener niveles saludables de colesterol</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="three-panel-layout">
        <div className="panel left-panel">
          <RiskFactors />
        </div>
        <div className="panel center-panel">
          <PredictionForm onNewPrediction={handleNewPrediction} />
        </div>
        <div className="panel right-panel">
          <PredictionHistory history={predictionHistory} />
        </div>
      </div>

      <div className="insights-section">
        <DataInsights />
      </div>
    </div>
  );
}

export default MainComponent; 
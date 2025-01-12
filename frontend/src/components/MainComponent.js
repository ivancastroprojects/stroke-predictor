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
        <div className="header-content">
          <div className="header-main">
            <div className="header-image-container main-image">
              <img 
                src="https://img.freepik.com/free-photo/male-medical-figure-with-front-brain-highlighted_1048-11823.jpg?t=st=1736675548~exp=1736679148~hmac=4135fd45049d3154d538d5356d0da210b696c391acb614806777882b260b84e6&w=1380" 
                alt="Stroke Awareness Illustration" 
                className="header-image"
              />
            </div>
            <div className="header-text">
              <h1 className="header-title">Stroke Risk Assessment Tool</h1>
              <p className="main-description">
                This tool uses artificial intelligence to assess your risk of having a stroke 
                based on various health factors. Early detection and prevention are crucial 
                for reducing risk.
              </p>
            </div>
          </div>

          <div className="stroke-types-grid">
            <div className="stroke-type-card">
              <img 
                src="https://img.freepik.com/free-vector/human-with-ischemic-stroke_1308-111399.jpg" 
                alt="Ischemic Stroke" 
                className="stroke-type-image"
              />
              <div className="stroke-type-content">
                <h3>Ischemic Stroke</h3>
                <p>Occurs when a blood clot blocks blood flow to the brain. This is the most common type, accounting for 87% of cases.</p>
              </div>
            </div>

            <div className="stroke-type-card">
              <img 
                src="https://img.freepik.com/free-vector/human-anatomy-with-atherosclerosis-stroke_1308-112370.jpg" 
                alt="Atherosclerotic Stroke" 
                className="stroke-type-image"
              />
              <div className="stroke-type-content">
                <h3>Cerebral Atherosclerosis</h3>
                <p>The buildup of plaque in cerebral arteries gradually reduces blood flow, increasing the risk of clots forming.</p>
              </div>
            </div>

            <div className="stroke-type-card">
              <img 
                src="https://img.freepik.com/free-vector/human-with-hemorrhagic-stroke_1308-111966.jpg" 
                alt="Hemorrhagic Stroke" 
                className="stroke-type-image"
              />
              <div className="stroke-type-content">
                <h3>Hemorrhagic Stroke</h3>
                <p>Occurs when a blood vessel ruptures and bleeds into the brain. Although less common, it tends to be more severe.</p>
              </div>
            </div>
          </div>

          <div className="info-grid">
            <div className="info-card warning-signs">
              <h3>Warning Signs (F.A.S.T)</h3>
              <ul>
                <li><strong>F</strong>ace: Drooping or numbness on one side</li>
                <li><strong>A</strong>rm: Weakness or numbness in one arm</li>
                <li><strong>S</strong>peech: Difficulty speaking or understanding</li>
                <li><strong>T</strong>ime: Call emergency services immediately if you notice these symptoms</li>
              </ul>
            </div>

            <div className="info-card">
              <h3>Main Risk Factors</h3>
              <ul>
                <li>High blood pressure</li>
                <li>Diabetes</li>
                <li>Heart disease</li>
                <li>Advanced age</li>
                <li>Family history</li>
              </ul>
            </div>

            <div className="info-card">
              <h3>Prevention</h3>
              <ul>
                <li>Regular blood pressure monitoring</li>
                <li>Healthy diet and exercise</li>
                <li>No smoking</li>
                <li>Limited alcohol consumption</li>
                <li>Maintain healthy cholesterol levels</li>
              </ul>
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
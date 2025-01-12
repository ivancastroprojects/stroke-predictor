import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './PredictionForm.css';

const GLUCOSE_RANGES = {
  low: '< 70 mg/dL',
  normal: '70-140 mg/dL',
  high: '> 140 mg/dL',
  average: '106'
};

const BMI_RANGES = {
  underweight: '< 18.5',
  normal: '18.5-24.9',
  overweight: '25-29.9',
  obese: '≥ 30',
  average: '28.5'
};

function PredictionForm({ onNewPrediction }) {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    hypertension: '0',
    heart_disease: '0',
    ever_married: 'No',
    work_type: 'Private',
    residence_type: 'Urban',
    avg_glucose_level: '',
    bmi: '',
    smoking_status: 'never smoked'
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevState => ({
      ...prevState,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Error en la predicción');
      }

      const result = await response.json();
      
      const predictionData = {
        ...formData,
        prediction: result.prediction,
        timestamp: new Date().toISOString()
      };
      
      const history = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
      const newHistory = [predictionData, ...history];
      localStorage.setItem('predictionHistory', JSON.stringify(newHistory));
      
      if (onNewPrediction) {
        onNewPrediction(predictionData);
      }

      navigate('/results', { 
        state: { 
          prediction: result.prediction,
          formData: formData,
          featureContributions: result.feature_contributions,
          riskFactors: result.risk_factors,
          featureImportance: result.feature_importance
        } 
      });
    } catch (error) {
      console.error('Error:', error);
      alert('Error al realizar la predicción');
    }
  };

  return (
    <div className="prediction-form-container">
      <h2 className="form-title">Stroke Risk Assessment</h2>
      <form className="prediction-form" onSubmit={handleSubmit}>
        <div className="form-section">
          <div className="form-group">
            <label>
              Age <span className="required">*</span>
            </label>
            <input
              type="number"
              name="age"
              value={formData.age}
              onChange={handleChange}
              placeholder="Enter your age"
              min="18"
              max="100"
              required
            />
            <div className="field-info">Must be between 18 and 100 years</div>
          </div>

          <div className="form-group">
            <label>
              Gender <span className="required">*</span>
            </label>
            <select
              name="gender"
              value={formData.gender}
              onChange={handleChange}
              required
            >
              <option value="">Select gender</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </div>

          <div className="form-group">
            <label>
              Hypertension <span className="required">*</span>
            </label>
            <select
              name="hypertension"
              value={formData.hypertension}
              onChange={handleChange}
              required
            >
              <option value="">Select option</option>
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
            <div className="field-info">History of high blood pressure</div>
          </div>

          <div className="form-group">
            <label>
              Heart Disease <span className="required">*</span>
            </label>
            <select
              name="heart_disease"
              value={formData.heart_disease}
              onChange={handleChange}
              required
            >
              <option value="">Select option</option>
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
            <div className="field-info">History of heart conditions</div>
          </div>

          <div className="form-group">
            <label>
              Average Glucose Level <span className="required">*</span>
            </label>
            <input
              type="number"
              name="avg_glucose_level"
              value={formData.avg_glucose_level}
              onChange={handleChange}
              placeholder="Enter glucose level"
              step="0.1"
              min="50"
              max="300"
              required
            />
            <div className="range-info">
              <span className="range-item low">
                <span className="range-dot low"></span>
                Low: &lt;70
              </span>
              <span className="range-item normal">
                <span className="range-dot normal"></span>
                Normal: 70-140
              </span>
              <span className="range-item high">
                <span className="range-dot high"></span>
                High: &gt;140
              </span>
            </div>
            <div className="field-info">Measured in mg/dL</div>
          </div>

          <div className="form-group">
            <label>
              BMI <span className="required">*</span>
            </label>
            <input
              type="number"
              name="bmi"
              value={formData.bmi}
              onChange={handleChange}
              placeholder="Enter BMI"
              step="0.1"
              min="15"
              max="50"
              required
            />
            <div className="range-info">
              <span className="range-item low">
                <span className="range-dot low"></span>
                Underweight: &lt;18.5
              </span>
              <span className="range-item normal">
                <span className="range-dot normal"></span>
                Normal: 18.5-24.9
              </span>
              <span className="range-item high">
                <span className="range-dot high"></span>
                Overweight: &gt;25
              </span>
            </div>
            <div className="field-info">Body Mass Index (kg/m²)</div>
          </div>

          <button type="submit" className="submit-button">
            Calculate Risk
          </button>
        </div>
      </form>
    </div>
  );
}

export default PredictionForm; 
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
          <div className="form-group age">
            <label htmlFor="age">
              Age
              <span className="required">*</span>
            </label>
            <input
              type="number"
              id="age"
              name="age"
              value={formData.age}
              onChange={handleChange}
              required
              min="0"
              max="120"
              placeholder="Enter age"
            />
            <span className="field-info">Between 0 and 120 years</span>
          </div>

          <div className="form-group gender">
            <label htmlFor="gender">
              Gender
              <span className="required">*</span>
            </label>
            <select
              id="gender"
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
            <label htmlFor="residence_type">Residence Type</label>
            <select
              id="residence_type"
              name="residence_type"
              value={formData.residence_type}
              onChange={handleChange}
            >
              <option value="Urban">Urban</option>
              <option value="Rural">Rural</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="hypertension">
              Hypertension
              <span className="required">*</span>
            </label>
            <select
              id="hypertension"
              name="hypertension"
              value={formData.hypertension}
              onChange={handleChange}
              required
            >
              <option value="0">No</option>
              <option value="1">Yes</option>
            </select>
            <span className="field-info">Have you been diagnosed with hypertension?</span>
          </div>

          <div className="form-group">
            <label htmlFor="heart_disease">
              Heart Disease
              <span className="required">*</span>
            </label>
            <select
              id="heart_disease"
              name="heart_disease"
              value={formData.heart_disease}
              onChange={handleChange}
              required
            >
              <option value="0">No</option>
              <option value="1">Yes</option>
            </select>
            <span className="field-info">Have you been diagnosed with any heart disease?</span>
          </div>

          <div className="form-group glucose">
            <label htmlFor="avg_glucose_level">Average Glucose Level (mg/dL)</label>
            <input
              type="number"
              id="avg_glucose_level"
              name="avg_glucose_level"
              className="form-control"
              value={formData.avg_glucose_level}
              onChange={handleChange}
              placeholder={`Average: ${GLUCOSE_RANGES.average} mg/dL`}
              required
            />
            <div className="range-info">
              <div className="range-item low">
                <span className="range-dot low"></span>
                <span>Low: {GLUCOSE_RANGES.low}</span>
              </div>
              <div className="range-item normal">
                <span className="range-dot normal"></span>
                <span>Normal: {GLUCOSE_RANGES.normal}</span>
              </div>
              <div className="range-item high">
                <span className="range-dot high"></span>
                <span>High: {GLUCOSE_RANGES.high}</span>
              </div>
            </div>
          </div>

          <div className="form-group bmi">
            <label htmlFor="bmi">Body Mass Index (BMI)</label>
            <input
              type="number"
              id="bmi"
              name="bmi"
              className="form-control"
              value={formData.bmi}
              onChange={handleChange}
              placeholder={`Average: ${BMI_RANGES.average}`}
              required
            />
            <div className="range-info">
              <div className="range-item low">
                <span className="range-dot low"></span>
                <span>Underweight: {BMI_RANGES.underweight}</span>
              </div>
              <div className="range-item normal">
                <span className="range-dot normal"></span>
                <span>Normal: {BMI_RANGES.normal}</span>
              </div>
              <div className="range-item high">
                <span className="range-dot high"></span>
                <span>Overweight: {BMI_RANGES.overweight}</span>
              </div>
            </div>
            <div className="helper-text">
              BMI = weight(kg) / height²(m)
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="work_type">Work Type</label>
            <select
              id="work_type"
              name="work_type"
              value={formData.work_type}
              onChange={handleChange}
            >
              <option value="Private">Private</option>
              <option value="Self-employed">Self-employed</option>
              <option value="Govt_job">Government</option>
              <option value="children">Child</option>
              <option value="Never_worked">Never worked</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="smoking_status">Smoking Status</label>
            <select
              id="smoking_status"
              name="smoking_status"
              value={formData.smoking_status}
              onChange={handleChange}
            >
              <option value="never smoked">Never smoked</option>
              <option value="formerly smoked">Former smoker</option>
              <option value="smokes">Current smoker</option>
              <option value="Unknown">Unknown</option>
            </select>
          </div>

          <button type="submit" className="submit-button">
            Make Prediction
          </button>
        </div>
      </form>
    </div>
  );
}

export default PredictionForm; 
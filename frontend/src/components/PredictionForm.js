import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import BackgroundRays from './BackgroundRays';
import './PredictionForm.css';

function PredictionForm({ onNewPrediction }) {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    hypertension: '0',
    heart_disease: '0',
    ever_married: 'No',
    work_type: 'Private',
    Residence_type: 'Urban',
    avg_glucose_level: '',
    bmi: '',
    smoking_status: 'never smoked'
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevState => ({
      ...prevState,
      [name]: value
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsSubmitting(true);
    setSubmitError(null);
    
    try {
        const apiUrl = process.env.NODE_ENV === 'production' 
            ? '/api/predict'
            : `${process.env.REACT_APP_API_URL}/api/predict`;
            
        console.log('Enviando predicción a:', apiUrl);
        
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(formData),
            credentials: 'include',
            mode: 'cors'
        });

        if (!response.ok) {
            throw new Error(`Error HTTP: ${response.status}`);
        }

        const data = await response.json();
        console.log('Respuesta recibida:', data);

        const predictionData = {
            ...formData,
            prediction: data.probability,
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
                prediction: data.probability,
                formData: formData,
                featureContributions: data.feature_contributions,
                riskFactors: data.risk_factors,
                featureImportance: data.feature_importance
            }
        });
    } catch (err) {
        console.error('Error al hacer la predicción:', err);
        setSubmitError(err.message);
    } finally {
        setIsSubmitting(false);
    }
  };

  return (
    <>
      <BackgroundRays />
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
                Ever Married <span className="required">*</span>
              </label>
              <select
                name="ever_married"
                value={formData.ever_married}
                onChange={handleChange}
                required
              >
                <option value="">Select option</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>

            <div className="form-group">
              <label>
                Work Type <span className="required">*</span>
              </label>
              <select
                name="work_type"
                value={formData.work_type}
                onChange={handleChange}
                required
              >
                <option value="">Select work type</option>
                <option value="Private">Private</option>
                <option value="Self-employed">Self-employed</option>
                <option value="Govt_job">Government Job</option>
                <option value="children">Children</option>
                <option value="Never_worked">Never worked</option>
              </select>
            </div>

            <div className="form-group">
              <label>
                Residence Type <span className="required">*</span>
              </label>
              <select
                name="Residence_type"
                value={formData.Residence_type}
                onChange={handleChange}
                required
              >
                <option value="">Select residence type</option>
                <option value="Urban">Urban</option>
                <option value="Rural">Rural</option>
              </select>
            </div>

            <div className="form-group">
              <label>
                Smoking Status <span className="required">*</span>
              </label>
              <select
                name="smoking_status"
                value={formData.smoking_status}
                onChange={handleChange}
                required
              >
                <option value="">Select smoking status</option>
                <option value="formerly smoked">Formerly smoked</option>
                <option value="never smoked">Never smoked</option>
                <option value="smokes">Currently smokes</option>
                <option value="Unknown">Unknown</option>
              </select>
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
    </>
  );
}

export default PredictionForm; 
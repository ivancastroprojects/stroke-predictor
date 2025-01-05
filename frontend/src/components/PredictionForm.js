import React, { useState } from 'react';

export default function PredictionForm({ onSubmit }) {
  const [formData, setFormData] = useState({
    gender: "",
    age: "",
    hypertension: "",
    heart_disease: "",
    ever_married: "",
    work_type: "",
    Residence_type: "",
    avg_glucose_level: "",
    bmi: "",
    smoking_status: "",
  });

  const fieldRanges = {
    avg_glucose_level: {
      placeholder: "Ej: 106.15",
      min: 55,
      max: 271,
      info: "Rango normal: 70-140 mg/dL"
    },
    bmi: {
      placeholder: "Ej: 28.89",
      min: 10,
      max: 60,
      info: "Rango saludable: 18.5-24.9"
    },
    age: {
      placeholder: "Ej: 43",
      min: 0,
      max: 100,
      info: "Edad en años"
    }
  };

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    const dataToSubmit = {
      gender: formData.gender || "",
      age: formData.age || "",
      hypertension: formData.hypertension || "",
      heart_disease: formData.heart_disease || "",
      ever_married: formData.ever_married || "No",
      work_type: formData.work_type || "",
      Residence_type: formData.Residence_type || "Urban",
      avg_glucose_level: formData.avg_glucose_level || 106.15,
      bmi: formData.bmi || 28.89,
      smoking_status: formData.smoking_status || "",
    };

    if (!dataToSubmit.age || !dataToSubmit.gender || !dataToSubmit.hypertension || !dataToSubmit.heart_disease || !dataToSubmit.smoking_status || !dataToSubmit.work_type) {
      alert("Por favor, complete todos los campos obligatorios.");
      return;
    }

    onSubmit(dataToSubmit);
  };

  return (
    <div className="form-container">
      <h2>Predicción de Riesgo de Stroke</h2>
      <form onSubmit={handleSubmit}>
        {/* Age */}
        <div className="form-group">
          <label>
            Edad: <span className="required">*</span>
            <input
              type="number"
              name="age"
              onChange={handleChange}
              placeholder={fieldRanges.age.placeholder}
              min={fieldRanges.age.min}
              max={fieldRanges.age.max}
              title={fieldRanges.age.info}
              required
            />
            <span className="field-info">{fieldRanges.age.info}</span>
          </label>
        </div>

        {/* Gender */}
        <div className="form-group">
          <label>
            Género: <span className="required">*</span>
            <select name="gender" onChange={handleChange} required>
              <option value="">Seleccionar Género</option>
              <option value="Male">Masculino</option>
              <option value="Female">Femenino</option>
              <option value="Other">Otro</option>
            </select>
          </label>
        </div>

        {/* Hypertension */}
        <div className="form-group">
          <label>
            Hipertensión: <span className="required">*</span>
            <select name="hypertension" onChange={handleChange} required>
              <option value="">Seleccionar</option>
              <option value="1">Sí</option>
              <option value="0">No</option>
            </select>
          </label>
        </div>

        {/* Heart Disease */}
        <div className="form-group">
          <label>
            Enfermedad Cardíaca: <span className="required">*</span>
            <select name="heart_disease" onChange={handleChange} required>
              <option value="">Seleccionar</option>
              <option value="1">Sí</option>
              <option value="0">No</option>
            </select>
          </label>
        </div>

        {/* Smoking Status */}
        <div className="form-group">
          <label>
            Hábito de Fumar: <span className="required">*</span>
            <select name="smoking_status" onChange={handleChange} required>
              <option value="">Seleccionar</option>
              <option value="formerly smoked">Ex fumador</option>
              <option value="never smoked">Nunca fumó</option>
              <option value="smokes">Fumador actual</option>
              <option value="Unknown">Desconocido</option>
            </select>
          </label>
        </div>

        {/* Work Type */}
        <div className="form-group">
          <label>
            Tipo de Trabajo: <span className="required">*</span>
            <select name="work_type" onChange={handleChange} required>
              <option value="">Seleccionar</option>
              <option value="Private">Privado</option>
              <option value="Self-employed">Autónomo</option>
              <option value="Govt_job">Gobierno</option>
              <option value="children">Estudiante</option>
              <option value="Never_worked">Sin trabajo</option>
            </select>
          </label>
        </div>

        {/* Residence Type */}
        <div className="form-group">
          <label>
            Tipo de Residencia:
            <select name="Residence_type" onChange={handleChange}>
              <option value="">Seleccionar</option>
              <option value="Urban">Urbana</option>
              <option value="Rural">Rural</option>
            </select>
          </label>
        </div>

        {/* Ever Married */}
        <div className="form-group">
          <label>
            Estado Civil:
            <select name="ever_married" onChange={handleChange} required>
              <option value="">Seleccionar</option>
              <option value="Yes">Casado/a</option>
              <option value="No">Soltero/a</option>
            </select>
          </label>
        </div>

        {/* Glucose Level */}
        <div className="form-group">
          <label>
            Nivel de Glucosa:
            <input
              type="number"
              name="avg_glucose_level"
              onChange={handleChange}
              placeholder={fieldRanges.avg_glucose_level.placeholder}
              min={fieldRanges.avg_glucose_level.min}
              max={fieldRanges.avg_glucose_level.max}
              step="0.01"
              title={fieldRanges.avg_glucose_level.info}
              required
            />
            <span className="field-info">{fieldRanges.avg_glucose_level.info}</span>
          </label>
        </div>

        {/* BMI */}
        <div className="form-group">
          <label>
            IMC:
            <input
              type="number"
              name="bmi"
              onChange={handleChange}
              placeholder={fieldRanges.bmi.placeholder}
              min={fieldRanges.bmi.min}
              max={fieldRanges.bmi.max}
              step="0.01"
              title={fieldRanges.bmi.info}
              required
            />
            <span className="field-info">{fieldRanges.bmi.info}</span>
          </label>
        </div>

        <button type="submit" className="submit-button">
          Analizar Riesgo
        </button>
      </form>
    </div>
  );
} 
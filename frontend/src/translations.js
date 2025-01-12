const translations = {
  en: {
    // Stroke Info Section
    strokeTitle: "What is a Stroke?",
    strokeDescription: "A stroke occurs when blood flow to part of the brain is blocked or when a blood vessel in the brain bursts. In either case, parts of the brain become damaged or die. A stroke can cause lasting brain damage, long-term disability, or even death.",
    
    // Stroke Types
    strokeTypes: {
      ischemicTitle: "Ischemic Stroke (Most Common)",
      ischemicDesc: "Occurs when blood vessels to the brain become narrowed or blocked, causing severely reduced blood flow.",
      hemorrhagicTitle: "Hemorrhagic Stroke",
      hemorrhagicDesc: "Happens when an artery in the brain leaks blood or ruptures, causing pressure that damages brain cells.",
      tiaTitle: "TIA (Mini-stroke)",
      tiaDesc: "A temporary blockage of blood flow to the brain, serving as a serious warning sign of stroke risk."
    },

    // Warning Signs
    warningTitle: "Remember F.A.S.T. Warning Signs:",
    warningSigns: {
      f: "Face Drooping",
      a: "Arm Weakness",
      s: "Speech Difficulty",
      t: "Time to Call Emergency"
    },

    // Form Sections
    formTitle: "Stroke Risk Assessment",
    formSections: {
      basic: "Basic Data",
      health: "Health",
      lifestyle: "Lifestyle"
    },

    // Form Fields
    formFields: {
      age: {
        label: "Age",
        info: "Between 0 and 120 years"
      },
      gender: {
        label: "Gender",
        options: {
          select: "Select gender",
          male: "Male",
          female: "Female"
        }
      },
      residence: {
        label: "Residence Type",
        options: {
          urban: "Urban",
          rural: "Rural"
        }
      },
      hypertension: {
        label: "Hypertension",
        info: "Have you been diagnosed with hypertension?",
        options: {
          yes: "Yes",
          no: "No"
        }
      },
      heartDisease: {
        label: "Heart Disease",
        info: "Have you been diagnosed with any heart disease?",
        options: {
          yes: "Yes",
          no: "No"
        }
      },
      glucose: {
        label: "Glucose Level",
        info: "Average blood glucose level (mg/dL)"
      },
      bmi: {
        label: "BMI",
        info: "Body Mass Index (kg/m²)"
      },
      workType: {
        label: "Work Type",
        options: {
          private: "Private",
          selfEmployed: "Self-employed",
          government: "Government",
          children: "Children",
          neverWorked: "Never worked"
        }
      },
      smoking: {
        label: "Smoking Status",
        options: {
          never: "Never smoked",
          former: "Former smoker",
          smokes: "Smoker",
          unknown: "Unknown"
        }
      }
    },

    // Buttons
    submitButton: "Make Prediction",

    // Results
    results: {
      title: "Analysis Results",
      subtitle: "Analysis based on provided risk factors",
      riskLevel: {
        title: "Risk Level",
        high: "High",
        moderate: "Moderate",
        low: "Low"
      },
      factorsContribution: "Factors Contribution",
      mainRiskFactors: "Main Risk Factors",
      riskIndicator: "Risk"
    },

    // Risk Factors
    riskFactors: {
      title: "Risk Factors",
      age: {
        name: "Age",
        desc: "Risk increases significantly with age."
      },
      hypertension: {
        name: "Hypertension",
        desc: "High blood pressure is a major risk factor."
      },
      residence: {
        name: "Residence",
        desc: "Type of residence can influence access to medical care."
      },
      glucose: {
        name: "Glucose Level",
        desc: "Elevated glucose levels increase risk."
      },
      bmi: {
        name: "BMI",
        desc: "Body Mass Index affects cardiovascular risk."
      }
    },

    // History
    history: {
      title: "History",
      noHistory: "No predictions recorded",
      details: {
        age: "Age",
        gender: "Gender",
        bmi: "BMI",
        glucose: "Glucose",
        hta: "HBP",
        card: "Heart D."
      }
    }
  },
  es: {
    // Sección de Información sobre ACV
    strokeTitle: "¿Qué es un ACV?",
    strokeDescription: "Un ACV ocurre cuando el flujo sanguíneo a una parte del cerebro se bloquea o cuando un vaso sanguíneo en el cerebro se rompe. En cualquier caso, partes del cerebro se dañan o mueren. Un ACV puede causar daño cerebral permanente, discapacidad a largo plazo o incluso la muerte.",
    
    // Tipos de ACV
    strokeTypes: {
      ischemicTitle: "ACV Isquémico (Más Común)",
      ischemicDesc: "Ocurre cuando los vasos sanguíneos del cerebro se estrechan o bloquean, causando una reducción severa del flujo sanguíneo.",
      hemorrhagicTitle: "ACV Hemorrágico",
      hemorrhagicDesc: "Sucede cuando una arteria en el cerebro tiene una fuga o se rompe, causando presión que daña las células cerebrales.",
      tiaTitle: "AIT (Mini-ACV)",
      tiaDesc: "Un bloqueo temporal del flujo sanguíneo al cerebro, que sirve como una seria señal de advertencia de riesgo de ACV."
    },

    // Señales de Advertencia
    warningTitle: "Recuerda las Señales de Advertencia R.Á.P.I.D.O:",
    warningSigns: {
      f: "Rostro Caído",
      a: "Alteración del Habla",
      s: "Pérdida de Fuerza",
      t: "Inmediatamente al 112"
    },

    // Secciones del Formulario
    formTitle: "Evaluación de Riesgo de ACV",
    formSections: {
      basic: "Datos Básicos",
      health: "Salud",
      lifestyle: "Estilo de Vida"
    },

    // Campos del Formulario
    formFields: {
      age: {
        label: "Edad",
        info: "Entre 0 y 120 años"
      },
      gender: {
        label: "Género",
        options: {
          select: "Seleccione el género",
          male: "Masculino",
          female: "Femenino"
        }
      },
      residence: {
        label: "Tipo de Residencia",
        options: {
          urban: "Urbana",
          rural: "Rural"
        }
      },
      hypertension: {
        label: "Hipertensión",
        info: "¿Ha sido diagnosticado con hipertensión?",
        options: {
          yes: "Sí",
          no: "No"
        }
      },
      heartDisease: {
        label: "Enfermedad Cardíaca",
        info: "¿Ha sido diagnosticado con alguna enfermedad cardíaca?",
        options: {
          yes: "Sí",
          no: "No"
        }
      },
      glucose: {
        label: "Nivel de Glucosa",
        info: "Nivel promedio de glucosa en sangre (mg/dL)"
      },
      bmi: {
        label: "IMC",
        info: "Índice de Masa Corporal (kg/m²)"
      },
      workType: {
        label: "Tipo de Trabajo",
        options: {
          private: "Privado",
          selfEmployed: "Autónomo",
          government: "Gobierno",
          children: "Niño",
          neverWorked: "Nunca trabajó"
        }
      },
      smoking: {
        label: "Estado de Fumador",
        options: {
          never: "Nunca fumó",
          former: "Ex fumador",
          smokes: "Fumador",
          unknown: "Desconocido"
        }
      }
    },

    // Botones
    submitButton: "Realizar Predicción",

    // Resultados
    results: {
      title: "Resultados del Análisis",
      subtitle: "Análisis basado en los factores de riesgo proporcionados",
      riskLevel: {
        title: "Nivel de Riesgo",
        high: "Alto",
        moderate: "Moderado",
        low: "Bajo"
      },
      factorsContribution: "Contribución de Factores",
      mainRiskFactors: "Factores de Riesgo Principales",
      riskIndicator: "Riesgo"
    },

    // Factores de Riesgo
    riskFactors: {
      title: "Factores de Riesgo",
      age: {
        name: "Edad",
        desc: "El riesgo aumenta significativamente con la edad."
      },
      hypertension: {
        name: "Hipertensión",
        desc: "La presión arterial alta es un factor de riesgo importante."
      },
      residence: {
        name: "Residencia",
        desc: "El tipo de residencia puede influir en el acceso a atención médica."
      },
      glucose: {
        name: "Nivel de glucosa",
        desc: "Niveles elevados de glucosa aumentan el riesgo."
      },
      bmi: {
        name: "IMC",
        desc: "El índice de masa corporal afecta el riesgo cardiovascular."
      }
    },

    // Historial
    history: {
      title: "Historial",
      noHistory: "No hay predicciones registradas",
      details: {
        age: "Edad",
        gender: "Género",
        bmi: "IMC",
        glucose: "Glucosa",
        hta: "HTA",
        card: "Card."
      }
    }
  }
};

export default translations; 
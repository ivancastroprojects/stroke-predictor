import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import dump
import seaborn as sns
import os
import shutil
from datetime import datetime
import logging

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (
    train_test_split, 
    RandomizedSearchCV, 
    StratifiedKFold, 
    cross_validate
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    make_scorer, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score, 
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from scipy.stats import uniform
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMBPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel

# Configuración global de matplotlib
plt.style.use('default')
sns.set_theme(style="whitegrid")
plt.ioff()  # Desactivar modo interactivo

# Constantes globales
RANDOM_STATE = 42
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
TEST_SIZE = 0.2
CV_SPLITS = 10
CV_REPEATS = 3
N_ITER_SEARCH = 50

# Asegurar que el directorio models existe
os.makedirs(MODELS_DIR, exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(MODELS_DIR, 'training.log'), mode='w'),
        logging.StreamHandler()
    ]
)

# Configuración de modelos
MODEL_CONFIGS = {
    'Random Forest': {
        'model': RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced_subsample',
            criterion='entropy',
            min_samples_leaf=3,
            bootstrap=True,
            oob_score=True
        ),
        'params': {
            'model__n_estimators': [800, 1000, 1200],
            'model__max_depth': [8, 10, 12],
            'model__min_samples_split': [3, 5, 7],
            'model__max_features': ['sqrt', 'log2'],
            'model__max_samples': [0.8, 0.9],
            'smote__k_neighbors': [3, 5]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            validation_fraction=0.2,
            n_iter_no_change=15,
            subsample=0.8,
            max_features='sqrt',
            init='zero'
        ),
        'params': {
            'model__n_estimators': [300, 500, 700],
            'model__learning_rate': [0.01, 0.03, 0.05],
            'model__max_depth': [4, 5, 6],
            'model__min_samples_split': [5, 7, 10],
            'model__min_samples_leaf': [3, 4, 5],
            'smote__k_neighbors': [3, 5]
        }
    },
    'Ensemble': {
        'model': None,  # Se creará usando create_ensemble_pipeline
        'params': {
            'smote__k_neighbors': [3, 5]
        }
    }
}

class CalibratedEnsembleClassifier:
    """Ensemble personalizado que combina Random Forest y Gradient Boosting con calibración"""
    def __init__(self, rf_model, gb_model, weights=[0.6, 0.4]):
        self.rf_model = rf_model
        self.gb_model = gb_model
        self.weights = weights
        self.calibrated_rf = None
        self.calibrated_gb = None
        self.classes_ = None
        
    def fit(self, X, y):
        # Guardar las clases únicas
        self.classes_ = np.unique(y)
        
        # Entrenar y calibrar Random Forest
        self.calibrated_rf = CalibratedClassifierCV(
            self.rf_model, cv=5, method='sigmoid', n_jobs=-1
        )
        self.calibrated_rf.fit(X, y)
        
        # Entrenar y calibrar Gradient Boosting
        self.calibrated_gb = CalibratedClassifierCV(
            self.gb_model, cv=5, method='sigmoid', n_jobs=-1
        )
        self.calibrated_gb.fit(X, y)
        
        return self
    
    def predict_proba(self, X):
        # Obtener probabilidades calibradas
        rf_proba = self.calibrated_rf.predict_proba(X)
        gb_proba = self.calibrated_gb.predict_proba(X)
        
        # Combinar predicciones usando los pesos
        return (
            self.weights[0] * rf_proba +
            self.weights[1] * gb_proba
        )
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)
    
    def get_params(self, deep=True):
        """Necesario para ser compatible con sklearn"""
        return {
            'rf_model': self.rf_model,
            'gb_model': self.gb_model,
            'weights': self.weights
        }
    
    def set_params(self, **parameters):
        """Necesario para ser compatible con sklearn"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def preprocess_data(df):
    """
    Preprocesa los datos aplicando transformaciones y creando nuevas características.
    Retorna un DataFrame con todas las características procesadas.
    """
    df = df.copy()
    
    # 1. Manejo de valores nulos y outliers primero
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df['bmi'] = np.clip(df['bmi'], 10, 50)  # Limitar BMI a rangos realistas
    df['avg_glucose_level'] = np.clip(df['avg_glucose_level'], 50, 300)
    
    # 2. Normalización de características numéricas continuas
    continuous_features = ['age', 'bmi', 'avg_glucose_level']
    for feature in continuous_features:
        mean = df[feature].mean()
        std = df[feature].std()
        df[f'{feature}_normalized'] = (df[feature] - mean) / std
    
    # 3. Características de edad más detalladas
    df['age_squared'] = df['age'] ** 2
    df['age_cubed'] = df['age'] ** 3
    df['elderly'] = (df['age'] >= 65).astype(int)
    df['very_elderly'] = (df['age'] >= 80).astype(int)
    df['young_adult'] = ((df['age'] >= 18) & (df['age'] < 35)).astype(int)
    df['middle_aged'] = ((df['age'] >= 35) & (df['age'] < 65)).astype(int)
    
    # 4. Características de riesgo médico más detalladas
    df['high_glucose'] = (df['avg_glucose_level'] > 200).astype(int)
    df['very_high_glucose'] = (df['avg_glucose_level'] > 250).astype(int)
    df['pre_diabetes'] = ((df['avg_glucose_level'] >= 140) & (df['avg_glucose_level'] <= 199)).astype(int)
    df['obesity'] = (df['bmi'] >= 30).astype(int)
    df['severe_obesity'] = (df['bmi'] >= 35).astype(int)
    df['morbid_obesity'] = (df['bmi'] >= 40).astype(int)
    df['underweight'] = (df['bmi'] < 18.5).astype(int)
    
    # 5. Interacciones médicas más completas
    # Edad y condiciones médicas
    df['age_glucose'] = df['age'] * df['high_glucose']
    df['age_pressure'] = df['age'] * df['hypertension']
    df['age_heart'] = df['age'] * df['heart_disease']
    df['elderly_glucose'] = df['elderly'] * df['high_glucose']
    df['elderly_pressure'] = df['elderly'] * df['hypertension']
    df['elderly_heart'] = df['elderly'] * df['heart_disease']
    
    # Interacciones entre condiciones médicas
    df['glucose_pressure'] = df['high_glucose'] * df['hypertension']
    df['glucose_heart'] = df['high_glucose'] * df['heart_disease']
    df['pressure_heart'] = df['hypertension'] * df['heart_disease']
    df['obesity_glucose'] = df['obesity'] * df['high_glucose']
    df['obesity_pressure'] = df['obesity'] * df['hypertension']
    df['obesity_heart'] = df['obesity'] * df['heart_disease']
    
    # 6. Score de riesgo compuesto más detallado
    df['risk_score'] = (
        df['elderly'].astype(int) * 2 +  # Mayor peso para edad avanzada
        df['very_elderly'].astype(int) * 3 +  # Aún más peso para muy ancianos
        df['high_glucose'].astype(int) * 2 +
        df['very_high_glucose'].astype(int) * 3 +
        df['obesity'].astype(int) +
        df['severe_obesity'].astype(int) * 2 +
        df['morbid_obesity'].astype(int) * 3 +
        df['hypertension'].astype(int) * 2 +
        df['heart_disease'].astype(int) * 2 +
        df['glucose_pressure'].astype(int) +
        df['glucose_heart'].astype(int) +
        df['pressure_heart'].astype(int)
    )
    
    # 7. Codificación de categóricas binarias
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 0})
    df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
    df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})
    
    # Verificar que todas las transformaciones se aplicaron correctamente
    expected_columns = set([
        # Columnas originales
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
        'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
        'smoking_status', 'stroke',
        
        # Columnas normalizadas
        'age_normalized', 'bmi_normalized', 'avg_glucose_level_normalized',
        
        # Características derivadas de edad
        'age_squared', 'age_cubed', 'elderly', 'very_elderly', 'young_adult', 'middle_aged',
        
        # Características de riesgo médico
        'high_glucose', 'very_high_glucose', 'pre_diabetes',
        'obesity', 'severe_obesity', 'morbid_obesity', 'underweight',
        
        # Interacciones
        'age_glucose', 'age_pressure', 'age_heart',
        'elderly_glucose', 'elderly_pressure', 'elderly_heart',
        'glucose_pressure', 'glucose_heart', 'pressure_heart',
        'obesity_glucose', 'obesity_pressure', 'obesity_heart',
        
        # Score compuesto
        'risk_score'
    ])
    
    missing_cols = expected_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan las siguientes columnas después del preprocesamiento: {missing_cols}")
    
    return df

def load_data():
    # Usar ruta relativa desde el directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'healthcare-dataset-stroke-data.csv')
    
    df = pd.read_csv(csv_path)
    df = df.drop('id', axis=1)
    df = preprocess_data(df)
    
    # Imprimir todas las columnas disponibles
    logging.info("Columnas disponibles en el DataFrame:")
    logging.info(df.columns.tolist())
    
    # Definir columnas categóricas que necesitan codificación one-hot
    categorical = ['work_type', 'smoking_status']
    
    # Definir columnas numéricas en el orden correcto
    numerical = [
        # Características base normalizadas
        'age_normalized',
        'bmi_normalized',
        'avg_glucose_level_normalized',
        
        # Características base binarias
        'hypertension',
        'heart_disease',
        'gender',
        'ever_married',
        'Residence_type',
        
        # Características derivadas de edad
        'age_squared',
        'age_cubed',
        'elderly',
        'very_elderly',
        'young_adult',
        'middle_aged',
        
        # Características de riesgo médico
        'high_glucose',
        'very_high_glucose',
        'pre_diabetes',
        'obesity',
        'severe_obesity',
        'morbid_obesity',
        'underweight',
        
        # Interacciones edad-condiciones
        'age_glucose',
        'age_pressure',
        'age_heart',
        'elderly_glucose',
        'elderly_pressure',
        'elderly_heart',
        
        # Interacciones entre condiciones
        'glucose_pressure',
        'glucose_heart',
        'pressure_heart',
        'obesity_glucose',
        'obesity_pressure',
        'obesity_heart',
        
        # Score compuesto
        'risk_score'
    ]
    
    # Verificar que todas las columnas existen
    missing_cols = [col for col in numerical + categorical if col not in df.columns]
    if missing_cols:
        logging.error(f"Columnas faltantes en el DataFrame: {missing_cols}")
        raise ValueError(f"Columnas faltantes en el DataFrame: {missing_cols}")
    
    y = df['stroke']
    X = df[categorical + numerical]
    
    return X, y, categorical, numerical

def evaluate_model(X_test, y_test, model):
    """Evaluación detallada del modelo con múltiples métricas"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision'
    }
    
    scores = cross_validate(
        model, X_test, y_test,
        scoring=scoring,
        cv=cv,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Logging detallado de métricas
    for metric in scoring.keys():
        train_score = np.mean(scores[f'train_{metric}'])
        test_score = np.mean(scores[f'test_{metric}'])
        logging.info(f"{metric.upper()}: Train={train_score:.3f}, Test={test_score:.3f}")
    
    return {k.replace('test_', ''): scores[k] for k in scores if k.startswith('test_')}

def analyze_feature_importance(model, X, numerical):
    feature_importance = {}
    base_model = model.named_steps['model']
    
    # Obtener nombres de características después de la transformación
    feature_names = []
    for name, transformer in model.named_steps['preprocessor'].named_transformers_.items():
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out())
        else:
            feature_names.extend(X[numerical].columns)
    
    # Obtener importancias
    if hasattr(base_model, 'feature_importances_'):
        importances = base_model.feature_importances_
        
        # Agrupar importancias relacionadas con la edad
        age_related_importance = sum(
            importances[i] for i, name in enumerate(feature_names)
            if any(term in name.lower() for term in ['age', 'elderly', 'age_group'])
        )
        
        # Normalizar y guardar
        total_importance = np.sum(importances)
        feature_importance['Age_Related'] = age_related_importance / total_importance * 100
        
        # Guardar resto de características
        for idx, feature in enumerate(feature_names):
            if not any(term in feature.lower() for term in ['age', 'elderly', 'age_group']):
                importance = (importances[idx] / total_importance) * 100
                feature_importance[feature] = importance
    
    return feature_importance

def get_optimized_model(name, X, y, categorical, numerical):
    """Obtiene el modelo optimizado según la configuración especificada"""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Modelo {name} no encontrado en la configuración")
        
    config = MODEL_CONFIGS[name]
    
    # Crear pipeline según el tipo de modelo
    if name == 'Ensemble':
        pipeline = create_ensemble_pipeline(categorical, numerical)
    else:
        pipeline = create_base_pipeline(categorical, numerical, config['model'])
    
    # Ajustar los nombres de los parámetros
    param_grid = {}
    for param, values in config['params'].items():
        param_grid[param] = values
    
    # Crear validación cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Configurar búsqueda aleatoria
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=N_ITER_SEARCH,
        cv=cv,
        scoring=['roc_auc', 'f1', 'precision', 'recall'],
        refit='f1',  # Optimizar para F1-score
        n_jobs=-1,
        random_state=RANDOM_STATE,
        error_score='raise'
    )
    
    print(f"\nOptimizando {name}...")
    random_search.fit(X, y)
    print(f"Mejor score: {random_search.best_score_:.3f}")
    
    return random_search.best_estimator_

def create_base_pipeline(categorical_features, numerical_features, model=None):
    """Pipeline que preserva la importancia de características clave e incluye SMOTE"""
    # Separar características normalizadas
    normalized_features = [f for f in numerical_features if f.endswith('_normalized')]
    other_numerical = [f for f in numerical_features if not f.endswith('_normalized')]
    
    # Crear transformadores
    transformers = []
    
    # Agregar transformador para características ya normalizadas
    if normalized_features:
        transformers.append(('normalized', 'passthrough', normalized_features))
    
    # Agregar transformador para otras características numéricas
    if other_numerical:
        transformers.append(('num', StandardScaler(), other_numerical))
    
    # Agregar transformador para características categóricas
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))
    
    # Crear el preprocessor
    preprocessor = ColumnTransformer(transformers=transformers)
    
    if model is not None:
        return IMBPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.5)),
            ('model', model)
        ])

    return preprocessor

def create_lda_pipeline(categorical, numerical):
    return create_base_pipeline(
        categorical_features=categorical,
        numerical_features=numerical,
        model=LinearDiscriminantAnalysis()
    )

def evaluate_multiple_models(X, y, categorical, numerical):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    results = {}
    
    for name, config in MODEL_CONFIGS.items():
        logging.info(f"\nEntrenando {name}...")
        try:
            model = get_optimized_model(name, X_train, y_train, categorical, numerical)
            scores = evaluate_model(X_test, y_test, model)
            
            # Generar visualizaciones
            plot_learning_curves(model, X_train, y_train, X_test, y_test, name)
            
            # Corregir el acceso a feature_importances_
            if hasattr(model.named_steps['model'], 'feature_importances_'):
                feature_names = (
                    model.named_steps['preprocessor']
                    .named_transformers_['cat']
                    .get_feature_names_out(categorical).tolist() +
                    numerical
                )
                plot_feature_importances(model.named_steps['model'], feature_names, name)
            
            results[name] = {
                'cv_scores': scores,
                'pipeline': model,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
            logging.info(f"\nResumen de {name}:")
            logging.info(f"Tamaño del conjunto de entrenamiento: {len(X_train)}")
            logging.info(f"Tamaño del conjunto de prueba: {len(X_test)}")
            logging.info(f"Distribución de clases en entrenamiento: {np.bincount(y_train)}")
            logging.info(f"Distribución de clases en prueba: {np.bincount(y_test)}")
            
        except Exception as e:
            logging.error(f"Error entrenando {name}: {str(e)}")
            continue
    
    if not results:
        raise ValueError("No se pudo entrenar ningún modelo correctamente")
        
    return results

def evaluate_single_model(model, X, y, X_train, X_test, y_train, y_test, name):
    # Evaluar con validación cruzada
    scores = evaluate_model(X, y, model)
    cv_mean = np.mean(scores)
    cv_std = np.std(scores)
    
    # Entrenar y evaluar en test
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Crear y guardar matriz de confusión
    save_confusion_matrix(model, X_test, y_test, name)
    
    print(f"{name}:")
    print(f"  CV ROC AUC: {cv_mean:.3f} (±{cv_std:.3f})")
    print(f"  Test ROC AUC: {test_auc:.3f}")
    
    return {
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'test_auc': test_auc,
        'pipeline': model
    }

def save_confusion_matrix(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {name}')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.savefig(f'{MODELS_DIR}/confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()

def evaluate_model_detailed(pipeline, X_test, y_test):
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    
    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall
    }

def clean_models_directory():
    """Limpia la carpeta models antes de generar nuevos modelos"""
    # Crear la carpeta si no existe
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print("Carpeta models creada.")
        return
    
    # Eliminar todo el contenido de la carpeta
    for filename in os.listdir(MODELS_DIR):
        file_path = os.path.join(MODELS_DIR, filename)
        try:
            if os.path.isfile(file_path):
                try:
                    os.unlink(file_path)
                    print(f"Eliminado: {filename}")
                except PermissionError:
                    print(f"No se pudo eliminar {filename} - archivo en uso")
            elif os.path.isdir(file_path):
                try:
                    shutil.rmtree(file_path)
                    print(f"Eliminado: {filename}")
                except PermissionError:
                    print(f"No se pudo eliminar {filename} - directorio en uso")
        except Exception as e:
            print(f'Error al eliminar {filename}: {e}')
    
    print("Carpeta models limpiada.")

def save_model_artifacts(model, importance, results):
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Guardar archivos usando with
    with open(f'{MODELS_DIR}/stroke_prediction_model.joblib', 'wb') as f:
        dump(model, f)
    with open(f'{MODELS_DIR}/feature_importance.joblib', 'wb') as f:
        dump(importance, f)
        
    # Guardar visualizaciones
    save_model_visualizations(results)

def save_model_visualizations(results):
    plt.style.use('default')
    sns.set_theme()
    
    # Comparación de modelos usando las métricas correctas
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    metrics = ['f1', 'roc_auc', 'precision', 'recall']
    
    for i, metric in enumerate(metrics):
        names = list(results.keys())
        scores = [np.mean(results[name]['cv_scores'][metric]) for name in names]
        stds = [np.std(results[name]['cv_scores'][metric]) for name in names]
        
        axes[i].bar(names, scores, yerr=stds)
        axes[i].set_title(f'{metric.upper()} Score')
        axes[i].set_ylabel('Score')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{MODELS_DIR}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_eda_visualizations(df):
    """Genera visualizaciones detalladas para el análisis exploratorio de datos"""
    os.makedirs(f'{MODELS_DIR}/eda', exist_ok=True)
    
    try:
        # 1. Distribución de variables numéricas por resultado
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, col in enumerate(['age', 'avg_glucose_level', 'bmi']):
            sns.kdeplot(data=df, x=col, hue='stroke', fill=True, ax=axes[i])
            axes[i].set_title(f'Distribución de {col} por Stroke')
        plt.tight_layout()
        plt.savefig(f'{MODELS_DIR}/eda/numerical_distributions.png')
        plt.close()
        
        # 2. Correlaciones entre variables numéricas
        plt.figure(figsize=(10, 8))
        numerical_cols = ['age', 'avg_glucose_level', 'bmi', 'stroke']
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlaciones entre Variables Numéricas')
        plt.savefig(f'{MODELS_DIR}/eda/correlation_heatmap.png')
        plt.close()
        
        # 3. Análisis de factores de riesgo
        risk_factors = ['hypertension', 'heart_disease', 'smoking_status']
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, factor in enumerate(risk_factors):
            stroke_rates = df.groupby(factor)['stroke'].mean() * 100
            stroke_rates.plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Tasa de Stroke por {factor}')
            axes[i].set_ylabel('Tasa de Stroke (%)')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{MODELS_DIR}/eda/risk_factors.png')
        plt.close()
        
        # 4. Análisis por edad y género
        plt.figure(figsize=(12, 6))
        age_bins = [0, 20, 40, 60, 80, 100]
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=['0-20', '21-40', '41-60', '61-80', '80+'])
        
        stroke_by_age_gender = df.groupby(['age_group', 'gender'])['stroke'].mean() * 100
        stroke_by_age_gender = stroke_by_age_gender.unstack()
        stroke_by_age_gender.plot(marker='o')
        plt.title('Tasa de Stroke por Edad y Género')
        plt.xlabel('Grupo de Edad')
        plt.ylabel('Tasa de Stroke (%)')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{MODELS_DIR}/eda/age_gender_analysis.png')
        plt.close()
        
        # 5. Análisis de glucosa y BMI
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.boxplot(data=df, x='stroke', y='avg_glucose_level', ax=axes[0])
        axes[0].set_title('Niveles de Glucosa por Stroke')
        
        sns.boxplot(data=df, x='stroke', y='bmi', ax=axes[1])
        axes[1].set_title('BMI por Stroke')
        
        plt.tight_layout()
        plt.savefig(f'{MODELS_DIR}/eda/glucose_bmi_analysis.png')
        plt.close()
        
        # 6. Análisis multivariado de riesgo (versión mejorada)
        plt.figure(figsize=(15, 8))
        
        # Calcular score de riesgo de manera más intuitiva
        risk_factors = {
            'Edad > 60': df['age'] > 60,
            'Hipertensión': df['hypertension'] == 1,
            'Enf. Cardíaca': df['heart_disease'] == 1,
            'Glucosa Alta': df['avg_glucose_level'] > 200,
            'Obesidad': df['bmi'] > 30
        }
        
        # Calcular score total
        risk_score = sum(risk_factors.values())
        
        # Calcular porcentaje de stroke por número de factores
        risk_analysis = pd.DataFrame({
            'Factores de Riesgo': risk_score,
            'Stroke': df['stroke']
        })
        
        stroke_rates = risk_analysis.groupby('Factores de Riesgo')['Stroke'].agg([
            ('Porcentaje de Stroke', lambda x: x.mean() * 100),
            ('Cantidad de Pacientes', 'count')
        ]).reset_index()
        
        # Crear gráfico de barras con dos ejes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Barras para el porcentaje de stroke
        bars = ax1.bar(stroke_rates['Factores de Riesgo'], 
                       stroke_rates['Porcentaje de Stroke'],
                       color='lightcoral', alpha=0.7)
        ax1.set_xlabel('Número de Factores de Riesgo Presentes')
        ax1.set_ylabel('Porcentaje de Casos de Stroke (%)', color='darkred')
        ax1.tick_params(axis='y', labelcolor='darkred')
        
        # Añadir línea de tendencia para cantidad de pacientes
        ax2 = ax1.twinx()
        line = ax2.plot(stroke_rates['Factores de Riesgo'],
                        stroke_rates['Cantidad de Pacientes'],
                        'b-', linewidth=2, marker='o')
        ax2.set_ylabel('Cantidad de Pacientes', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # Añadir etiquetas en las barras
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', color='darkred')
        
        # Añadir leyenda
        plt.title('Relación entre Factores de Riesgo y Probabilidad de Stroke', 
                  pad=20, fontsize=12)
        
        # Añadir texto explicativo
        plt.figtext(0.02, -0.05, 
                    'Factores de Riesgo Considerados:\n' + 
                    '\n'.join([f'- {factor}' for factor in risk_factors.keys()]),
                    fontsize=8, ha='left')
        
        plt.tight_layout()
        plt.savefig(f'{MODELS_DIR}/eda/risk_score_analysis.png', 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        # 7. Distribución de trabajo y estado civil
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        work_stroke = df.groupby('work_type')['stroke'].mean() * 100
        work_stroke.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Tasa de Stroke por Tipo de Trabajo')
        axes[0].set_ylabel('Tasa de Stroke (%)')
        
        married_stroke = df.groupby('ever_married')['stroke'].mean() * 100
        married_stroke.plot(kind='bar', ax=axes[1])
        axes[1].set_title('Tasa de Stroke por Estado Civil')
        axes[1].set_ylabel('Tasa de Stroke (%)')
        
        plt.tight_layout()
        plt.savefig(f'{MODELS_DIR}/eda/work_marriage_analysis.png')
        plt.close()
        
        logging.info("Visualizaciones EDA completadas exitosamente")
        
    except Exception as e:
        logging.error(f"Error generando visualizaciones: {str(e)}")
        raise

def analyze_best_model(model, X_test, y_test, feature_importance):
    """Genera visualizaciones detalladas del mejor modelo"""
    try:
        os.makedirs(f'{MODELS_DIR}/model_analysis', exist_ok=True)
        
        # 1. Curva ROC
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC')
            plt.legend(loc="lower right")
            plt.savefig(f'{MODELS_DIR}/model_analysis/roc_curve.png', bbox_inches='tight')
            plt.close()
        except Exception as e:
            logging.error(f"Error generando curva ROC: {str(e)}")
        
        # 2. Importancia de características
        if feature_importance:
            try:
                plt.figure(figsize=(12, 6))
                importance_df = pd.DataFrame(
                    list(feature_importance.items()), 
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=True)
                
                sns.barplot(data=importance_df, y='Feature', x='Importance')
                plt.title('Importancia de Características')
                plt.savefig(f'{MODELS_DIR}/model_analysis/feature_importance.png', bbox_inches='tight')
                plt.close()
            except Exception as e:
                logging.error(f"Error generando gráfico de importancia: {str(e)}")
        
        # 3. Matriz de confusión
        try:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues')
            plt.title('Matriz de Confusión (Porcentajes)')
            plt.ylabel('Real')
            plt.xlabel('Predicho')
            plt.savefig(f'{MODELS_DIR}/model_analysis/confusion_matrix_percent.png', bbox_inches='tight')
            plt.close()
        except Exception as e:
            logging.error(f"Error generando matriz de confusión: {str(e)}")
        
    except Exception as e:
        logging.error(f"Error en analyze_best_model: {str(e)}")
        raise

def plot_learning_curves(model, X_train, y_train, X_test, y_test, name):
    """Visualiza las curvas de aprendizaje del modelo"""
    try:
        train_scores = []
        test_scores = []
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        for size in train_sizes:
            sample_size = int(len(X_train) * size)
            model.fit(X_train[:sample_size], y_train[:sample_size])
            
            train_scores.append(model.score(X_train[:sample_size], y_train[:sample_size]))
            test_scores.append(model.score(X_test, y_test))
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores, 'o-', label='Training score')
        plt.plot(train_sizes, test_scores, 'o-', label='Test score')
        plt.title(f'Learning Curves - {name}')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(f'{MODELS_DIR}/learning_curve_{name.lower().replace(" ", "_")}.png')
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting learning curves: {str(e)}")

def plot_feature_importances(model, feature_names, name):
    """Visualiza la importancia de características para cada modelo"""
    try:
        if not hasattr(model, 'feature_importances_'):
            logging.info(f"El modelo {name} no tiene atributo feature_importances_")
            return
            
        importances = model.feature_importances_
        if len(importances) != len(feature_names):
            logging.warning(f"Dimensiones no coinciden: importances={len(importances)}, features={len(feature_names)}")
            return
            
        # Ordenar por importancia
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        plt.figure(figsize=(12, 6))
        plt.title(f'Feature Importances - {name}')
        plt.bar(range(len(sorted_importances)), sorted_importances)
        plt.xticks(range(len(sorted_importances)), sorted_features, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{MODELS_DIR}/feature_importance_{name.lower().replace(" ", "_")}.png')
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting feature importance for {name}: {str(e)}")

def create_ensemble_pipeline(categorical_features, numerical_features):
    """Crea un pipeline con ensemble de modelos y SMOTE"""
    # Crear modelos base
    rf_model = MODEL_CONFIGS['Random Forest']['model']
    gb_model = MODEL_CONFIGS['Gradient Boosting']['model']
    
    # Crear preprocessor
    preprocessor = create_base_pipeline(categorical_features, numerical_features)
    
    # Crear ensemble calibrado
    ensemble = CalibratedEnsembleClassifier(rf_model, gb_model)
    
    # Crear pipeline completo
    return IMBPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.3)),
        ('model', ensemble)
    ])

def main():
    logging.info("Iniciando entrenamiento")
    start_time = datetime.now()
    
    clean_models_directory()
    
    try:
        # Cargar y preprocesar datos usando la ruta correcta
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, 'healthcare-dataset-stroke-data.csv')
        
        df = pd.read_csv(csv_path)
        df = preprocess_data(df)
        
        # Generar visualizaciones con los datos preprocesados
        generate_eda_visualizations(df)
        logging.info("Visualizaciones EDA generadas")
        
        # Preparar datos para el modelo
        X, y, categorical, numerical = load_data()
        logging.info(f"Dataset cargado: {X.shape}")
        
        try:
            # Entrenar y evaluar modelos
            results = evaluate_multiple_models(X, y, categorical, numerical)
            
            # Seleccionar mejor modelo
            if results:
                best_model = max(
                    results.items(), 
                    key=lambda x: (
                        np.mean(x[1]['cv_scores']['f1']) + 
                        np.mean(x[1]['cv_scores']['roc_auc'])
                    ) / 2
                )
                
                logging.info(f"Mejor modelo: {best_model[0]}")
                logging.info(f"F1 Score: {np.mean(best_model[1]['cv_scores']['f1']):.3f}")
                logging.info(f"ROC AUC: {np.mean(best_model[1]['cv_scores']['roc_auc']):.3f}")
                
                # Analizar mejor modelo
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
                )
                importance = analyze_feature_importance(best_model[1]['pipeline'], X, numerical)
                analyze_best_model(best_model[1]['pipeline'], X_test, y_test, importance)
                logging.info("Análisis del mejor modelo completado")
                
                save_model_artifacts(best_model[1]['pipeline'], importance, results)
            else:
                logging.error("No se obtuvieron resultados de ningún modelo")
                
        except Exception as e:
            logging.error(f"Error en el entrenamiento de modelos: {str(e)}")
            raise
            
    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {str(e)}")
        raise
    
    duration = datetime.now() - start_time
    logging.info(f"Entrenamiento completado en {duration}")

if __name__ == "__main__":
    main()

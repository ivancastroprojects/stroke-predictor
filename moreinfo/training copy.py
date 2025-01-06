import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Configurar backend no interactivo
import matplotlib.pyplot as plt
from joblib import dump
import seaborn as sns
import os
import shutil
from datetime import datetime
import logging

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as IMBPipeline
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (
    RepeatedStratifiedKFold, 
    cross_val_score, 
    train_test_split, 
    RandomizedSearchCV, 
    StratifiedKFold,
    cross_validate
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, make_scorer, f1_score, precision_score, recall_score
from scipy.stats import randint, uniform

# Configurar el estilo global una sola vez
plt.style.use('default')
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

# Constantes globales
RANDOM_STATE = 42
MODELS_DIR = './models'
TEST_SIZE = 0.2
CV_SPLITS = 5
CV_REPEATS = 3
N_ITER_SEARCH = 20

# Constantes para el manejo de plots
plt.rcParams['figure.max_open_warning'] = 0  # Evitar warnings de demasiadas figuras
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{MODELS_DIR}/training.log'),
        logging.StreamHandler()
    ]
)

# Configuración de modelos más eficiente
MODEL_CONFIGS = {
    'Random Forest': {
        'model': RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced_subsample',
            criterion='entropy'  # Mejor para datos desbalanceados
        ),
        'params': {
            'model__n_estimators': [1000],  # Fijo para estabilidad
            'model__max_depth': [None, 10, 15, 20],
            'model__min_samples_split': [5, 10],
            'model__min_samples_leaf': [4, 8],
            'model__max_features': ['sqrt', 'log2'],
            'model__max_samples': [0.8]  # Bootstrap con 80% de las muestras
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            validation_fraction=0.2,
            n_iter_no_change=5,  # Early stopping
            subsample=0.8  # Reducir overfitting
        ),
        'params': {
            'model__n_estimators': [500],
            'model__learning_rate': [0.01, 0.05],
            'model__max_depth': [3, 4],
            'model__min_samples_split': [5],
            'model__min_samples_leaf': [4]
        }
    }
}

def preprocess_data(df):
    df = df.copy()
    
    # 1. Manejo de valores nulos más específico
    df['bmi'] = df['bmi'].fillna(df.groupby(['gender', 'age'])['bmi'].transform('median'))
    
    # 2. Discretización de variables numéricas (bins más significativos médicamente)
    df['age_group'] = pd.cut(
        df['age'], 
        bins=[0, 30, 40, 50, 60, 70, 80, 100],
        labels=['<30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']
    )
    
    df['glucose_group'] = pd.cut(
        df['avg_glucose_level'],
        bins=[0, 70, 100, 125, 150, 180, 400],
        labels=['low', 'normal', 'pre-diabetic', 'diabetic', 'high', 'very_high']
    )
    
    df['bmi_group'] = pd.cut(
        df['bmi'],
        bins=[0, 18.5, 25, 30, 35, 100],
        labels=['underweight', 'normal', 'overweight', 'obese', 'morbid_obese']
    )
    
    # 3. Codificación más informativa
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 0})
    df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
    df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})
    
    # 4. Características de riesgo basadas en criterios médicos
    df['diabetes_risk'] = ((df['avg_glucose_level'] >= 125) & 
                          (df['age'] > 40)).astype(int)
    
    df['cardiovascular_risk'] = ((df['hypertension'] == 1) | 
                                (df['heart_disease'] == 1)).astype(int)
    
    df['lifestyle_risk'] = ((df['smoking_status'].isin(['formerly smoked', 'smokes'])) & 
                           (df['bmi'] > 30)).astype(int)
    
    df['age_risk'] = (df['age'] >= 65).astype(int)
    
    # 5. Interacciones importantes
    df['combined_health_score'] = (
        df['cardiovascular_risk'] * 2 +
        df['diabetes_risk'] * 2 +
        df['lifestyle_risk'] +
        df['age_risk'] * 1.5
    )
    
    return df

def load_data():
    df = pd.read_csv('./healthcare-dataset-stroke-data.csv')
    df = df.drop('id', axis=1)
    df = preprocess_data(df)
    
    categorical = ['work_type', 'smoking_status', 'age_group', 'glucose_group', 'bmi_group']
    numerical = [
        'age', 'avg_glucose_level', 'bmi',
        'gender', 'hypertension', 'heart_disease',
        'ever_married', 'Residence_type',
        'diabetes_risk', 'cardiovascular_risk', 'lifestyle_risk',
        'age_risk', 'combined_health_score'
    ]
    
    y = df['stroke']
    X = df[categorical + numerical]
    
    return X, y, categorical, numerical

def evaluate_model(X, y, model):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Definir métricas personalizadas
    scoring = {
        'roc_auc': 'roc_auc',
        'f1': make_scorer(f1_score, zero_division=0),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'average_precision': 'average_precision'  # Añadir PR-AUC
    }
    
    # Evaluar con todas las métricas
    scores = cross_validate(
        model, X, y,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        return_train_score=True  # Para detectar overfitting
    )
    
    return {k.replace('test_', ''): scores[k] for k in scores if k.startswith('test_')}

def analyze_feature_importance(model, X, numerical):
    """Analiza la importancia de características para diferentes tipos de modelos"""
    feature_importance = {}
    
    # Obtener el modelo base
    base_model = model.named_steps['model']
    
    # Obtener nombres de características después de la transformación
    feature_names = []
    for name, transformer in model.named_steps['preprocessor'].named_transformers_.items():
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out())
        else:
            feature_names.extend(X[numerical].columns)
    
    # Obtener importancias según el tipo de modelo
    if hasattr(base_model, 'coef_'):  # Para modelos lineales (LDA)
        importances = np.abs(base_model.coef_[0])
    elif hasattr(base_model, 'feature_importances_'):  # Para árboles y bosques
        importances = base_model.feature_importances_
    else:  # Para otros modelos (como SVM) usar permutation importance
        importances = np.ones(len(feature_names))  # Importancia uniforme como fallback
    
    # Normalizar las importancias
    total = np.sum(importances)
    
    for idx, feature in enumerate(feature_names):
        importance = (importances[idx] / total) * 100
        feature_importance[feature] = importance
    
    # Agrupar características similares
    grouped_importance = {
        'Edad': feature_importance.get('age', 0),
        'Hipertensión': feature_importance.get('hypertension', 0),
        'Nivel de Glucosa': feature_importance.get('avg_glucose_level', 0),
        'IMC': feature_importance.get('bmi', 0),
        'Enf. Cardíacas': feature_importance.get('heart_disease', 0),
        'Estado Civil': sum(v for k, v in feature_importance.items() if 'married' in k),
        'Tabaquismo': sum(v for k, v in feature_importance.items() if 'smoking' in k),
        'Tipo de Trabajo': sum(v for k, v in feature_importance.items() if 'work_type' in k),
        'Género': sum(v for k, v in feature_importance.items() if 'gender' in k),
        'Residencia': sum(v for k, v in feature_importance.items() if 'Residence_type' in k)
    }
    
    # Normalizar los porcentajes agrupados
    total = sum(grouped_importance.values())
    grouped_importance = {k: (v/total) for k, v in grouped_importance.items()}
    
    # Ordenar por importancia
    grouped_importance = dict(sorted(grouped_importance.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True))
    
    return grouped_importance

def get_optimized_model(name, X, y, categorical, numerical):
    config = MODEL_CONFIGS[name]
    pipeline = create_base_pipeline(categorical, numerical, config['model'], y)
    
    # Usar F1 como métrica principal
    scoring = {
        'f1': make_scorer(f1_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'precision': make_scorer(precision_score, zero_division=0),
        'roc_auc': 'roc_auc'
    }
    
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=config['params'],
        n_iter=50,  # Más iteraciones
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring=scoring,
        refit='f1',  # Optimizar para F1
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    print(f"\nOptimizando {name}...")
    random_search.fit(X, y)
    
    return random_search.best_estimator_

def create_base_pipeline(categorical_features, numerical_features, model=None, y=None):
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_pipeline, categorical_features),
            ('num', numerical_pipeline, numerical_features)
        ],
        remainder='drop'
    )

    if model is not None and y is not None:
        return Pipeline([
            ('preprocessor', preprocessor),
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
        print(f"\nEntrenando {name}...")
        model = get_optimized_model(name, X_train, y_train, categorical, numerical)
        
        # Entrenar el modelo
        model.fit(X_train, y_train)
        
        # Evaluar el modelo
        scores = evaluate_model(X_test, y_test, model)
        
        # Calcular predicciones y métricas adicionales
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Imprimir resumen del modelo
        print(f"\nResumen de {name}:")
        print(f"F1-Score: {np.mean(scores['f1']):.3f} (±{np.std(scores['f1']):.3f})")
        print(f"ROC-AUC: {np.mean(scores['roc_auc']):.3f} (±{np.std(scores['roc_auc']):.3f})")
        print(f"Precisión: {np.mean(scores['precision']):.3f} (±{np.std(scores['precision']):.3f})")
        print(f"Recall: {np.mean(scores['recall']):.3f} (±{np.std(scores['recall']):.3f})")
        
        # Calcular distribución de predicciones
        pos_rate = np.mean(y_pred)
        print(f"Tasa de predicciones positivas: {pos_rate:.1%}")
        print(f"Proporción real de casos positivos: {np.mean(y_test):.1%}")
        
        results[name] = {
            'cv_scores': scores,
            'pipeline': model,
            'predictions': {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        }
        
        # Guardar matriz de confusión
        save_confusion_matrix(model, X_test, y_test, name)
    
    return results

def save_confusion_matrix(model, X_test, y_test, name):
    try:
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Matriz de Confusión - {name}')
        ax.set_ylabel('Real')
        ax.set_xlabel('Predicho')
        plt.savefig(f'{MODELS_DIR}/confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close(fig)
    except Exception as e:
        logging.error(f"Error al guardar matriz de confusión: {str(e)}")

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
    models_dir = './models'
    
    # Crear la carpeta si no existe
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print("Carpeta models creada.")
        return
    
    # Eliminar todo el contenido de la carpeta
    for filename in os.listdir(models_dir):
        file_path = os.path.join(models_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            print(f"Eliminado: {filename}")
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
    try:
        names = list(results.keys())
        metrics = ['f1', 'roc_auc', 'precision', 'recall']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            scores = [np.mean(results[name]['cv_scores'][metric]) for name in names]
            errors = [np.std(results[name]['cv_scores'][metric]) for name in names]
            
            axes[i].bar(names, scores, yerr=errors)
            axes[i].set_title(f'{metric.upper()} Score')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'{MODELS_DIR}/model_comparison.png')
        plt.close(fig)
    except Exception as e:
        logging.error(f"Error al guardar visualizaciones: {str(e)}")

def generate_eda_visualizations(df):
    """Genera visualizaciones para el análisis exploratorio de datos"""
    os.makedirs(f'{MODELS_DIR}/eda', exist_ok=True)
    
    # Configurar estilo global
    plt.style.use('default')  # Usar estilo por defecto
    sns.set_theme(style="whitegrid")  # Configurar estilo de seaborn
    sns.set_palette("husl")  # Configurar paleta de colores
    
    try:
        # 1. Distribución de edad y riesgo de stroke
        plt.figure(figsize=(12, 6))
        sns.kdeplot(data=df, x='age', hue='stroke', fill=True, common_norm=False)
        plt.title('Distribución de Edad y Riesgo de ACV')
        plt.xlabel('Edad')
        plt.ylabel('Densidad')
        plt.savefig(f'{MODELS_DIR}/eda/age_stroke_distribution.png', bbox_inches='tight')
        plt.close()
        
        # 2. Mapa de calor de correlaciones
        plt.figure(figsize=(10, 8))
        numerical_cols = ['age', 'avg_glucose_level', 'bmi']
        correlation_matrix = df[numerical_cols + ['stroke']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlación entre Variables Numéricas')
        plt.savefig(f'{MODELS_DIR}/eda/correlation_heatmap.png', bbox_inches='tight')
        plt.close()
        
        # 3. Comparación de factores de riesgo por género
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        risk_factors = ['hypertension', 'heart_disease', 'smoking_status', 'work_type']
        
        for i, factor in enumerate(risk_factors):
            ax = axes[i//2, i%2]
            df_grouped = df.groupby(['gender', factor])['stroke'].mean().unstack()
            df_grouped.plot(kind='bar', ax=ax)
            ax.set_title(f'Riesgo de ACV por Género y {factor}')
            ax.set_ylabel('Probabilidad de ACV')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{MODELS_DIR}/eda/gender_risk_factors.png', bbox_inches='tight')
        plt.close()
        
        # 4. Distribución de glucosa por estado de ACV
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='stroke', y='avg_glucose_level')
        plt.title('Niveles de Glucosa por Estado de ACV')
        plt.xlabel('ACV')
        plt.ylabel('Nivel de Glucosa')
        plt.savefig(f'{MODELS_DIR}/eda/glucose_stroke.png', bbox_inches='tight')
        plt.close()
        
        # 5. Pirámide de edad por género
        plt.figure(figsize=(12, 8))
        bins = np.linspace(df['age'].min(), df['age'].max(), 20)
        male_data = df[df['gender'] == 'Male']['age']
        female_data = df[df['gender'] == 'Female']['age']
        
        plt.hist(male_data, bins=bins, alpha=0.5, color='blue', label='Hombres')
        plt.hist(female_data, bins=bins, alpha=0.5, color='red', label='Mujeres')
        plt.title('Distribución de Edad por Género')
        plt.xlabel('Edad')
        plt.ylabel('Cantidad de Personas')
        plt.legend()
        plt.savefig(f'{MODELS_DIR}/eda/age_gender_pyramid.png', bbox_inches='tight')
        plt.close()
        
        # 6. Gráfico de violín para BMI
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x='stroke', y='bmi', hue='gender')
        plt.title('Distribución de IMC por Estado de ACV y Género')
        plt.xlabel('ACV')
        plt.ylabel('IMC')
        plt.savefig(f'{MODELS_DIR}/eda/bmi_violin.png', bbox_inches='tight')
        plt.close()
        
        # 7. Tendencia de riesgo por edad y género
        plt.figure(figsize=(12, 6))
        df['age_group'] = pd.cut(df['age'], bins=[0, 20, 40, 60, 80, 100], 
                                labels=['0-20', '21-40', '41-60', '61-80', '80+'])
        risk_by_age = df.groupby(['age_group', 'gender'], observed=True)['stroke'].mean().unstack()
        risk_by_age.plot(marker='o')
        plt.title('Tendencia de Riesgo de ACV por Edad y Género')
        plt.xlabel('Grupo de Edad')
        plt.ylabel('Probabilidad de ACV')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{MODELS_DIR}/eda/risk_trend.png', bbox_inches='tight')
        plt.close()
        
        logging.info("Visualizaciones EDA completadas exitosamente")
        
    except Exception as e:
        logging.error(f"Error generando visualizaciones: {str(e)}")
        raise

def analyze_best_model(model, X_test, y_test, feature_importance):
    """Genera visualizaciones detalladas del mejor modelo"""
    os.makedirs(f'{MODELS_DIR}/model_analysis', exist_ok=True)
    
    # 1. Curva ROC
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.savefig(f'{MODELS_DIR}/model_analysis/roc_curve.png', bbox_inches='tight')
    plt.close()
    
    # 2. Importancia de características
    plt.figure(figsize=(12, 6))
    importance_df = pd.DataFrame(
        list(feature_importance.items()), 
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=True)
    
    sns.barplot(data=importance_df, y='Feature', x='Importance')
    plt.title('Importancia de Características')
    plt.savefig(f'{MODELS_DIR}/model_analysis/feature_importance.png', bbox_inches='tight')
    plt.close()
    
    # 3. Matriz de confusión con porcentajes
    cm = confusion_matrix(y_test, model.predict(X_test))
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues')
    plt.title('Matriz de Confusión (Porcentajes)')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.savefig(f'{MODELS_DIR}/model_analysis/confusion_matrix_percent.png', bbox_inches='tight')
    plt.close()

def main():
    logging.info("Iniciando entrenamiento")
    start_time = datetime.now()
    
    try:
        clean_models_directory()
        
        # Cargar y preprocesar datos
        df = pd.read_csv('./healthcare-dataset-stroke-data.csv')
        df = preprocess_data(df)
        
        # Generar visualizaciones en un bloque try separado
        try:
            generate_eda_visualizations(df)
            logging.info("Visualizaciones EDA generadas")
        except Exception as e:
            logging.error(f"Error en visualizaciones EDA: {str(e)}")
        
        # Preparar datos para el modelo
        X, y, categorical, numerical = load_data()
        logging.info(f"Dataset cargado: {X.shape}")
        
        # Entrenar y evaluar modelos
        results = evaluate_multiple_models(X, y, categorical, numerical)
        
        # Seleccionar el mejor modelo basado en F1-score
        best_model = max(
            results.items(), 
            key=lambda x: np.mean(x[1]['cv_scores']['f1'])
        )
        
        logging.info(f"Mejor modelo: {best_model[0]}")
        logging.info(f"F1 Score medio: {np.mean(best_model[1]['cv_scores']['f1']):.3f}")
        
        # Analizar mejor modelo
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        importance = analyze_feature_importance(best_model[1]['pipeline'], X, numerical)
        analyze_best_model(best_model[1]['pipeline'], X_test, y_test, importance)
        logging.info("Análisis del mejor modelo completado")
        
        save_model_artifacts(best_model[1]['pipeline'], importance, results)
        
    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {str(e)}")
        raise
    finally:
        # Limpiar recursos de matplotlib
        plt.close('all')
    
    duration = datetime.now() - start_time
    logging.info(f"Entrenamiento completado en {duration}")

if __name__ == "__main__":
    main()

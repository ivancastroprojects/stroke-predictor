import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
import joblib
import seaborn as sns
import os
import shutil
from datetime import datetime
import logging

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as IMBPipeline
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, average_precision_score, roc_curve
from scipy.stats import randint, uniform

# Constantes globales
RANDOM_STATE = 42
MODELS_DIR = './models'
TEST_SIZE = 0.2
CV_SPLITS = 10
CV_REPEATS = 3
N_ITER_SEARCH = 50

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{MODELS_DIR}/training.log'),
        logging.StreamHandler()
    ]
)

# Configuración de modelos
MODEL_CONFIGS = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
        'params': {
            'model__n_estimators': randint(100, 500),
            'model__max_depth': [None] + list(range(5, 30)),
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 10),
            'model__max_features': ['sqrt', 'log2', None]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'params': {
            'model__n_estimators': randint(100, 500),
            'model__learning_rate': uniform(0.01, 0.3),
            'model__max_depth': randint(3, 10),
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 10),
            'model__subsample': uniform(0.6, 0.4)
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=RANDOM_STATE),
        'params': {
            'model__C': uniform(0.1, 10),
            'model__gamma': ['scale', 'auto'] + list(uniform(0.001, 0.1).rvs(10)),
            'model__kernel': ['rbf', 'poly'],
            'model__class_weight': ['balanced', None]
        }
    }
}

def preprocess_data(df):
    df = df.copy()  # Evitar warnings de SettingWithCopyWarning
    
    # Procesamiento en una sola pasada
    df.loc[:, 'bmi'] = pd.to_numeric(df['bmi'], errors='coerce').fillna(df['bmi'].mean())
    df.loc[df['avg_glucose_level'] > 300, 'avg_glucose_level'] = 300
    df.loc[:, 'work_type'] = df['work_type'].replace({'Never_worked': 'Unemployed', 'children': 'Unemployed'})
    
    # Crear rangos de edad eficientemente
    df.loc[:, 'age_group'] = pd.cut(df['age'], 
                                   bins=[0, 13, 25, 40, 60, 100],
                                   labels=['Child', 'Young', 'Adult', 'Middle', 'Elder'])
    
    return df

def load_data():
    df = pd.read_csv('./healthcare-dataset-stroke-data.csv')
    df = df.drop('id', axis=1)
    
    # Aplicar preprocesamiento
    df = preprocess_data(df)
    
    # Definir columnas categóricas y numéricas (sin age_group)
    categorical = ['gender', 'hypertension', 'heart_disease', 'ever_married', 
                  'work_type', 'Residence_type', 'smoking_status']
    numerical = ['avg_glucose_level', 'bmi', 'age']
    
    y = df['stroke']
    X = df.drop(['stroke', 'age_group'], axis=1)  # Eliminamos age_group
    
    return X, y, categorical, numerical

def evaluate_model(X, y, model):
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, 
        n_repeats=CV_REPEATS, 
        random_state=RANDOM_STATE
    )
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=1)
    return scores

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
    if name not in MODEL_CONFIGS:
        return create_lda_pipeline(categorical, numerical)
        
    config = MODEL_CONFIGS[name]
    pipeline = create_base_pipeline(categorical, numerical, config['model'])
    
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=config['params'],
        n_iter=N_ITER_SEARCH,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    
    print(f"\nOptimizando {name}...")
    random_search.fit(X, y)
    print(f"Mejor score: {random_search.best_score_:.3f}")
    
    return random_search.best_estimator_

def create_base_pipeline(categorical_features, numerical_features, model=None):
    # Pipeline para características numéricas
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline para características categóricas
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combinación de pipelines para preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )

    # Si se proporciona un modelo, crear un pipeline completo
    if model is not None:
        return Pipeline([
            ('preprocessor', preprocessor),
    ('model', model)
])

    # Si no se proporciona modelo, devolver solo el preprocessor
    return preprocessor

def create_lda_pipeline(categorical, numerical):
    return create_base_pipeline(
        categorical_features=categorical,
        numerical_features=numerical,
        model=LinearDiscriminantAnalysis()
    )

def evaluate_multiple_models(X, y, categorical, numerical):
    # Split the data usando la constante global
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Crear todos los modelos incluyendo LDA
    models = {
        name: get_optimized_model(name, X, y, categorical, numerical)
        for name in ['Random Forest', 'Gradient Boosting', 'SVM']
    }
    models['LDA'] = create_lda_pipeline(categorical, numerical)
    
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluando {name}...")
        results[name] = evaluate_single_model(
            model, X, y, X_train, X_test, y_train, y_test, name
        )
    
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
    plt.style.use('default')  # Usar estilo por defecto
    sns.set_theme()  # Aplicar tema de seaborn
    
    # Comparación de modelos
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    cv_means = [results[name]['cv_mean'] for name in names]
    cv_stds = [results[name]['cv_std'] for name in names]
    
    ax.errorbar(names, cv_means, yerr=cv_stds, fmt='o', capsize=5)
    ax.set_title('Comparación de Modelos - ROC AUC')
    ax.set_ylabel('ROC AUC Score')
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{MODELS_DIR}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_eda_visualizations(df):
    """Genera visualizaciones para el análisis exploratorio de datos"""
    os.makedirs(f'{MODELS_DIR}/eda', exist_ok=True)
    
    # Configurar estilo global
    plt.style.use('default')  # Usar estilo por defecto de matplotlib
    sns.set_theme()  # Aplicar tema de seaborn
    
    try:
        # 1. Distribución de la variable objetivo
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=df, x='stroke')
        plt.title('Distribución de Casos de Stroke')
        # Añadir porcentajes sobre las barras
        total = len(df)
        for p in ax.patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center')
        plt.savefig(f'{MODELS_DIR}/eda/stroke_distribution.png', bbox_inches='tight')
        plt.close()
        
        # 2. Distribución de edad por stroke
        if 'age_group' in df.columns:  # Verificar si existe la columna
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df, x='age_group', y='age', hue='stroke')
            plt.title('Distribución de Edad por Grupo y Stroke')
            plt.savefig(f'{MODELS_DIR}/eda/age_distribution.png', bbox_inches='tight')
            plt.close()
        
        # 3. Glucosa vs BMI con stroke
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x='avg_glucose_level', y='bmi', hue='stroke', alpha=0.6)
        plt.title('Relación entre Nivel de Glucosa y BMI')
        plt.savefig(f'{MODELS_DIR}/eda/glucose_bmi_relation.png', bbox_inches='tight')
        plt.close()
        
        # 4. Factores de riesgo categóricos
        categorical_cols = ['gender', 'hypertension', 'heart_disease', 'smoking_status']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        for i, col in enumerate(categorical_cols):
            ax = axes[i//2, i%2]
            sns.barplot(data=df, x=col, y='stroke', ax=ax)
            ax.set_title(f'Tasa de Stroke por {col}')
            ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f'{MODELS_DIR}/eda/risk_factors.png', bbox_inches='tight')
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
    
    clean_models_directory()
    
    try:
        # Cargar y preprocesar datos primero
        df = pd.read_csv('./healthcare-dataset-stroke-data.csv')
        df = preprocess_data(df)  # Aplicar preprocesamiento antes de visualizar
        
        # Generar visualizaciones con los datos preprocesados
        generate_eda_visualizations(df)
        logging.info("Visualizaciones EDA generadas")
        
        # Preparar datos para el modelo
        X, y, categorical, numerical = load_data()
        logging.info(f"Dataset cargado: {X.shape}")
        
        # Entrenar y evaluar modelos
        results = evaluate_multiple_models(X, y, categorical, numerical)
        best_model = max(results.items(), key=lambda x: x[1]['test_auc'])
        
        logging.info(f"Mejor modelo: {best_model[0]}")
        logging.info(f"Test ROC AUC: {best_model[1]['test_auc']:.3f}")
        
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
    
    duration = datetime.now() - start_time
    logging.info(f"Entrenamiento completado en {duration}")

if __name__ == "__main__":
    main()

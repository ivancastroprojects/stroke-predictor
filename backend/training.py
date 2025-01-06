from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import datetime
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Configuración global de visualización
plt.style.use('default')
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./training.log'),
        logging.StreamHandler()
    ]
)

# Constantes
MODELS_DIR = './models'
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
CV_REPEATS = 3

# Configuración de modelos
MODEL_CONFIGS = {
    'LDA': {
        'model': LinearDiscriminantAnalysis(),
        'name': 'Linear Discriminant Analysis'
    },
    'Random Forest': {
        'model': RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced_subsample',
            criterion='entropy',
            n_estimators=1000,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=4,
            max_features='sqrt',
            max_samples=0.8
        ),
        'name': 'Random Forest'
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            n_estimators=500,
            learning_rate=0.01,
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=4,
            subsample=0.8,
            validation_fraction=0.2,
            n_iter_no_change=5
        ),
        'name': 'Gradient Boosting'
    }
}

def ensure_model_directory():
    """Asegura que exista el directorio de modelos y sus subdirectorios"""
    directories = [
        MODELS_DIR,
        f'{MODELS_DIR}/eda',
        f'{MODELS_DIR}/model_analysis'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_data():
    df = pd.read_csv('./healthcare-dataset-stroke-data.csv')
    df = df.drop('id', axis=1)
    categorical = ['hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    numerical = ['avg_glucose_level', 'bmi', 'age']
    y = df['stroke']
    X = df.drop('stroke', axis=1)
    return df,X, y, categorical, numerical

def evaluate_model(X, y, model):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    return scores

def generate_eda_visualizations(df):
    """Genera visualizaciones para el análisis exploratorio de datos"""
    try:
        # 1. Distribución de edad y riesgo de stroke
        plt.figure(figsize=(12, 6))
        sns.kdeplot(data=df, x='age', hue='stroke', fill=True, common_norm=False)
        plt.title('Distribución de Edad y Riesgo de ACV')
        plt.xlabel('Edad')
        plt.ylabel('Densidad')
        plt.savefig(f'{MODELS_DIR}/eda/age_stroke_distribution.png', bbox_inches='tight')
        plt.close()
        
        # 2. Análisis de glucosa y BMI
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.boxplot(data=df, x='stroke', y='avg_glucose_level', ax=ax1)
        ax1.set_title('Niveles de Glucosa por Estado de ACV')
        ax1.set_xlabel('ACV')
        ax1.set_ylabel('Nivel de Glucosa')
        
        sns.boxplot(data=df, x='stroke', y='bmi', ax=ax2)
        ax2.set_title('IMC por Estado de ACV')
        ax2.set_xlabel('ACV')
        ax2.set_ylabel('IMC')
        
        plt.tight_layout()
        plt.savefig(f'{MODELS_DIR}/eda/glucose_bmi_analysis.png', bbox_inches='tight')
        plt.close()
        
        # 3. Análisis de trabajo y estado civil
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        df_work = df.groupby('work_type')['stroke'].mean().sort_values(ascending=False)
        df_work.plot(kind='bar', ax=ax1)
        ax1.set_title('Tasa de ACV por Tipo de Trabajo')
        ax1.set_ylabel('Probabilidad de ACV')
        
        df_marriage = df.groupby('ever_married')['stroke'].mean().sort_values(ascending=False)
        df_marriage.plot(kind='bar', ax=ax2)
        ax2.set_title('Tasa de ACV por Estado Civil')
        ax2.set_ylabel('Probabilidad de ACV')
        
        plt.tight_layout()
        plt.savefig(f'{MODELS_DIR}/eda/work_marriage_analysis.png', bbox_inches='tight')
        plt.close()
        
        # 4. Análisis de factores de riesgo
        risk_factors = ['hypertension', 'heart_disease', 'smoking_status']
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, factor in enumerate(risk_factors):
            df_risk = df.groupby(factor)['stroke'].mean().sort_values(ascending=False)
            df_risk.plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Tasa de ACV por {factor}')
            axes[i].set_ylabel('Probabilidad de ACV')
        
        plt.tight_layout()
        plt.savefig(f'{MODELS_DIR}/eda/risk_score_analysis.png', bbox_inches='tight')
        plt.close()
        
        logging.info("Visualizaciones EDA completadas exitosamente")
        
    except Exception as e:
        logging.error(f"Error generando visualizaciones EDA: {str(e)}")

def analyze_model_performance(model, X_test, y_test):
    """Genera visualizaciones del rendimiento del modelo"""
    try:
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
        
        # 2. Matriz de confusión
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
        
        logging.info("Análisis del modelo completado exitosamente")
        
    except Exception as e:
        logging.error(f"Error en el análisis del modelo: {str(e)}")

def analyze_feature_importance(model, X, numerical):
    """Analiza la importancia de características para diferentes tipos de modelos"""
    try:
        feature_importance = {}
        
        # Obtener el modelo base
        base_model = model.named_steps['m']
        
        # Obtener nombres de características después de la transformación
        feature_names = []
        for name, transformer in model.named_steps['t'].named_transformers_.items():
            if name == 'o' and hasattr(transformer, 'get_feature_names_out'):
                # Para características categóricas one-hot encoded
                feature_names.extend(transformer.get_feature_names_out())
            elif name == 'imp':
                # Para características numéricas
                feature_names.extend(numerical)
        
        # Obtener importancias según el tipo de modelo
        if hasattr(base_model, 'coef_'):  # Para modelos lineales (LDA)
            importances = np.abs(base_model.coef_[0])
        elif hasattr(base_model, 'feature_importances_'):  # Para árboles y bosques
            importances = base_model.feature_importances_
        else:
            importances = np.ones(len(feature_names))
        
        # Verificar que tengamos el mismo número de nombres e importancias
        if len(feature_names) != len(importances):
            logging.warning(f"Desajuste en características: {len(feature_names)} nombres vs {len(importances)} importancias")
            return None
        
        # Normalizar las importancias individuales
        total = np.sum(np.abs(importances))
        for idx, feature in enumerate(feature_names):
            importance = (np.abs(importances[idx]) / total) * 100
            feature_importance[feature] = importance
        
        # Agrupar características similares
        grouped_importance = {
            'Edad': sum(v for k, v in feature_importance.items() if 'age' in str(k).lower()),
            'Tipo de Trabajo': sum(v for k, v in feature_importance.items() if 'work_type' in str(k).lower()),
            'Nivel de Glucosa': sum(v for k, v in feature_importance.items() if 'glucose' in str(k).lower()),
            'Tabaquismo': sum(v for k, v in feature_importance.items() if 'smoking' in str(k).lower()),
            'Hipertensión': sum(v for k, v in feature_importance.items() if 'hypertension' in str(k).lower()),
            'Estado Civil': sum(v for k, v in feature_importance.items() if 'married' in str(k).lower()),
            'Residencia': sum(v for k, v in feature_importance.items() if 'residence' in str(k).lower()),
            'IMC': sum(v for k, v in feature_importance.items() if 'bmi' in str(k).lower()),
            'Enf. Cardíacas': sum(v for k, v in feature_importance.items() if 'heart' in str(k).lower())
        }
        
        # Normalizar los porcentajes agrupados para que sumen exactamente 100%
        total_importance = sum(grouped_importance.values())
        if total_importance > 0:
            # Multiplicar por 100 después de la normalización para obtener porcentajes
            grouped_importance = {k: (v/total_importance) * 100 
                                for k, v in grouped_importance.items()}
            
            # Ordenar por importancia de mayor a menor
            grouped_importance = dict(sorted(
                grouped_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            # Verificar que la suma sea 100%
            total = sum(grouped_importance.values())
            if abs(total - 100) > 0.01:  # Si hay una diferencia mayor a 0.01%
                factor = 100 / total
                grouped_importance = {k: v * factor for k, v in grouped_importance.items()}
            
            return grouped_importance
        else:
            logging.error("No se pudieron calcular las importancias de las características")
            return None
            
    except Exception as e:
        logging.error(f"Error en analyze_feature_importance: {str(e)}")
        return None

def save_model_artifacts(model, scores, feature_importance):
    """Guarda el modelo y sus métricas"""
    try:
        # Guardar el modelo
        dump(model, f'{MODELS_DIR}/stroke_prediction_model.joblib')
        
        # Guardar métricas en un archivo de texto
        with open(f'{MODELS_DIR}/model_metrics.txt', 'w') as f:
            f.write(f"ROC AUC Score: {np.mean(scores):.3f} (±{np.std(scores):.3f})\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if feature_importance is not None:
                f.write("Importancia de Características:\n")
                for feature, importance in feature_importance.items():
                    f.write(f"{feature}: {importance:.2f}%\n")
            else:
                f.write("\nNo se pudo calcular la importancia de las características\n")
        
        # Guardar y visualizar importancia de características solo si está disponible
        if feature_importance is not None:
            dump(feature_importance, f'{MODELS_DIR}/feature_importance.joblib')
            
            plt.figure(figsize=(12, 6))
            features = list(feature_importance.keys())
            importances = list(feature_importance.values())
            
            sns.barplot(x=importances, y=features)
            plt.title('Importancia de Características en la Predicción de ACV')
            plt.xlabel('Importancia (%)')
            plt.tight_layout()
            plt.savefig(f'{MODELS_DIR}/model_analysis/feature_importance.png', bbox_inches='tight')
            plt.close()
        
        logging.info("Modelo y métricas guardados exitosamente")
        
    except Exception as e:
        logging.error(f"Error guardando artefactos del modelo: {str(e)}")

def evaluate_multiple_models(X, y, categorical, numerical):
    """Evalúa múltiples modelos y retorna sus resultados"""
    results = {}
    
    for name, config in MODEL_CONFIGS.items():
        logging.info(f"\nEntrenando {name}...")
        
        # Crear pipeline para el modelo actual
transformer = ColumnTransformer(transformers=[
    ('imp', SimpleImputer(strategy='median'), numerical),
            ('o', OneHotEncoder(handle_unknown='ignore'), categorical)
])

pipeline = Pipeline(steps=[
    ('t', transformer),
    ('p', PowerTransformer(method='yeo-johnson', standardize=True)),
    ('over', SMOTE()),
            ('m', config['model'])
])

        # Evaluar modelo
        try:
scores = evaluate_model(X, y, pipeline)
            
            results[name] = {
                'pipeline': pipeline,
                'scores': scores,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            }
            
            logging.info(f"{name}:")
            logging.info(f"ROC AUC Score: {np.mean(scores):.3f} (±{np.std(scores):.3f})")
            
        except Exception as e:
            logging.error(f"Error evaluando {name}: {str(e)}")
            continue
    
    return results

def select_best_model(results):
    """Selecciona el mejor modelo basado en el ROC AUC score medio"""
    if not results:
        raise ValueError("No hay resultados de modelos para comparar")
    
    best_model = max(results.items(), key=lambda x: x[1]['mean_score'])
    logging.info(f"\nMejor modelo: {best_model[0]}")
    logging.info(f"ROC AUC Score: {best_model[1]['mean_score']:.3f} (±{best_model[1]['std_score']:.3f})")
    
    return best_model[0], best_model[1]['pipeline']

def plot_model_comparison(results):
    """Genera una visualización comparativa de los modelos"""
    try:
        plt.figure(figsize=(10, 6))
        
        names = list(results.keys())
        means = [results[name]['mean_score'] for name in names]
        stds = [results[name]['std_score'] for name in names]
        
        bars = plt.bar(names, means, yerr=stds, capsize=5)
        plt.title('Comparación de Modelos - ROC AUC Score')
        plt.ylabel('ROC AUC Score')
        plt.ylim(0.5, 1.0)  # ROC AUC score range
        
        # Añadir valores sobre las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{MODELS_DIR}/model_analysis/model_comparison.png', bbox_inches='tight')
        plt.close()
        
        logging.info("Comparación de modelos guardada exitosamente")
        
    except Exception as e:
        logging.error(f"Error generando comparación de modelos: {str(e)}")

def main():
    logging.info("Iniciando entrenamiento del modelo")
    start_time = datetime.now()
    
    try:
        # Asegurar que existan los directorios necesarios
        ensure_model_directory()
        
        # Cargar datos
        df, X, y, categorical, numerical = load_data()
        logging.info(f"Dataset cargado: {X.shape}")
        
        # Generar visualizaciones EDA
        generate_eda_visualizations(df)
        
        # Evaluar múltiples modelos
        results = evaluate_multiple_models(X, y, categorical, numerical)
        
        # Generar visualización comparativa
        plot_model_comparison(results)
        
        # Seleccionar el mejor modelo
        best_model_name, best_pipeline = select_best_model(results)
        
        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # Entrenar el mejor modelo
        best_pipeline.fit(X_train, y_train)
        
        # Analizar importancia de características
        feature_importance = analyze_feature_importance(best_pipeline, X, numerical)
        if feature_importance:
            logging.info("\nImportancia de características:")
            for feature, importance in feature_importance.items():
                logging.info(f"{feature}: {importance:.2f}%")
        
        # Generar visualizaciones del rendimiento del modelo
        analyze_model_performance(best_pipeline, X_test, y_test)
        
        # Guardar el modelo y sus métricas
        save_model_artifacts(best_pipeline, results[best_model_name]['scores'], feature_importance)
        
        duration = datetime.now() - start_time
        logging.info(f"\nEntrenamiento completado en {duration}")
        
    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {str(e)}")
        raise
    finally:
        plt.close('all')

if __name__ == "__main__":
    main()

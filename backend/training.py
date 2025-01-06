from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, StackingClassifier
import pandas as pd
import numpy as np
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import datetime
from sklearn.metrics import roc_curve, confusion_matrix, classification_report, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib

# Configuración global de visualización
matplotlib.use('Agg')  # Usar backend no interactivo
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
        'model': LinearDiscriminantAnalysis(
            solver='lsqr',
            shrinkage='auto',
            priors=[0.25, 0.75]  # Ajustado para mejor balance médico
        ),
        'name': 'Linear Discriminant Analysis'
    },
    'Logistic Regression': {
        'model': LogisticRegression(
            random_state=RANDOM_STATE,
            C=0.2,  # Aumentado para capturar relaciones más complejas
            penalty='l2',
            solver='saga',
            max_iter=2500,
            class_weight={0: 1, 1: 5},  # Mayor énfasis en detectar casos positivos
            n_jobs=-1
        ),
        'name': 'Logistic Regression'
    },
    'Random Forest': {
        'model': RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight={0: 1, 1: 5},  # Mayor énfasis en detectar casos positivos
            criterion='entropy',
            n_estimators=1200,
            max_depth=8,  # Aumentado para capturar patrones más complejos
            min_samples_split=12,
            min_samples_leaf=6,
            max_features='sqrt',
            max_samples=0.85,  # Aumentado para mejor estabilidad
            bootstrap=True
        ),
        'name': 'Random Forest'
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

def handle_outliers(X):
    """Maneja outliers usando IQR y clipping"""
    numerical = ['avg_glucose_level', 'bmi', 'age']
    for col in numerical:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        X[col] = X[col].clip(lower=lower, upper=upper)
    return X

def create_interaction_features(X):
    """Crea características de interacción más sofisticadas"""
    X = X.copy()
    
    # Transformaciones logarítmicas para variables sesgadas
    X['log_glucose'] = np.log1p(X['avg_glucose_level'])
    X['log_bmi'] = np.log1p(X['bmi'])
    
    # Interacciones edad-enfermedades
    X['age_hypertension'] = X['age'] * X['hypertension']
    X['age_heart_disease'] = X['age'] * X['heart_disease']
    
    # Interacción edad-glucosa (factor de riesgo importante)
    X['age_glucose'] = X['age'] * X['log_glucose']
    
    # Características polinómicas para edad
    X['age_squared'] = X['age'] ** 2
    
    # Categorización de BMI según OMS
    X['bmi_category'] = pd.cut(X['bmi'], 
                              bins=[0, 18.5, 24.9, 29.9, float('inf')],
                              labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Grupos de edad con más granularidad
    X['age_group'] = pd.cut(X['age'], 
                           bins=[0, 20, 40, 50, 60, 70, float('inf')],
                           labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder', 'Very_Elder'])
    
    # Grupos de glucosa basados en riesgo
    X['glucose_category'] = pd.cut(X['avg_glucose_level'],
                                 bins=[0, 70, 100, 125, float('inf')],
                                 labels=['Low', 'Normal', 'Pre-diabetic', 'Diabetic'])
    
    return X

def load_data():
    df = pd.read_csv('./healthcare-dataset-stroke-data.csv')
    df = df.drop('id', axis=1)
    
    # Manejo de outliers
    df = handle_outliers(df)
    
    # Crear características de interacción
    df = create_interaction_features(df)
    
    # Definir variables categóricas y numéricas
    categorical = ['hypertension', 'heart_disease', 'ever_married', 'work_type', 
                  'Residence_type', 'smoking_status', 'age_group', 'bmi_category', 
                  'glucose_category']
    
    numerical = ['avg_glucose_level', 'bmi', 'age', 'age_hypertension', 
                'age_heart_disease', 'log_glucose', 'log_bmi', 'age_glucose', 
                'age_squared']
    
    # Definir orden para variables ordinales basado en riesgo
    smoking_order = ['never smoked', 'Unknown', 'formerly smoked', 'smokes']
    work_type_order = ['children', 'Never_worked', 'Private', 'Self-employed', 'Govt_job']
    
    y = df['stroke']
    X = df.drop('stroke', axis=1)
    
    return df, X, y, categorical, numerical, smoking_order, work_type_order

def log_dataset_statistics(df, y):
    """Registra estadísticas detalladas del dataset"""
    logging.info("\n=== Estadísticas del Dataset ===")
    
    # Distribución de clases
    class_dist = y.value_counts()
    logging.info("\nDistribución de clases:")
    logging.info(f"No Stroke: {class_dist[0]} ({class_dist[0]/len(y)*100:.2f}%)")
    logging.info(f"Stroke: {class_dist[1]} ({class_dist[1]/len(y)*100:.2f}%)")
    logging.info(f"Ratio de desbalanceo: 1:{class_dist[0]/class_dist[1]:.2f}")
    
    # Valores faltantes
    missing = df.isnull().sum()
    if missing.any():
        logging.info("\nValores faltantes por columna:")
        for col in missing[missing > 0].index:
            logging.info(f"{col}: {missing[col]} ({missing[col]/len(df)*100:.2f}%)")
    
    # Estadísticas numéricas
    numerical = ['age', 'avg_glucose_level', 'bmi']
    logging.info("\nEstadísticas de variables numéricas:")
    for col in numerical:
        stats = df[col].describe()
        logging.info(f"\n{col}:")
        logging.info(f"  Media: {stats['mean']:.2f}")
        logging.info(f"  Mediana: {stats['50%']:.2f}")
        logging.info(f"  Std: {stats['std']:.2f}")
        logging.info(f"  Rango: [{stats['min']:.2f} - {stats['max']:.2f}]")

def log_outlier_info(df, numerical_cols):
    """Registra información sobre outliers detectados"""
    logging.info("\n=== Información de Outliers ===")
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        
        if len(outliers) > 0:
            logging.info(f"\n{col}:")
            logging.info(f"  Outliers detectados: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
            logging.info(f"  Rango normal: [{lower:.2f} - {upper:.2f}]")
            logging.info(f"  Valores extremos: [{outliers.min():.2f} - {outliers.max():.2f}]")

def evaluate_model(X, y, model):
    """Evaluación extendida del modelo con múltiples métricas"""
    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    
    # ROC AUC
    roc_scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    
    # Precision
    precision_scores = cross_val_score(model, X, y, scoring='precision', cv=cv, n_jobs=-1)
    
    # Recall
    recall_scores = cross_val_score(model, X, y, scoring='recall', cv=cv, n_jobs=-1)
    
    # F1
    f1_scores = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1)
    
    logging.info("\nMétricas de validación cruzada:")
    logging.info(f"ROC AUC: {np.mean(roc_scores):.3f} (±{np.std(roc_scores):.3f})")
    logging.info(f"Precision: {np.mean(precision_scores):.3f} (±{np.std(precision_scores):.3f})")
    logging.info(f"Recall: {np.mean(recall_scores):.3f} (±{np.std(recall_scores):.3f})")
    logging.info(f"F1: {np.mean(f1_scores):.3f} (±{np.std(f1_scores):.3f})")
    
    return {
        'roc_auc': roc_scores,
        'precision': precision_scores,
        'recall': recall_scores,
        'f1': f1_scores
    }

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

def _generate_model_visualizations(y_test, y_pred, y_pred_proba):
    """Genera visualizaciones adicionales del modelo"""
    try:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.savefig(f'{MODELS_DIR}/model_analysis/roc_curve.png', bbox_inches='tight')
        plt.close()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curva Precision-Recall')
        plt.savefig(f'{MODELS_DIR}/model_analysis/precision_recall_curve.png', bbox_inches='tight')
        plt.close()
        
        # Matriz de confusión normalizada
        cm_percent = confusion_matrix(y_test, y_pred, normalize='true') * 100
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=['No Stroke', 'Stroke'],
                   yticklabels=['No Stroke', 'Stroke'])
        plt.title('Matriz de Confusión (%)')
        plt.ylabel('Real')
        plt.xlabel('Predicho')
        plt.savefig(f'{MODELS_DIR}/model_analysis/confusion_matrix_percent.png', bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logging.error(f"Error generando visualizaciones: {str(e)}")

def analyze_model_performance(model, X_test, y_test):
    """Análisis detallado del rendimiento del modelo"""
    try:
        logging.info("\n=== Análisis Detallado del Modelo ===")
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Classification Report
        logging.info("\nInforme de Clasificación:")
        logging.info("\n" + classification_report(y_test, y_pred))
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        logging.info("\nMatriz de Confusión:")
        logging.info(f"\n[[TN={cm[0,0]} FP={cm[0,1]}]")
        logging.info(f" [FN={cm[1,0]} TP={cm[1,1]}]]")
        
        # Métricas adicionales
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        
        logging.info("\nMétricas Detalladas:")
        logging.info(f"Precision: {precision:.3f}")
        logging.info(f"Recall (Sensibilidad): {recall:.3f}")
        logging.info(f"Especificidad: {specificity:.3f}")
        
        # Curva Precision-Recall
        avg_precision = average_precision_score(y_test, y_pred_proba)
        logging.info(f"Average Precision Score: {avg_precision:.3f}")
        
        # Visualizaciones
        _generate_model_visualizations(y_test, y_pred, y_pred_proba)
        
        logging.info("Análisis del modelo completado exitosamente")
        
        return {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'avg_precision': avg_precision
        }
        
    except Exception as e:
        logging.error(f"Error en el análisis del modelo: {str(e)}")
        return None

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

def save_model_artifacts(model, scores, feature_importance, performance_metrics=None):
    """Guarda el modelo y sus métricas"""
    try:
        # Guardar el modelo
        dump(model, f'{MODELS_DIR}/stroke_prediction_model.joblib')
        
        # Guardar métricas en un archivo de texto
        with open(f'{MODELS_DIR}/model_metrics.txt', 'w') as f:
            f.write(f"=== Métricas del Modelo ===\n\n")
            
            # Métricas de validación cruzada
            f.write("Validación Cruzada:\n")
            f.write(f"ROC AUC: {np.mean(scores['roc_auc']):.3f} (±{np.std(scores['roc_auc']):.3f})\n")
            f.write(f"Precision: {np.mean(scores['precision']):.3f} (±{np.std(scores['precision']):.3f})\n")
            f.write(f"Recall: {np.mean(scores['recall']):.3f} (±{np.std(scores['recall']):.3f})\n")
            f.write(f"F1: {np.mean(scores['f1']):.3f} (±{np.std(scores['f1']):.3f})\n\n")
            
            # Métricas de rendimiento final
            if performance_metrics:
                f.write("Rendimiento en Conjunto de Prueba:\n")
                f.write(f"Precision: {performance_metrics['precision']:.3f}\n")
                f.write(f"Recall: {performance_metrics['recall']:.3f}\n")
                f.write(f"Especificidad: {performance_metrics['specificity']:.3f}\n")
                f.write(f"Average Precision: {performance_metrics['avg_precision']:.3f}\n\n")
            
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

def evaluate_multiple_models(X, y, categorical, numerical, smoking_order, work_type_order):
    """Evalúa múltiples modelos y retorna sus resultados"""
    results = {}
    
    for name, config in MODEL_CONFIGS.items():
        logging.info(f"\nEntrenando {name}...")
        
        # Transformadores específicos para cada tipo de variable
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        smoking_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('ordinal', OrdinalEncoder(categories=[smoking_order], 
                                     handle_unknown='use_encoded_value', 
                                     unknown_value=-1))
        ])
        
        work_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Never_worked')),
            ('ordinal', OrdinalEncoder(categories=[work_type_order], 
                                     handle_unknown='use_encoded_value', 
                                     unknown_value=-1))
        ])
        
        numerical_transformer = Pipeline([
            ('imputer', KNNImputer(n_neighbors=7)),  # Aumentado para mejor robustez
            ('scaler', RobustScaler(quantile_range=(5, 95))),  # Más robusto a outliers
            ('power', PowerTransformer(method='yeo-johnson', standardize=True))
        ])
        
        # Ajustar el balanceo según el modelo
        if name == 'LDA':
            sampler = SMOTETomek(random_state=42, sampling_strategy=0.4)  # Aumentado para mejor balance
        else:
            sampler = ADASYN(random_state=42, sampling_strategy=0.4, n_neighbors=7)  # Ajustado para mejor calidad
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical),
            ('cat', categorical_transformer, [col for col in categorical 
                                           if col not in ['smoking_status', 'work_type']]),
            ('smoke', smoking_transformer, ['smoking_status']),
            ('work', work_transformer, ['work_type'])
        ])
        
        pipeline_steps = [('t', preprocessor)]
        if sampler:
            pipeline_steps.append(('over', sampler))
        pipeline_steps.append(('m', config['model']))
        
        pipeline = Pipeline(steps=pipeline_steps)
        
        # Evaluar modelo con validación cruzada estratificada
        try:
            scores = evaluate_model(X, y, pipeline)
            
            results[name] = {
                'pipeline': pipeline,
                'scores': scores,
                'mean_roc_auc': np.mean(scores['roc_auc']),
                'std_roc_auc': np.std(scores['roc_auc']),
                'mean_precision': np.mean(scores['precision']),
                'std_precision': np.std(scores['precision']),
                'mean_recall': np.mean(scores['recall']),
                'std_recall': np.std(scores['recall']),
                'mean_f1': np.mean(scores['f1']),
                'std_f1': np.std(scores['f1'])
            }
            
            logging.info(f"{name} - Resultados:")
            logging.info(f"ROC AUC: {results[name]['mean_roc_auc']:.3f} (±{results[name]['std_roc_auc']:.3f})")
            logging.info(f"Precision: {results[name]['mean_precision']:.3f} (±{results[name]['std_precision']:.3f})")
            logging.info(f"Recall: {results[name]['mean_recall']:.3f} (±{results[name]['std_recall']:.3f})")
            logging.info(f"F1: {results[name]['mean_f1']:.3f} (±{results[name]['std_f1']:.3f})")
            
        except Exception as e:
            logging.error(f"Error evaluando {name}: {str(e)}")
            continue
    
    return results

def select_best_model(results):
    """Selecciona el mejor modelo basado en el ROC AUC score medio"""
    if not results:
        raise ValueError("No hay resultados de modelos para comparar")
    
    best_model = max(results.items(), key=lambda x: x[1]['mean_roc_auc'])
    logging.info(f"\nMejor modelo: {best_model[0]}")
    logging.info(f"ROC AUC Score: {best_model[1]['mean_roc_auc']:.3f} (±{best_model[1]['std_roc_auc']:.3f})")
    
    return best_model[0], best_model[1]['pipeline']

def plot_model_comparison(results):
    """Genera una visualización comparativa de los modelos"""
    try:
        plt.figure(figsize=(10, 6))
        
        names = list(results.keys())
        means = [results[name]['mean_roc_auc'] for name in names]
        stds = [results[name]['std_roc_auc'] for name in names]
        
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
        
        # Cargar datos con el nuevo preprocesamiento
        df, X, y, categorical, numerical, smoking_order, work_type_order = load_data()
        logging.info(f"Dataset cargado: {X.shape}")
        
        # Logging detallado del dataset
        log_dataset_statistics(df, y)
        log_outlier_info(df, ['age', 'avg_glucose_level', 'bmi'])
        
        # Generar visualizaciones EDA
        generate_eda_visualizations(df)
        
        # Evaluar múltiples modelos con el nuevo preprocesamiento
        results = evaluate_multiple_models(X, y, categorical, numerical, smoking_order, work_type_order)
        
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
        performance_metrics = analyze_model_performance(best_pipeline, X_test, y_test)
        
        # Guardar el modelo y sus métricas
        save_model_artifacts(
            best_pipeline, 
            results[best_model_name]['scores'],
            feature_importance,
            performance_metrics
        )
        
        duration = datetime.now() - start_time
        logging.info(f"\nEntrenamiento completado en {duration}")
        
    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {str(e)}")
        raise
    finally:
        plt.close('all')

if __name__ == "__main__":
    main()

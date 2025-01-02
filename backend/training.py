import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as IMBPipeline
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


def load_data():
    df = pd.read_csv('./healthcare-dataset-stroke-data.csv')
    df = df.drop('id', axis=1)
    categorical = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    numerical = ['avg_glucose_level', 'bmi', 'age']
    
    y = df['stroke']
    X = df.drop('stroke', axis=1)
    return X, y, categorical, numerical

# Definición de la función para evaluar el modelo
def evaluate_model(X, y, model):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=1)
    return scores

# Load data
X, y, categorical, numerical = load_data()
print(X.shape, y.shape)

# Define the LDA model
model = LinearDiscriminantAnalysis()

# Prepare the pipeline
pipelineTransformer = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='median'), numerical),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)  # Manejo de categorías desconocidas
])

pipeline = IMBPipeline(steps=[
    ('transformer', pipelineTransformer),
    ('powert', PowerTransformer(method='yeo-johnson', standardize=True)),
    ('smote', SMOTE()), 
    ('model', model)
])

# Evaluate the model
scores = evaluate_model(X, y, pipeline)
#print('LDA %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Plot the results
plt.boxplot([scores], labels=['LDA'], showmeans=True)
plt.show()

# Fit the pipeline on the entire dataset
pipeline.fit(X, y)

# Save the trained pipeline
dump(pipeline, './stroke_prediction_model.joblib')

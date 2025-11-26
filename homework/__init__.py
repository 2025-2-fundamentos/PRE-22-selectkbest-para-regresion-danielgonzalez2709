"""
Homework: SelectKBest para regresión - Auto MPG Dataset
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


def load_and_prepare_data():
    """Carga y prepara los datos del dataset auto_mpg."""
    dataset = pd.read_csv("files/input/auto_mpg.csv")
    dataset = dataset.dropna()
    dataset["Origin"] = dataset["Origin"].map(
        {1: "USA", 2: "Europe", 3: "Japan"},
    )
    y = dataset.pop("MPG")
    x = dataset.copy()
    return x, y


class MPGClassifier(BaseEstimator, ClassifierMixin):
    """Clasificador que convierte MPG a clases y luego devuelve los valores originales."""

    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, X, y):
        # Convertir y a clases (índices)
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)

        # Entrenar el clasificador
        self.classifier.fit(X, y_encoded)
        return self

    def predict(self, X):
        # Predecir clases
        y_pred_encoded = self.classifier.predict(X)
        # Convertir de vuelta a valores de MPG
        y_pred = self.label_encoder_.inverse_transform(y_pred_encoded)
        return y_pred

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)


def create_estimator():
    """Crea el pipeline con SelectKBest y el modelo."""

    # Identificar columnas numéricas y categóricas
    numeric_features = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year']
    categorical_features = ['Origin']

    # Preprocesamiento: OneHotEncoder para Origin
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )

    # Clasificador base con alta capacidad
    base_classifier = GradientBoostingClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )

    # Pipeline completo
    estimator = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_classif, k='all')),
        ('classifier', MPGClassifier(base_classifier))
    ])

    return estimator


def train_and_save_model():
    """Entrena el modelo y lo guarda en estimator.pickle."""

    # Cargar datos
    x, y = load_and_prepare_data()

    # Crear estimador
    estimator = create_estimator()

    # Entrenar
    print("Entrenando modelo...")
    estimator.fit(x, y)

    # Guardar modelo
    with open("estimator.pickle", "wb") as file:
        pickle.dump(estimator, file)

    print("Modelo guardado en estimator.pickle")

    return estimator


if __name__ == "__main__":
    train_and_save_model()


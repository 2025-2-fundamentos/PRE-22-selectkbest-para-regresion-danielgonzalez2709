"""Script para probar el modelo."""

import pandas as pd
import numpy as np
from homework import load_and_prepare_data, create_estimator

# Cargar datos
x, y = load_and_prepare_data()

print(f"Tipo de y: {type(y)}")
print(f"Dtype de y: {y.dtype}")
print(f"Primeros valores de y: {y.head()}")
print(f"Valores únicos: {y.nunique()}")

# Crear y entrenar estimador
estimator = create_estimator()
print("\nEntrenando modelo...")
estimator.fit(x, y)

# Predecir
y_pred = estimator.predict(x)
print(f"\nTipo de y_pred: {type(y_pred)}")
print(f"Dtype de y_pred: {y_pred.dtype}")
print(f"Primeros valores de y_pred: {y_pred[:5]}")

# Verificar si son iguales
print(f"\nValores únicos en y_pred: {len(np.unique(y_pred))}")
print(f"Valores únicos en y: {len(np.unique(y))}")

# Intentar calcular accuracy
try:
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
except Exception as e:
    print(f"\nError al calcular accuracy: {e}")
    
    # Intentar convertir a valores discretos
    print("\nIntentando comparar valores directamente...")
    matches = (y.values == y_pred).sum()
    total = len(y)
    manual_accuracy = matches / total
    print(f"Accuracy manual: {manual_accuracy:.4f} ({matches}/{total})")


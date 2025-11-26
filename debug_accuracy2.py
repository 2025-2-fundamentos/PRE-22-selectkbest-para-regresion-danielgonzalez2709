"""Debug accuracy_score with many unique values."""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import type_of_target

# Cargar datos reales
dataset = pd.read_csv("files/input/auto_mpg.csv")
dataset = dataset.dropna()
y = dataset["MPG"]

print(f"Unique values in y: {y.nunique()}")
print(f"Type of target: {type_of_target(y)}")

# Crear predicciones simuladas (todas correctas)
y_pred = y.values.copy()

print(f"\nType of target y_pred: {type_of_target(y_pred)}")

# Intentar accuracy_score
try:
    acc = accuracy_score(y, y_pred)
    print(f"Accuracy: {acc}")
except Exception as e:
    print(f"Error: {e}")

# Verificar el umbral
# sklearn usa un umbral para decidir si es continuo o multiclass
# Veamos cuántos valores únicos necesitamos
for n_unique in [10, 20, 50, 100, 127]:
    # Crear datos con n_unique valores
    test_y = np.random.choice(np.arange(n_unique, dtype=float), size=392)
    print(f"\nn_unique={n_unique}, type_of_target: {type_of_target(test_y)}")


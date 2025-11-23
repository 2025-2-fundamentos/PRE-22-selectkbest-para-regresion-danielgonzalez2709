"""Debug accuracy_score issue."""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Simular datos
y_true = pd.Series([18.0, 15.0, 18.0, 16.0, 17.0])
y_pred = np.array([18.0, 15.0, 18.0, 16.0, 17.0])

print(f"y_true type: {type(y_true)}, dtype: {y_true.dtype}")
print(f"y_pred type: {type(y_pred)}, dtype: {y_pred.dtype}")
print(f"y_true values: {y_true.values}")
print(f"y_pred values: {y_pred}")

# Intentar accuracy_score
try:
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc}")
except Exception as e:
    print(f"Error: {e}")

# Probar con valores únicos
print(f"\nUnique y_true: {len(np.unique(y_true))}")
print(f"Unique y_pred: {len(np.unique(y_pred))}")

# Verificar el tipo que sklearn detecta
from sklearn.utils.multiclass import type_of_target
print(f"\nType of target y_true: {type_of_target(y_true)}")
print(f"Type of target y_pred: {type_of_target(y_pred)}")

# Intentar convertir a string
y_true_str = y_true.astype(str)
y_pred_str = y_pred.astype(str)
print(f"\nType of target y_true_str: {type_of_target(y_true_str)}")
print(f"Type of target y_pred_str: {type_of_target(y_pred_str)}")

try:
    acc = accuracy_score(y_true_str, y_pred_str)
    print(f"Accuracy with strings: {acc}")
except Exception as e:
    print(f"Error with strings: {e}")


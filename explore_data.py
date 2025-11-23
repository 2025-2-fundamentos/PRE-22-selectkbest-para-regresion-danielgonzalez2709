"""Script para explorar los datos del dataset auto_mpg."""

import pandas as pd
import numpy as np

# Cargar datos
dataset = pd.read_csv("files/input/auto_mpg.csv")
print("Dataset original:")
print(dataset.head(10))
print(f"\nShape: {dataset.shape}")
print(f"\nInfo:")
print(dataset.info())
print(f"\nValores nulos:")
print(dataset.isnull().sum())

# Limpiar datos
dataset = dataset.dropna()
print(f"\nShape después de dropna: {dataset.shape}")

# Mapear Origin
dataset["Origin"] = dataset["Origin"].map(
    {1: "USA", 2: "Europe", 3: "Japan"},
)

# Separar target
y = dataset.pop("MPG")
x = dataset.copy()

print(f"\nTarget (MPG):")
print(f"Min: {y.min()}, Max: {y.max()}, Mean: {y.mean():.2f}, Std: {y.std():.2f}")
print(f"Unique values: {y.nunique()}")
print(f"\nValores únicos de MPG (primeros 20):")
print(sorted(y.unique())[:20])

print(f"\nFeatures:")
print(x.head())
print(f"\nDtypes:")
print(x.dtypes)


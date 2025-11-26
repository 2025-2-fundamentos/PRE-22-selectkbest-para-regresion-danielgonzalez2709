"""Autograding script."""


def load_data():

    import pandas as pd

    dataset = pd.read_csv("files/input/auto_mpg.csv")
    dataset = dataset.dropna()
    dataset["Origin"] = dataset["Origin"].map(
        {1: "USA", 2: "Europe", 3: "Japan"},
    )
    y = dataset.pop("MPG")
    x = dataset.copy()

    return x, y


def load_estimator():

    import os
    import pickle

    if not os.path.exists("estimator.pickle"):
        return None
    with open("estimator.pickle", "rb") as file:
        estimator = pickle.load(file)

    return estimator


def test_01():

    from sklearn.metrics import accuracy_score

    x, y = load_data()
    estimator = load_estimator()

    # Convertir a strings para que accuracy_score los trate como clases discretas
    y_true_str = y.astype(str)
    y_pred_str = estimator.predict(x).astype(str)

    accuracy = accuracy_score(
        y_true=y_true_str,
        y_pred=y_pred_str,
    )

    assert accuracy > 0.9545

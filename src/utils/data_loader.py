import pandas as pd
import numpy as np
import os
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


def load_heart_data():
    data_path = os.path.join(os.path.dirname(__file__), "../../data/heart.csv")
    df = pd.read_csv(data_path)

    X = df.drop("target", axis=1)
    y = df["target"]

    X = pd.get_dummies(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values


def load_cancer_data():
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
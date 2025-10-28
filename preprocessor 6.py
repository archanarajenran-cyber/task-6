"""
knn_iris.py

K-Nearest Neighbors example using Iris dataset (CSV).
- Loads /mnt/data/Iris.csv (adjust path if needed)
- Normalizes features
- Trains KNN for multiple k values and prints accuracies
- Shows confusion matrix for best k
- Plots accuracy vs k and decision boundaries (PCA -> 2D)
- Saves plots to current directory

Requirements:
pip install pandas numpy scikit-learn matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

DATA_PATH = "/mnt/data/Iris.csv"  # change if your file is elsewhere

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # Try to detect label column - many Iris CSVs have 'species' or 'class' or 'Species'
    label_cols = [c for c in df.columns if c.lower() in ("species", "class", "label", "target")]
    if label_cols:
        y = df[label_cols[0]].values
        X = df.drop(columns=[label_cols[0]]).select_dtypes(include=[np.number]).values
    else:
        # fallback: assume last column is the label
        X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
        y = df.iloc[:, -1].values
    return X, y, df

def preprocess(X_train, X_test):

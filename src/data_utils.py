# src/data_utils.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_clinical_data(csv_path, feature_cols, time_col, event_col):
    df = pd.read_csv(csv_path)
    X = df[feature_cols].values
    time = df[time_col].values
    event = df[event_col].values
    return X, time, event, df


def train_val_split(X_img, X_tab, time, event, test_size=0.2, seed=42):
    return train_test_split(
        X_img, X_tab, time, event,
        test_size=test_size,
        random_state=seed,
        stratify=event
    )

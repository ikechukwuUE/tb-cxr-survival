# src/data_utils.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import *
import os
import re
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def generate_synthetic_tb_clinical_data(
    n_samples=1000,
    seed=RANDOM_SEED,
    censoring_rate=0.3
):
    """
    This function helps us generate our own synthetic survival data 
    using selected clinical variables.
    """
    np.random.seed(seed)

    # Clinical covariates
    age = np.random.normal(45, 15, n_samples).clip(18, 85)
    sex = np.random.binomial(1, 0.6, n_samples)  # male=1
    hiv = np.random.binomial(1, 0.25, n_samples)
    diabetes = np.random.binomial(1, 0.2, n_samples)
    bmi = np.random.normal(20, 3, n_samples).clip(14, 35)
    hemoglobin = np.random.normal(11.5, 1.8, n_samples).clip(7, 17)
    albumin = np.random.normal(3.2, 0.6, n_samples).clip(1.8, 5)
    smear_positive = np.random.binomial(1, 0.55, n_samples)
    retreatment = np.random.binomial(1, 0.15, n_samples)

    # Assemble dataframe
    X = pd.DataFrame({
        "age": age,
        "sex": sex,
        "hiv": hiv,
        "diabetes": diabetes,
        "bmi": bmi,
        "hemoglobin": hemoglobin,
        "albumin": albumin,
        "smear_positive": smear_positive,
        "retreatment": retreatment
    })

    # True coefficients (log hazard ratios)
    beta = np.array([
        0.03,   # age
        0.15,   # sex
        0.9,    # HIV
        0.5,    # diabetes
       -0.08,   # BMI
       -0.12,   # hemoglobin
       -0.6,    # albumin
        0.4,    # smear+
        0.7     # retreatment
    ])

    linear_predictor = np.dot(X.values, beta)

    # Baseline hazard
    baseline_hazard = 0.01

    # Event times
    event_time = np.random.exponential(
        scale=1 / (baseline_hazard * np.exp(linear_predictor))
    )

    # Censoring
    censor_time = np.random.exponential(
        scale=event_time.mean(),
        size=n_samples
    )

    time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)

    return X, time, event


def load_tb_cxr_dataset(
    data_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode="binary",
    seed=RANDOM_SEED
):
    """
    Load TB Chest X-ray images from a Kaggle-style directory
    using a TensorFlow data pipeline.
    """

    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        labels="inferred" if label_mode else None,
        label_mode="binary" if label_mode else None,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed
    )

    # Normalize images to [0, 1]
    def normalize(image, label=None):
        image = tf.cast(image, tf.float32) / 255.0
        return (image, label) if label is not None else image

    dataset = dataset.map(normalize, num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def train_val_split(X_img, X_tab, time, event, test_size=0.2, seed=42):
    return train_test_split(
        X_img, X_tab, time, event,
        test_size=test_size,
        random_state=seed,
        stratify=event
    )


def create_matched_dataframe(image_dir, metadata_path, output_path="data/processed/master_dataset.csv"):
    """
    Hybrid Data Generator:
    1. Uses REAL Age/Sex/Findings from Shenzhen Metadata.
    2. Generates SYNTHETIC Survival times based on 'Findings' severity.
    """

    # 1. Load Real Metadata
    # Assuming metadata CSV has: study_id, sex, age, findings
    df = pd.read_csv(metadata_path)
    
    # Clean Sex (Male=1, Female=0)
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0})
    
    # 2. Map Images to Metadata
    # Shenzhen files usually look like "CHNCXR_0001_0.png"
    # We need to ensure study_id matches the filename or ID in filename
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    df['filename'] = image_files[:len(df)] # Simple align (Ensure sorted order matches!)

    n_samples = len(df)
    np.random.seed(42)

    # 3. Generate Missing Clinical Covariates (HIV, Diabetes, etc.)
    # (These are NOT in the metadata, so we simulate them)
    df['hiv'] = np.random.binomial(1, 0.15, n_samples) # 15% prevalence
    df['diabetes'] = np.random.binomial(1, 0.10, n_samples)
    df['bmi'] = np.random.normal(21, 3, n_samples)
    df['albumin'] = np.random.normal(3.5, 0.5, n_samples)
    df['hemoglobin'] = np.random.normal(12, 1.5, n_samples)

    # 4. Generate "Smart" Survival Outcomes based on Real Findings
    # We define keywords that indicate severe TB
    severe_keywords = ['bilateral', 'cavity', 'effusion', 'miliary', 'multiple']
    
    def calculate_risk(row):
        base_risk = 0
        # Real Data Impact: Older age = higher risk
        base_risk += (row['age'] - 40) * 0.01 
        # Real Data Impact: Findings severity
        finding_text = str(row['findings']).lower()
        if any(k in finding_text for k in severe_keywords):
            base_risk += 1.5 # High risk boost for severe findings
        
        # Synthetic Comorbidity Impact
        if row['hiv'] == 1: base_risk += 1.2
        return base_risk

    # Calculate risk score for every patient
    risk_scores = df.apply(calculate_risk, axis=1)
    
    # Convert Risk to Time (Inverse relationship)
    # Higher risk = Shorter time to event
    baseline_hazard = 0.005
    time_to_event = np.random.exponential(1 / (baseline_hazard * np.exp(risk_scores)))
    
    # Censoring (some patients survive)
    censor_time = np.random.uniform(0, time_to_event.max(), n_samples)
    
    df['time'] = np.minimum(time_to_event, censor_time)
    df['event'] = (time_to_event <= censor_time).astype(int)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Hybrid dataset created with {len(df)} samples.")
    return df
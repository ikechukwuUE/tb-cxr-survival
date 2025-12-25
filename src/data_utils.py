# src/data_utils.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import *
import os
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


def create_matched_dataframe(image_dir, output_path="data/processed/master_dataset.csv"):
    """
    Creates a synchronized dataset linking Shenzhen CXR images with synthetic clinical data.
    
    This addresses the data heterogeneity challenge highlighted in D'Souza et al., 2023,
    ensuring every image has a corresponding clinical vector for multimodal training.
    """
    import os
    import pandas as pd
    
    # 1. Get all valid image files
    valid_exts = {".png", ".jpg", ".jpeg"}
    image_files = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    image_files.sort() # Ensure deterministic order
    
    n_samples = len(image_files)
    print(f"Found {n_samples} images. Generating matching clinical data...")
    
    # 2. Generate matching clinical data
    # We utilize the synthetic generator we defined earlier
    X_clinical, time, event = generate_synthetic_tb_clinical_data(n_samples=n_samples)
    
    # 3. Combine into a Master DataFrame
    df = X_clinical.copy()
    df["image_id"] = image_files
    df["time"] = time
    df["event"] = event
    
    # Save for reproducibility
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Master dataset saved to {output_path}")
    return df
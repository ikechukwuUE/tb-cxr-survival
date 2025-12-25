import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import *

AUTOTUNE = tf.data.AUTOTUNE

def create_matched_dataframe(image_dir, metadata_path, output_path="data/processed/master_dataset.csv"):
    """
    Step 1: Master Dataset Generation
    - Loads Real Shenzhen Metadata (Age, Sex, Findings).
    - Preserves Study ID from Filenames.
    - Synthesizes missing clinical features (HIV, Diabetes, etc.).
    - Generates 'Smart' Survival Labels.
    """
    # 1. Load Real Metadata
    df = pd.read_csv(metadata_path)
    
    # 2. Align with Images
    # Shenzhen filenames: "CHNCXR_0001_0.png" -> ID is "0001"
    # We sort to ensure alignment if metadata is also sorted
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    
    # Filter metadata to match available images
    min_len = min(len(df), len(image_files))
    df = df.iloc[:min_len].copy()
    df['filename'] = image_files[:min_len]
    
    # Extract Study ID for Splitting (Prevents Leakage)
    # Extracts "0001" from "CHNCXR_0001_0.png"
    df['patient_id'] = df['filename'].apply(lambda x: x.split('_')[1] if '_' in x else x)

    # 3. Clean Real Data
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0})
    
    # 4. Generate Synthetic Clinical Data
    np.random.seed(42)
    n = len(df)
    
    df['hiv'] = np.random.binomial(1, 0.15, n)
    df['diabetes'] = np.random.binomial(1, 0.10, n)
    df['bmi'] = np.random.normal(21, 3, n).clip(15, 35)
    df['hemoglobin'] = np.random.normal(12.5, 1.5, n).clip(8, 16)
    df['albumin'] = np.random.normal(3.5, 0.5, n).clip(2.5, 5.0)
    df['smear_positive'] = np.random.binomial(1, 0.5, n)
    df['retreatment'] = np.random.binomial(1, 0.2, n)

    # 5. Generate Smart Survival Targets
    # Risk increases with Age, HIV, and Severe Findings
    severe_keywords = ['bilateral', 'cavity', 'effusion', 'miliary']
    
    def get_risk(row):
        r = 0
        r += (row['age'] - 40) * 0.01
        if any(k in str(row['findings']).lower() for k in severe_keywords):
            r += 1.5
        if row['hiv'] == 1: r += 1.2
        return r

    risk_scores = df.apply(get_risk, axis=1)
    baseline_hazard = 0.005
    time_event = np.random.exponential(1 / (baseline_hazard * np.exp(risk_scores)))
    censor = np.random.uniform(0, time_event.max(), n)
    
    df['time'] = np.minimum(time_event, censor)
    df['event'] = (time_event <= censor).astype(int)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

def split_and_standardize(df, test_size=0.2):
    """
    Step 2: Strict Preprocessing
    - Splits based on Patient ID (to prevent leakage).
    - Fits StandardScaler ONLY on Train.
    - Transforms Train and Val.
    """
    # Unique patient split
    unique_ids = df['patient_id'].unique()
    train_ids, val_ids = train_test_split(unique_ids, test_size=test_size, random_state=42)
    
    train_df = df[df['patient_id'].isin(train_ids)].copy()
    val_df = df[df['patient_id'].isin(val_ids)].copy()
    
    # Standardize Continuous Variables
    scaler = StandardScaler()
    
    # FIT on Train, TRANSFORM Train
    train_df[CONTINUOUS_COLS] = scaler.fit_transform(train_df[CONTINUOUS_COLS])
    
    # TRANSFORM Val (using Train stats)
    val_df[CONTINUOUS_COLS] = scaler.transform(val_df[CONTINUOUS_COLS])
    
    return train_df, val_df

def preprocess_image(file_path, target_size=IMG_SIZE):
    """
    Step 3: Image Standardization
    - Reads Image.
    - Resizes.
    - Applies Sample-wise Standardization (Mean=0, Std=1 per image).
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, target_size)
    
    # Per-Image Standardization
    # Allows Val set to be standardized exactly like Train without leaking stats
    img = tf.image.per_image_standardization(img)
    return img

def get_augmenter():
    """
    Step 4: Augmentation Configuration
    - Rotation and Brightness ONLY (No flipping).
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=0.1), # +/- 10% rotation
        tf.keras.layers.RandomContrast(factor=0.2)
    ])

def load_tb_cxr_dataset(dataframe, data_dir, batch_size=32, shuffle=True, augment=False):
    """
    Step 5: Pipeline Construction
    - Merges Image and Tabular Data.
    - Applies Augmentation on the fly.
    """
    file_paths = [os.path.join(data_dir, f) for f in dataframe['filename']]
    
    # Select preprocessed columns
    X_tab = dataframe[ALL_CLINICAL_COLS].values.astype("float32")
    y = dataframe[["time", "event"]].values.astype("float32")
    
    # Create Dataset
    ds = tf.data.Dataset.from_tensor_slices((file_paths, X_tab, y))

    def _mapper(path, tab, target):
        img = preprocess_image(path)
        return (img, tab), target

    # Augmentation Wrapper
    augmenter = get_augmenter()
    
    def _augment_wrapper(inputs, target):
        (img, tab) = inputs
        
        # Expand dims for Keras Layer (H,W,C) -> (1,H,W,C)
        img_batch = tf.expand_dims(img, 0)
        img_aug = augmenter(img_batch, training=True)
        img = tf.squeeze(img_aug, 0)
        
        return (img, tab), target

    ds = ds.map(_mapper, num_parallel_calls=AUTOTUNE)

    if augment:
        ds = ds.map(_augment_wrapper, num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds
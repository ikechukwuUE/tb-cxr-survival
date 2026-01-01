# src/config.py

# Configuration file for TB CXR Survival Prediction Project

# Columns to standardize (Continuous variables)
CONTINUOUS_COLS = ["age", "bmi", "hemoglobin", "albumin"]
# Columns to leave as binary/categorical
BINARY_COLS = ["sex", "hiv", "diabetes", "smear_positive", "retreatment"]
# All Clinical Columns for the model
ALL_CLINICAL_COLS = CONTINUOUS_COLS + BINARY_COLS

IMG_SIZE = (224, 224)
IMG_CHANNELS = 3

IMG_EMBED_DIM = 256
TABULAR_MULTIPLIER = 4  # hidden_dim = 4 × num_features

BATCH_SIZE = 32
EPOCHS = 100
WARMUP =10
# EPOCHS_FINE_TUNE = 30


BASE_LR = 1e-3
MIN_LR = 1e-6
# LEARNING_RATE = 1e-4
# LEARNING_RATE_FINE_TUNE = 1e-5


DROPOUT_TABULAR = 0.3
DROPOUT_FUSION = 0.45

L2_REG = 0.01

RANDOM_SEED = 42

# Just change this one string to start a fresh experiment
VERSION = "v4" 

# This updates automatically based on the version above
MODEL_NAME = f"tbsurvivalnet_{VERSION}"
FINE_TUNE_MODEL_NAME = f"{MODEL_NAME}_fine-tuned"

# Paths (Dynamic based on model name)
weights_path = f"outputs/models/{MODEL_NAME}.weights.h5"
logs_path    = f"outputs/tensorboard/{MODEL_NAME}"

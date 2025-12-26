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

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4

DROPOUT_TABULAR = 0.2
DROPOUT_FUSION = 0.3

L2_REG = 0.01

RANDOM_SEED = 42

# Just change this one string to start a fresh experiment
VERSION = "v2" 

# This updates automatically based on the version above
MODEL_NAME = f"tbsurvivalnet_{VERSION}"

# Paths (Dynamic based on model name)
weights_path = f"outputs/models/{MODEL_NAME}.weights.h5"
logs_path    = f"outputs/tensorboard/{MODEL_NAME}"

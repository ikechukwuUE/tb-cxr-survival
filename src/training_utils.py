# src/training_utils.py

import tensorflow as tf
from .survival_utils import cox_partial_likelihood

def compile_survival_model(model, lr=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=lambda y_true, y_pred: cox_partial_likelihood(
            y_true[:, 0], y_true[:, 1], y_pred
        )
    )
    return model

# src/callbacks.py

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard

# Import from our local utils
from src.survival_utils import harrell_c_index

class CIndexMetric(Callback):
    """
    Custom Callback that calculates the Concordance Index (C-Index)
    on the Validation/Test set at the end of every epoch.
    """
    def __init__(self, validation_dataset, name="val_c_index"):
        super(CIndexMetric, self).__init__()
        self.validation_dataset = validation_dataset
        self.metric_name = name
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        risk_scores = []
        true_times = []
        true_events = []
        
        # Predict on validation dataset manually to get ranking
        for (img, tab), target in self.validation_dataset:
            # Predict Log Hazard (Risk)
            pred = self.model.predict_on_batch([img, tab])
            risk_scores.append(pred.flatten())
            true_times.append(target[:, 0].numpy())
            true_events.append(target[:, 1].numpy())
            
        risk_scores = np.concatenate(risk_scores)
        true_times = np.concatenate(true_times)
        true_events = np.concatenate(true_events)
        
        # Calculate C-Index using CUSTOM function
        # Note: We pass risk_scores directly (Positive Risk = High Hazard)
        c_index = harrell_c_index(true_times, risk_scores, true_events)
        
        logs[self.metric_name] = c_index
        print(f" — {self.metric_name}: {c_index:.4f}")

class WarmupCosineDecay(Callback):
    """
    Learning Rate Scheduler:
    1. Linear Warmup: Increases LR from 0 to target_lr over 'warmup_epochs'.
    2. Cosine Decay: Smoothly decreases LR to min_lr over remaining epochs.
    """
    def __init__(self, total_epochs, warmup_epochs=5, target_lr=1e-4, min_lr=1e-6):
        super(WarmupCosineDecay, self).__init__()
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.min_lr = min_lr
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear Warmup
            lr = self.target_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine Decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.target_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            
        # FIX: Use 'learning_rate' instead of 'lr' for newer Keras versions
        if hasattr(self.model.optimizer, 'learning_rate'):
            # Check if it's a Variable (TF) or just a property
            if hasattr(self.model.optimizer.learning_rate, 'assign'):
                self.model.optimizer.learning_rate.assign(lr)
            else:
                self.model.optimizer.learning_rate = lr
        else:
            # Fallback for older Keras versions
            self.model.optimizer.lr = lr
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Log the LR so we can see it in TensorBoard
        if hasattr(self.model.optimizer, 'learning_rate'):
            logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        else:
            logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

def get_callbacks(model_name, val_data, total_epochs=50, save_dir="outputs/models", log_dir="outputs/tensorboard"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    c_index_cb = CIndexMetric(val_data)
    
    scheduler_cb = WarmupCosineDecay(
        total_epochs=total_epochs,
        warmup_epochs=5,
        target_lr=1e-4,
        min_lr=1e-6
    )
    
    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(save_dir, f"{model_name}_best.keras"),
        monitor="val_c_index",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    # # 4. Reduce LR: Lowers learning rate when stuck
    # reduce_lr = ReduceLROnPlateau(
    #     monitor="val_loss",
    #     factor=0.1,             # Multiply LR by 0.1
    #     patience=5,             # Wait 5 epochs before reducing
    #     verbose=1,
    #     min_lr=1e-6
    # )
    
    early_stopping_cb = EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1
    )
    
    tensorboard_cb = TensorBoard(log_dir=os.path.join(log_dir, model_name))

    return [c_index_cb, scheduler_cb, checkpoint_cb, early_stopping_cb, tensorboard_cb]
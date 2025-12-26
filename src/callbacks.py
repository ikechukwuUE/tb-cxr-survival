import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

def get_callbacks(model_name, save_dir="outputs/models", log_dir="outputs/tensorboard"):
    """
    Creates a comprehensive list of callbacks for training.
    
    Args:
        model_name (str): Unique name for the model (used for file naming).
        save_dir (str): Directory to save model checkpoints.
        log_dir (str): Directory to save TensorBoard logs.
        
    Returns:
        list: A list of Keras Callbacks.
    """
    
    # 1. Ensure directories exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 2. Model Checkpoint: Saves the BEST weights only
    # We save based on 'val_loss' to avoid overfitting
    checkpoint_path = os.path.join(save_dir, f"{model_name}_best.weights.h5")
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True, # Set False if you want to save the whole model
        mode="min"
    )
    
    # 3. Early Stopping: Stops training if no improvement
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,            # Wait 10 epochs before stopping
        verbose=1,
        restore_best_weights=True # Crucial: reverts to the best epoch after stopping
    )
    
    # 4. Reduce LR: Lowers learning rate when stuck
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,             # Multiply LR by 0.1
        patience=5,             # Wait 5 epochs before reducing
        verbose=1,
        min_lr=1e-6
    )
    
    # 5. TensorBoard: Visualization
    tensorboard = TensorBoard(
        log_dir=os.path.join(log_dir, model_name),
        histogram_freq=1
    )

    return [checkpoint, early_stopping, reduce_lr, tensorboard]
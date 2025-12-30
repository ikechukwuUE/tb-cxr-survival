# src/model_utils.py

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, applications
from src.config import *

def cox_ph_loss(y_true, y_pred):
    """
    Custom Cox Proportional Hazards Loss.
    """
    time = y_true[:, 0]
    event = y_true[:, 1]
    risk = y_pred[:, 0]
    
    # Risk Set Calculation
    time_i = tf.expand_dims(time, 1)
    time_j = tf.expand_dims(time, 0)
    risk_set_mask = tf.cast(time_j >= time_i, tf.float32)
    
    # Log-Sum-Exp Trick
    risk_max = tf.reduce_max(risk)
    exp_risk = tf.exp(risk - risk_max)
    masked_sum_exp = tf.reduce_sum(exp_risk * risk_set_mask, axis=1)
    log_sum_risk = tf.math.log(masked_sum_exp + 1e-8) + risk_max
    
    # Negative Partial Log Likelihood
    loss_per_sample = event * (risk - log_sum_risk)
    return -tf.reduce_mean(loss_per_sample)

def TBSurvivalNet(
    input_shape_img=(224, 224, 3), 
    num_clinical_features=9,
    l2_reg=L2_REG
):
    """
    TBSurvivalNet: A Multimodal Deep Learning Model for TB Prognosis.
    
    Combines DenseNet121 (Vision) and MLP (Clinical) using Cross-Attention.
    """
    
    # --- 1. Image Branch ---
    img_input = layers.Input(shape=input_shape_img, name="image_input")
    
    # Backbone: DenseNet121 (Frozen)
    base_model = applications.DenseNet121(
        weights="imagenet", 
        include_top=False, 
        input_tensor=img_input
    )
    base_model.trainable = False 
    
    x_img = base_model.output # (Batch, 7, 7, 1024)
    x_img = layers.Conv2D(256, (1, 1), activation='relu')(x_img) 
    x_img_seq = layers.Reshape((-1, 256))(x_img) # (Batch, 49, 256)

    # --- 2. Clinical Branch ---
    tab_input = layers.Input(shape=(num_clinical_features,), name="clinical_input")
    
    x_tab = layers.Dense(256, activation="gelu")(tab_input)
    x_tab = layers.BatchNormalization()(x_tab)
    x_tab = layers.Dropout(DROPOUT_TABULAR)(x_tab)
    x_tab_query = layers.Reshape((1, 256))(x_tab) # (Batch, 1, 256)

    # --- 3. Cross-Modal Attention Fusion ---
    # Tabular (Query) attends to Image Patches (Key/Value)
    attn_layer = layers.MultiHeadAttention(num_heads=4, key_dim=64)
    attn_out = attn_layer(query=x_tab_query, value=x_img_seq, key=x_img_seq)
    
    x_fused = layers.Add()([x_tab_query, attn_out])
    x_fused = layers.LayerNormalization()(x_fused)
    x_fused = layers.Flatten()(x_fused)

    # --- 4. Survival Head ---
    x = layers.Dense(64, activation="gelu", kernel_regularizer=regularizers.l2(l2_reg))(x_fused)
    x = layers.Dropout(DROPOUT_FUSION)(x)
    
    # Output: Log Hazard (Linear)
    output = layers.Dense(1, activation="linear", name="risk_score")(x)
    
    # Name the model explicitly "TBSurvivalNet"
    model = models.Model(inputs=[img_input, tab_input], outputs=output, name="TBSurvivalNet")
    
    return model


# src/model_utils.py (Update this function)

def unfreeze_model(model, num_layers_to_unfreeze=50, learning_rate=LEARNING_RATE_FINE_TUNE):
    """
    Unfreezes the top N layers of the CNN model for fine-tuning.
    """
    num_layers = len(model.layers)
    print(f"Total layers in model: {num_layers}")
    print(f"Unfreezing the last {num_layers_to_unfreeze} layers for fine-tuning...")
    
    # 1. Iterate through all layers
    for i, layer in enumerate(model.layers):
        # If the layer is in the last N layers...
        if i >= (num_layers - num_layers_to_unfreeze):
            
            # CRITICAL: Keep BatchNormalization layers frozen!
            # On small medical datasets, updating BN statistics during 
            # fine-tuning often destroys model performance.
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        else:
            # Ensure lower layers stay frozen
            layer.trainable = False
            
    # 2. Recompile with Low Learning Rate
    # Note: We must recompile for the 'trainable' flags to take effect
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # We assume cox_ph_loss is defined in this file or imported
    model.compile(optimizer=optimizer, loss=cox_ph_loss)
    
    print(f"Model recompiled. Fine-tuning started with LR={learning_rate}")
    return model
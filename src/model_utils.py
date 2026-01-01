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


class GatedFusion(layers.Layer):
    """
    TBSurvivalNet Gated Fusion Mechanism.
    Learns a weight 'z' (0 to 1) to dynamically balance Clinical vs. Visual features.
    Reference: 'Are Multimodal Transformers Robust to Missing Modality?' (CVPR 2022)
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.project_gate = layers.Dense(units, activation="sigmoid")
        
    def call(self, clinical_features, visual_features):
        # Concatenate both to decide the gate value
        combined = tf.concat([clinical_features, visual_features], axis=-1)
        z = self.project_gate(combined) # Gate value (Batch, 1, Units)
        
        # Weighted Sum
        # If z is 1, we trust Clinical more. If z is 0, we trust Visual more.
        return (z * clinical_features) + ((1 - z) * visual_features)

def TBSurvivalNet(
    input_shape_img=(224, 224, 3), 
    num_clinical_features=9,
    dropout_rate=DROPOUT_FUSION, # Increased dropout for small data
    l2_reg=L2_REG
):
    """
    TBSurvivalNet: EfficientNetV2 + Gated Cross-Attention.
    """
    
    # --- 1. Image Branch (EfficientNetV2B0 - SOTA Lightweight) ---
    img_input = layers.Input(shape=input_shape_img, name="image_input")
    
    # Switch to EfficientNetV2B0: Better accuracy/parameter efficiency than DenseNet
    base_model = applications.EfficientNetV2B0(
        weights="imagenet", 
        include_top=False, 
        input_tensor=img_input
    )
    base_model.trainable = False 
    
    x_img = base_model.output # (Batch, 7, 7, 1280)
    
    # Project to embedding dim
    x_img = layers.Conv2D(256, (1, 1), activation='gelu')(x_img) 
    x_img_seq = layers.Reshape((-1, IMG_EMBED_DIM))(x_img) # (Batch, 49, IMG_EMBED_DIM)

    # --- 2. Clinical Branch ---
    tab_input = layers.Input(shape=(num_clinical_features,), name="clinical_input")
    
    x_tab = layers.Dense(256, activation="gelu")(tab_input)
    x_tab = layers.BatchNormalization()(x_tab)
    x_tab = layers.Dropout(dropout_rate)(x_tab)
    x_tab_query = layers.Reshape((1, IMG_EMBED_DIM))(x_tab) # (Batch, 1, 256)

    # --- 3. Fusion: Gated Cross-Attention ---
    # A. Attention Block
    attn_layer = layers.MultiHeadAttention(num_heads=4, key_dim=64)
    attn_out = attn_layer(query=x_tab_query, value=x_img_seq, key=x_img_seq)
    
    # B. Gated Residual Connection (Instead of simple Add)
    # This allows the model to learn WHEN to attend to the image
    gate_block = GatedFusion(units=256)
    x_fused = gate_block(x_tab_query, attn_out)
    
    x_fused = layers.LayerNormalization()(x_fused)
    x_fused = layers.Flatten()(x_fused)

    # --- 4. Survival Head ---
    # Stronger regularization for the head
    x = layers.Dense(128, activation="gelu", kernel_regularizer=regularizers.l2(l2_reg))(x_fused)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation="gelu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    
    output = layers.Dense(1, activation="linear", name="risk_score")(x)
    
    model = models.Model(inputs=[img_input, tab_input], outputs=output, name=MODEL_NAME)
    
    return model


# def unfreeze_model(model, num_layers_to_unfreeze=50, learning_rate=LEARNING_RATE_FINE_TUNE):
#     """
#     Unfreezes the top N layers of the CNN model for fine-tuning.
#     """
#     num_layers = len(model.layers)
#     print(f"Total layers in model: {num_layers}")
#     print(f"Unfreezing the last {num_layers_to_unfreeze} layers for fine-tuning...")
    
#     # 1. Iterate through all layers
#     for i, layer in enumerate(model.layers):
#         # If the layer is in the last N layers...
#         if i >= (num_layers - num_layers_to_unfreeze):
            
#             # CRITICAL: Keep BatchNormalization layers frozen!
#             # On small medical datasets, updating BN statistics during 
#             # fine-tuning often destroys model performance.
#             if isinstance(layer, tf.keras.layers.BatchNormalization):
#                 layer.trainable = False
#             else:
#                 layer.trainable = True
#         else:
#             # Ensure lower layers stay frozen
#             layer.trainable = False
            
#     # 2. Recompile with Low Learning Rate
#     # Note: We must recompile for the 'trainable' flags to take effect
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
#     # We assume cox_ph_loss is defined in this file or imported
#     model.compile(optimizer=optimizer, loss=cox_ph_loss)
    
#     print(f"Model recompiled. Fine-tuning started with LR={learning_rate}")
#     # return model
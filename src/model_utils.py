import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, applications
from src.config import *
from src.survival_utils import cox_ph_loss

class GatedFusion(layers.Layer):
    """
    TBSurvivalNet Gated Fusion Mechanism.
    Learns a weight 'z' (0 to 1) to dynamically balance Clinical vs. Visual features.
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
    dropout_rate=DROPOUT_FUSION, 
    l2_reg=L2_REG
):
    """
    TBSurvivalNet: DenseNet121 + Gated Cross-Attention.
    """
    
    # --- 1. Image Branch (DenseNet121 - Robust for Medical Imaging) ---
    img_input = layers.Input(shape=input_shape_img, name="image_input")
    
    # Backbone: DenseNet121 (Pretrained on ImageNet)
    base_model = applications.DenseNet121(
        weights="imagenet", 
        include_top=False, 
        input_tensor=img_input
    )
    base_model.trainable = False 
    
    x_img = base_model.output # Shape: (Batch, 7, 7, 1024)
    
    # Project to embedding dim (1024 -> 256)
    x_img = layers.Conv2D(256, (1, 1), activation='gelu')(x_img) 
    x_img_seq = layers.Reshape((-1, IMG_EMBED_DIM))(x_img) # (Batch, 49, 256)

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
    
    # B. Gated Residual Connection
    gate_block = GatedFusion(units=256)
    x_fused = gate_block(x_tab_query, attn_out)
    
    x_fused = layers.LayerNormalization()(x_fused)
    x_fused = layers.Flatten()(x_fused)

    # --- 4. Survival Head ---
    x = layers.Dense(128, activation="gelu", kernel_regularizer=regularizers.l2(l2_reg))(x_fused)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation="gelu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    
    output = layers.Dense(1, activation="linear", name="risk_score")(x)
    
    model = models.Model(inputs=[img_input, tab_input], outputs=output, name=MODEL_NAME)
    
    return model

def unfreeze_model(model, num_layers_to_unfreeze=50, learning_rate=BASE_LR):
    """
    Unfreezes the top N layers of the backbone for fine-tuning.
    """
    # 1. Access the Backbone directly
    # In Functional API, we can find the nested model layer.
    # For DenseNet121, the layer name usually starts with 'densenet121'
    backbone = None
    for layer in model.layers:
        if 'densenet' in layer.name.lower():
            backbone = layer
            break
            
    if backbone is None:
        # Fallback for flat models (if backbone layers are directly in main model)
        # This handles the "ValueError: No such layer" we saw earlier
        print("Backbone layer not found as a unit. Unfreezing last N layers of the entire model.")
        total_layers = len(model.layers)
        for i, layer in enumerate(model.layers):
            if i >= (total_layers - num_layers_to_unfreeze):
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True
            else:
                layer.trainable = False
    else:
        # If backbone is a nested layer (Functional API standard way)
        print(f"Unfreezing DenseNet121 backbone...")
        backbone.trainable = True
        
        # Iterate through the backbone's internal layers
        # Freeze all except the last N
        total_backbone_layers = len(backbone.layers)
        for i, layer in enumerate(backbone.layers):
            if i < (total_backbone_layers - num_layers_to_unfreeze):
                layer.trainable = False
            else:
                # Keep BN frozen inside backbone too
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True

    # 2. Recompile
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=cox_ph_loss)
    
    print(f"Model recompiled. Fine-tuning started with LR={learning_rate}")
    return model
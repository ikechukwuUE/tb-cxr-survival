import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_last_conv_layer_name(model):
    """
    Automatically finds the last Convolutional layer in the model
    to attach Grad-CAM hooks.
    """
    # Iterate in reverse to find the last Conv2D
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Could not find a Conv2D layer in the model.")

def make_gradcam_heatmap(img_tensor, tab_tensor, model, last_conv_layer_name):
    """
    Computes the Grad-CAM heatmap.
    """
    # 1. Create a model that maps: [Inputs] -> [Last Conv Output, Final Prediction]
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # Cast inputs
        conv_outputs, predictions = grad_model([img_tensor, tab_tensor])
        
        # We want the gradient of the Risk Score (output 0)
        loss = predictions[:, 0]

    # 3. Calculate Gradients of the Loss w.r.t. the Conv Outputs
    grads = tape.gradient(loss, conv_outputs)

    # 4. Global Average Pooling of Gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Multiply each filter map by its importance weight
    conv_outputs = conv_outputs[0] # Remove batch dim
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Apply ReLU (we only care about positive influence)
    heatmap = tf.maximum(heatmap, 0)
    
    # Normalize if not empty
    if tf.math.reduce_max(heatmap) != 0:
        heatmap /= tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """
    Resizes heatmap and overlays it on the image.
    """
    # 1. Resize Heatmap to match Image Size (224x224)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # 2. Convert to RGB format (0-255)
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 3. Process Original Image (Handle Standardization)
    vmin, vmax = np.percentile(original_img, 1), np.percentile(original_img, 99)
    if vmax == vmin:
        img_display = original_img * 0
    else:
        img_display = (np.clip(original_img, vmin, vmax) - vmin) / (vmax - vmin)
        
    img_display = np.uint8(255 * img_display)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
    
    # 4. Overlay
    overlay = cv2.addWeighted(img_display, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Return as RGB
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

def visualize_gradcam(model, dataset, patient_idx=0):
    """
    Visualizes Grad-CAM with detailed Patient Information.
    """
    # 1. Get Data
    (images, tabular), targets = next(iter(dataset))
    
    if patient_idx >= len(images):
        print(f"Index {patient_idx} out of bounds for batch size {len(images)}")
        return

    # Prepare Inputs
    img_tensor = images[patient_idx:patient_idx+1] # (1, 224, 224, 3)
    tab_tensor = tabular[patient_idx:patient_idx+1]
    
    # 2. Get Prediction & Meta-Data
    risk_score = model.predict([img_tensor, tab_tensor], verbose=0)[0][0]
    
    # Decode Clinical Data (Indices based on ALL_CLINICAL_COLS in data_utils)
    # [Age, BMI, Hb, Alb, Sex, HIV, Diabetes, ...]
    # Sex is index 4, HIV is index 5
    is_male = tab_tensor[0, 4] > 0.5
    is_hiv = tab_tensor[0, 5] > 0.5
    
    sex_str = "Male" if is_male else "Female"
    hiv_str = "Positive" if is_hiv else "Negative"
    
    # Ground Truth
    time_true = targets[patient_idx, 0].numpy()
    event_true = targets[patient_idx, 1].numpy()
    event_str = "Event Occurred" if event_true == 1 else "Censored"

    # 3. Generate Grad-CAM
    try:
        target_layer = get_last_conv_layer_name(model)
        heatmap = make_gradcam_heatmap(img_tensor, tab_tensor, model, target_layer)
    except Exception as e:
        print(f"Grad-CAM Failed: {e}")
        return

    # 4. Create Overlay
    original_img = img_tensor[0].numpy()
    overlay = overlay_heatmap(original_img, heatmap)
    
    # 5. Visualization Layout
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Info Text Block
    info_text = (
        f"PATIENT REPORT:\n"
        f"--------------------------------\n"
        f"Risk Score:    {risk_score:.3f} (Log Hazard)\n"
        f"Survival Time: {time_true:.1f} days\n"
        f"Outcome:       {event_str}\n"
        f"--------------------------------\n"
        f"Sex: {sex_str} | HIV: {hiv_str}"
    )
    
    # Plot A: Original X-Ray
    vmin, vmax = np.percentile(original_img, 1), np.percentile(original_img, 99)
    disp_img = (np.clip(original_img, vmin, vmax) - vmin)/(vmax - vmin)
    
    axes[0].imshow(disp_img, cmap='gray')
    axes[0].set_title("Original Input (Standardized)", fontsize=12, fontweight='bold')
    axes[0].axis("off")
    
    # Plot B: Grad-CAM Overlay
    axes[1].imshow(overlay)
    axes[1].set_title(f"AI Risk Focus (Layer: {target_layer})", fontsize=12, fontweight='bold', color='darkred')
    axes[1].axis("off")
    
    # Add Text Box as Main Title
    plt.suptitle(info_text, fontsize=11, fontfamily='monospace', y=0.98, backgroundcolor='#f0f0f0')
    
    plt.tight_layout(rect=[0, 0, 1, 0.85]) # Make room for text at top
    plt.show()
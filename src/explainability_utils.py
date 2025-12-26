import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

def generate_gradcam(model, image, tabular_sample, layer_name="conv5_block16_concat", alpha=0.5):
    """
    Generates a Grad-CAM heatmap overlaid on the original image with robust normalization.
    """
    
    # --- 1. Validation & Setup ---
    # Check if layer exists to avoid crashing
    layer_names = [layer.name for layer in model.layers]
    if layer_name not in layer_names:
        print(f"Warning: Layer '{layer_name}' not found. Switching to last available layer: '{layer_names[-1]}'")
        layer_name = layer_names[-1]

    # --- 2. Build Gradient Model ---
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(layer_name).output,
            model.output
        ]
    )

    # --- 3. Prepare Inputs ---
    # Ensure inputs are tensors with batch dim
    img_tensor = tf.expand_dims(image, 0)
    tab_tensor = tf.expand_dims(tabular_sample, 0)

    # --- 4. Gradient Tape ---
    with tf.GradientTape() as tape:
        conv_out, prediction = grad_model([img_tensor, tab_tensor])
        
        # Handle output shape (if list or single tensor)
        score = prediction[0] if isinstance(prediction, list) else prediction

    # --- 5. Compute Gradients & Weights ---
    grads = tape.gradient(score, conv_out)
    
    # Global Average Pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # --- 6. Generate Heatmap ---
    conv_out = conv_out[0] # Remove batch dim
    
    # Weight the feature maps
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU (max with 0)
    heatmap = tf.maximum(heatmap, 0)

    # Normalize (Safe division)
    heatmap_max = tf.math.reduce_max(heatmap)
    if heatmap_max == 0:
        heatmap_max = 1e-10 # Prevent division by zero
    heatmap /= heatmap_max
    
    heatmap = heatmap.numpy()

    # --- 7. Visualization Preparation ---
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert to JET colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # --- 8. FIX: Robust Image Deprocessing ---
    # This fixes the "dark silhouette" issue by finding the real min/max of the image
    img_data = image.numpy()
    img_data = img_data - np.min(img_data)  # Shift to start at 0
    img_data = img_data / (np.max(img_data) + 1e-7) # Scale to [0, 1]
    original_img = np.uint8(img_data * 255) # Scale to [0, 255] and convert

    # --- 9. Superimpose ---
    superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)

    # --- 10. Plotting ---
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original CXR (Restored)")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    
    # Display predicted risk score
    pred_val = score.numpy().flatten()[0] if hasattr(score, 'numpy') else float(score)
    plt.title(f"Grad-CAM\nPredicted Risk: {pred_val:.4f}")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
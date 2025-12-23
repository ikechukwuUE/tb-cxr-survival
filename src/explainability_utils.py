# src/explainability_utils.py

import tensorflow as tf
import matplotlib.pyplot as plt

def generate_gradcam(model, image, tabular_dim, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.image_encoder.get_layer(layer_name).output,
            model.output
        ]
    )

    dummy_tab = tf.zeros((1, tabular_dim))

    with tf.GradientTape() as tape:
        conv_out, risk = grad_model(
            (tf.expand_dims(image, 0), dummy_tab)
        )

    grads = tape.gradient(risk, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")
    plt.title("Grad-CAM (Risk Contribution)")
    plt.show()

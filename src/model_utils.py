# src/model_utils.py

import tensorflow as tf
from tensorflow.keras import layers

class TBSurvivalNet(tf.keras.Model):
    def __init__(
        self,
        image_encoder,
        tabular_dim,
        img_embed_dim=256,
        tabular_hidden_dim=None
    ):
        super().__init__()
        self.image_encoder = image_encoder

        if tabular_hidden_dim is None:
            tabular_hidden_dim = max(32, 4 * tabular_dim)

        self.tabular_net = tf.keras.Sequential([
            layers.Dense(tabular_hidden_dim, activation=None),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.2)
        ])

        self.image_projection = tf.keras.Sequential([
            layers.Dense(img_embed_dim, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3)
        ])

        fusion_dim = img_embed_dim + tabular_hidden_dim

        self.fusion = tf.keras.Sequential([
            layers.Dense(fusion_dim // 2, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation=None)
        ])

    def call(self, inputs, training=False):
        images, tabular = inputs
        img_features = self.image_projection(
            self.image_encoder(images, training=training),
            training=training
        )
        tab_features = self.tabular_net(tabular, training=training)
        combined = tf.concat([img_features, tab_features], axis=1)
        return tf.squeeze(self.fusion(combined, training=training), axis=1)

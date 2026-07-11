import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    TimeDistributed,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    Bidirectional,
    LSTM
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

from models.attention import Attention


def build_model():

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    inputs = Input(shape=(16,224,224,3))

    x = TimeDistributed(base_model)(inputs)

    x = TimeDistributed(
        GlobalAveragePooling2D()
    )(x)

    x = Bidirectional(
        LSTM(
            128,
            return_sequences=True
        )
    )(x)

    x = Attention()(x)

    x = Dense(
        128,
        activation="relu"
    )(x)

    x = Dropout(0.5)(x)

    outputs = Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(inputs, outputs)

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model


if __name__ == "__main__":

    model = build_model()
    model.summary()
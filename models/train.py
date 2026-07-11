import os
import tensorflow as tf

from preprocessing.data_generator import VideoDataGenerator
from models.model import build_model

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger
)

os.makedirs("saved_models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

train_generator = VideoDataGenerator(
    split="train",
    batch_size=8
)

val_generator = VideoDataGenerator(
    split="val",
    batch_size=8
)

model = build_model()

callbacks = [

    ModelCheckpoint(
        "saved_models/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1
    ),

    CSVLogger(
        "logs/training_log.csv"
    )
]

history = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=20,

    callbacks=callbacks
)

model.save("saved_models/final_model.keras")
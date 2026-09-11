import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_model(seq_len, feature_dim, num_classes):
    inputs = keras.Input(shape=(seq_len, feature_dim))
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.GRU(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.GRU(32)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

    with open(os.path.join(DATA_DIR, "preprocess_stats.json")) as f:
        stats = json.load(f)
    words = stats["words"]
    print(f"Classes: {words}")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    model = build_model(X_train.shape[1], X_train.shape[2], len(words))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=15, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    random_chance = 1.0 / len(words)
    print(f"\nFinal val_accuracy: {val_acc:.3f} (random chance: {random_chance:.3f})")

    model_path = os.path.join(MODEL_DIR, "asl_model.keras")
    model.save(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()

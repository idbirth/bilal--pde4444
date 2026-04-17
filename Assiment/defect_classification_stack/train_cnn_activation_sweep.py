from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from common import (
    Metrics,
    ensure_dir,
    load_split_as_arrays,
    save_classification_report,
    save_confusion_matrix_plot,
    save_metrics,
    set_seed,
)


def activation_block(x: tf.Tensor, activation_name: str) -> tf.Tensor:
    if activation_name == "leaky_relu":
        return tf.keras.layers.LeakyReLU(negative_slope=0.1)(x)
    if activation_name == "gelu":
        return tf.keras.layers.Activation(tf.nn.gelu)(x)
    return tf.keras.layers.Activation(activation_name)(x)


def build_cnn(input_shape: tuple[int, ...], activation_name: str) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.03)(x)
    x = tf.keras.layers.RandomZoom(0.05)(x)

    for filters in [32, 64, 128]:
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = activation_block(x, activation_name)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = activation_block(x, activation_name)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"cnn_{activation_name}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def evaluate_predictions(y_test: np.ndarray, y_pred: np.ndarray, model_name: str) -> Metrics:
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    accuracy = float((y_pred == y_test).mean())
    return Metrics(model_name=model_name, accuracy=accuracy, precision=float(precision), recall=float(recall), f1=float(f1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the same CNN with different activation functions.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    X_train, y_train, class_names = load_split_as_arrays(args.data_dir, "train", size=(224, 224), grayscale=False)
    X_val, y_val, _ = load_split_as_arrays(args.data_dir, "val", size=(224, 224), grayscale=False)
    X_test, y_test, _ = load_split_as_arrays(args.data_dir, "test", size=(224, 224), grayscale=False)

    class_weights_arr = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weight = {int(i): float(w) for i, w in zip(np.unique(y_train), class_weights_arr)}

    activation_names = ["relu", "elu", "gelu", "selu", "leaky_relu"]
    all_metrics: list[Metrics] = []

    for activation_name in activation_names:
        run_dir = ensure_dir(output_dir / activation_name)
        model = build_cnn(input_shape=X_train.shape[1:], activation_name=activation_name)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )

        test_probs = model.predict(X_test, verbose=0).ravel()
        y_pred = (test_probs >= 0.5).astype(int)
        metrics = evaluate_predictions(y_test, y_pred, model_name=f"cnn_{activation_name}")
        all_metrics.append(metrics)

        save_metrics(metrics, run_dir / "metrics.json")
        save_confusion_matrix_plot(y_test, y_pred, class_names, run_dir / "confusion_matrix.png", f"CNN {activation_name} Confusion Matrix")
        save_classification_report(y_test, y_pred, class_names, run_dir / "classification_report.csv")
        pd.DataFrame(history.history).to_csv(run_dir / "training_history.csv", index=False)
        model.save(run_dir / "model.keras")

    summary = pd.DataFrame([m.to_dict() for m in all_metrics]).sort_values("f1", ascending=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()

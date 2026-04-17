from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras_tuner as kt
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


def build_model(hp: kt.HyperParameters, input_shape: tuple[int, ...]) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.BatchNormalization())

    num_layers = hp.Int("num_dense_layers", min_value=1, max_value=3)
    for i in range(num_layers):
        units = hp.Choice(f"units_{i}", values=[64, 128, 256, 512])
        activation = hp.Choice(f"activation_{i}", values=["relu", "elu", "gelu", "selu"])
        dropout = hp.Float(f"dropout_{i}", min_value=0.0, max_value=0.5, step=0.1)
        model.add(tf.keras.layers.Dense(units))
        if activation == "gelu":
            model.add(tf.keras.layers.Activation(tf.nn.gelu))
        else:
            model.add(tf.keras.layers.Activation(activation))
        model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    learning_rate = hp.Choice("learning_rate", values=[1e-4, 3e-4, 1e-3])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorFlow/Keras MLP with Random Search hyperparameter tuning.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    X_train, y_train, class_names = load_split_as_arrays(args.data_dir, "train", size=(224, 224), grayscale=False)
    X_val, y_val, _ = load_split_as_arrays(args.data_dir, "val", size=(224, 224), grayscale=False)
    X_test, y_test, _ = load_split_as_arrays(args.data_dir, "test", size=(224, 224), grayscale=False)

    class_weights_arr = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weight = {int(i): float(w) for i, w in zip(np.unique(y_train), class_weights_arr)}

    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model(hp, input_shape=X_train.shape[1:]),
        objective="val_accuracy",
        max_trials=args.max_trials,
        seed=args.seed,
        directory=str(output_dir / "tuner"),
        project_name="keras_mlp_random_search",
        overwrite=True,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    tuner.search(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    history = best_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    test_probs = best_model.predict(X_test, verbose=0).ravel()
    y_pred = (test_probs >= 0.5).astype(int)

    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    accuracy = float((y_pred == y_test).mean())

    metrics = Metrics(
        model_name="keras_mlp_random_search",
        accuracy=accuracy,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )

    save_metrics(metrics, output_dir / "metrics.json")
    save_confusion_matrix_plot(y_test, y_pred, class_names, output_dir / "confusion_matrix.png", "Keras MLP Confusion Matrix")
    save_classification_report(y_test, y_pred, class_names, output_dir / "classification_report.csv")
    pd.DataFrame(history.history).to_csv(output_dir / "training_history.csv", index=False)

    with (output_dir / "best_hyperparameters.json").open("w", encoding="utf-8") as f:
        json.dump(best_hps.values, f, indent=2)

    best_model.save(output_dir / "best_model.keras")
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()

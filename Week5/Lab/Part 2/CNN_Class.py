"""
Classification baseline built from Part2_CNN.py.
"""

from pathlib import Path

import tensorflow as tf


def load_datasets(
    data_dir: Path,
    batch_size: int = 6,
    img_height: int = 244,
    img_width: int = 244,
    validation_split: float = 0.3,
    seed: int = 123,
):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def build_classification_model(img_height: int, img_width: int, num_classes: int):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(img_height, img_width, 3)),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "D1"

    batch_size = 6
    img_height = 244
    img_width = 244
    epochs = 8

    image_count = len(list(data_dir.glob("*/*.JPG"))) + len(list(data_dir.glob("*/*.jpg")))

    print(f"Dataset path: {data_dir}")
    print(f"Image count: {image_count}")

    train_ds, val_ds, class_names = load_datasets(
        data_dir=data_dir,
        batch_size=batch_size,
        img_height=img_height,
        img_width=img_width,
    )

    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")

    model = build_classification_model(
        img_height=img_height,
        img_width=img_width,
        num_classes=num_classes,
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.summary()

    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Training history keys: {list(history.history.keys())}")


if __name__ == "__main__":
    main()

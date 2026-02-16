"""
Converted from: Week5/Lab/Part 2/Part 2_CNN.html
"""

from pathlib import Path

import PIL
import PIL.Image
import tensorflow as tf


def main():
    # define the data path to access the dataset
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "D1"
    print(data_dir.name)

    # print the number of images in the dataset
    image_count = len(list(data_dir.glob("*/*.jpg")))
    print(image_count)

    # split into train and test datasets with following parameters
    batch_size = 6
    img_height = 244
    img_width = 244

    # image size: Size to resize images to after they are read from disk
    # batch_size: Size of the batches of data
    # seed: Optional random seed for shuffling
    # validation_split: fraction of data to reserve for validation
    # subset: One of "training" or "validation"
    # class_names: explicit list of class names (must match subdirectories)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    num_classes = 8  # number of labels (classes) in our dataset

    # Rescaling normalizes image pixel values to [0, 1]
    # Flatten reshapes 2D feature maps to 1D before Dense layers
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(16, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(16, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(16, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # train the model
    model.fit(train_ds, validation_data=val_ds, epochs=8)

    model.summary()

    # test using the validation dataset
    results = model.evaluate(val_ds)
    print("Evaluation [loss, accuracy]:", results)

    # Lab task:
    # 1) Try tuning the CNN structure to improve the accuracy
    # 2) Try on a different dataset


if __name__ == "__main__":
    main()

import argparse
import csv
import io
import json
import os
import random
import time
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent
_MPL_CACHE_DIR = _BASE_DIR / ".mplconfig"
_CACHE_DIR = _BASE_DIR / ".cache"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def build_search_configs():
    return [
        {
            "name": "strong_flatten_baseline",
            "img_height": 128,
            "img_width": 128,
            "batch_size": 8,
            "conv_filters": [32, 64, 128],
            "head_type": "flatten",
            "dense_units": 256,
            "dropout_rate": 0.30,
            "use_batch_norm": False,
            "use_augmentation": False,
            "learning_rate": 0.001,
            "early_stopping_patience": 8,
            "lr_plateau_patience": 4,
        },
        {
            "name": "flatten_with_augmentation",
            "img_height": 128,
            "img_width": 128,
            "batch_size": 8,
            "conv_filters": [32, 64, 128],
            "head_type": "flatten",
            "dense_units": 256,
            "dropout_rate": 0.30,
            "use_batch_norm": False,
            "use_augmentation": True,
            "learning_rate": 0.001,
            "early_stopping_patience": 8,
            "lr_plateau_patience": 4,
        },
        {
            "name": "flatten_batch_norm",
            "img_height": 128,
            "img_width": 128,
            "batch_size": 8,
            "conv_filters": [32, 64, 128],
            "head_type": "flatten",
            "dense_units": 256,
            "dropout_rate": 0.30,
            "use_batch_norm": True,
            "use_augmentation": True,
            "learning_rate": 0.001,
            "early_stopping_patience": 8,
            "lr_plateau_patience": 4,
        },
        {
            "name": "flatten_lower_lr",
            "img_height": 128,
            "img_width": 128,
            "batch_size": 8,
            "conv_filters": [32, 64, 128],
            "head_type": "flatten",
            "dense_units": 256,
            "dropout_rate": 0.25,
            "use_batch_norm": True,
            "use_augmentation": True,
            "learning_rate": 0.0005,
            "early_stopping_patience": 8,
            "lr_plateau_patience": 4,
        },
        {
            "name": "larger_input_global_avg",
            "img_height": 128,
            "img_width": 128,
            "batch_size": 8,
            "conv_filters": [32, 64, 128],
            "head_type": "global_avg",
            "dense_units": 128,
            "dropout_rate": 0.35,
            "use_batch_norm": True,
            "use_augmentation": True,
            "learning_rate": 0.0005,
            "early_stopping_patience": 8,
            "lr_plateau_patience": 4,
        },
    ]


def build_stratified_split(data_dir, val_split=0.3, seed=123):
    rng = random.Random(seed)
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    split_summary = {"train": {}, "val": {}}

    for label_id, class_dir in enumerate(class_dirs):
        files = sorted([f for f in class_dir.iterdir() if f.is_file()])
        rng.shuffle(files)
        if not files:
            continue

        val_count = int(round(len(files) * val_split))
        val_count = max(1, val_count)
        val_count = min(val_count, len(files) - 1)

        class_val = files[:val_count]
        class_train = files[val_count:]

        train_paths.extend([str(p) for p in class_train])
        train_labels.extend([label_id] * len(class_train))
        val_paths.extend([str(p) for p in class_val])
        val_labels.extend([label_id] * len(class_val))

        split_summary["train"][class_dir.name] = len(class_train)
        split_summary["val"][class_dir.name] = len(class_val)

    train_pairs = list(zip(train_paths, train_labels))
    val_pairs = list(zip(val_paths, val_labels))
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    train_paths, train_labels = zip(*train_pairs)
    val_paths, val_labels = zip(*val_pairs)

    return {
        "class_names": class_names,
        "train_paths": list(train_paths),
        "train_labels": list(train_labels),
        "val_paths": list(val_paths),
        "val_labels": list(val_labels),
        "split_summary": split_summary,
    }


def dataset_from_paths(tf, split_data, img_height, img_width, batch_size, seed=123):
    def _load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [img_height, img_width])
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices(
        (split_data["train_paths"], split_data["train_labels"])
    )
    train_ds = train_ds.shuffle(
        len(split_data["train_paths"]), seed=seed, reshuffle_each_iteration=True
    )
    train_ds = train_ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (split_data["val_paths"], split_data["val_labels"])
    )
    val_ds = val_ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)

    return train_ds, val_ds


def build_model(tf, num_classes, config):
    layers = [tf.keras.layers.Rescaling(1.0 / 255)]

    if config["use_augmentation"]:
        layers.append(
            tf.keras.Sequential(
                [
                    tf.keras.layers.RandomFlip("horizontal"),
                    tf.keras.layers.RandomRotation(0.08),
                    tf.keras.layers.RandomZoom(0.1),
                ],
                name="data_augmentation",
            )
        )

    for filters in config["conv_filters"]:
        layers.append(tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"))
        if config["use_batch_norm"]:
            layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.MaxPooling2D())

    if config.get("head_type", "global_avg") == "flatten":
        layers.append(tf.keras.layers.Flatten())
    else:
        layers.append(tf.keras.layers.GlobalAveragePooling2D())
    layers.append(tf.keras.layers.Dense(config["dense_units"], activation="relu"))
    layers.append(tf.keras.layers.Dropout(config["dropout_rate"]))
    layers.append(tf.keras.layers.Dense(num_classes))

    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def to_jsonable_config(config):
    out = dict(config)
    out["conv_filters"] = list(out["conv_filters"])
    return out


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_history_csv(path, history_dict):
    keys = list(history_dict.keys())
    rows = []
    if keys:
        for i in range(len(history_dict[keys[0]])):
            row = {"epoch": i + 1}
            for key in keys:
                row[key] = history_dict[key][i]
            rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch"] + keys)
        writer.writeheader()
        writer.writerows(rows)


def model_summary_text(model):
    buf = io.StringIO()
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    return buf.getvalue()


def plot_iteration_curves(history_data, output_path):
    epochs = list(range(1, len(history_data.get("loss", [])) + 1))
    if not epochs:
        return
    plt.figure(figsize=(8, 5))
    if "accuracy" in history_data:
        plt.plot(epochs, history_data["accuracy"], marker="o", label="train_accuracy")
    if "val_accuracy" in history_data:
        plt.plot(epochs, history_data["val_accuracy"], marker="o", label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve Per Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_val_accuracy_by_iteration(results, output_path):
    if not results:
        return
    labels = [f"{row['iteration']:02d}" for row in results]
    values = [row["best_val_accuracy"] for row in results]
    plt.figure(figsize=(8, 5))
    plt.plot(labels, values, marker="o")
    for i, value in enumerate(values):
        plt.text(i, value + 0.01, f"{value:.3f}", ha="center", fontsize=8)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Iteration")
    plt.ylabel("Best Validation Accuracy")
    plt.title("Best Validation Accuracy by Iteration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_change_log(configs):
    change_log = []
    prev = None
    for i, cfg in enumerate(configs, start=1):
        current = to_jsonable_config(cfg)
        if prev is None:
            changes = {k: {"from": None, "to": v} for k, v in current.items() if k != "name"}
        else:
            changes = {}
            for key, value in current.items():
                if key == "name":
                    continue
                if prev.get(key) != value:
                    changes[key] = {"from": prev.get(key), "to": value}
        change_log.append({"iteration": i, "name": cfg["name"], "changes": changes})
        prev = current
    return change_log


def ensure_tensorflow():
    try:
        import tensorflow as tf  # noqa: WPS433
    except ModuleNotFoundError:
        return None
    return tf


def tune_models(base_dir, max_iterations, epochs, init_only=False):
    data_dir = base_dir / "D1"
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    history_root = base_dir / "tuing_history"
    runs_dir = history_root / "run"
    history_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    configs = build_search_configs()[:max_iterations]
    write_json(history_root / "changing_values.json", build_change_log(configs))
    write_json(
        history_root / "search_space.json",
        [to_jsonable_config(cfg) for cfg in build_search_configs()],
    )

    split_data = build_stratified_split(data_dir, val_split=0.3, seed=123)
    write_json(history_root / "split_summary.json", split_data["split_summary"])

    for iteration, config in enumerate(configs, start=1):
        run_dir = runs_dir / f"iteration_{iteration:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "config.json", to_jsonable_config(config))
        if not (run_dir / "metrics.json").exists():
            write_json(
                run_dir / "metrics.json",
                {
                    "iteration": iteration,
                    "name": config["name"],
                    "status": "initialized_not_run",
                },
            )

    if init_only:
        print(f"Initialized tuning folders and configs at: {history_root}")
        print("Stratified split summary:")
        print(json.dumps(split_data["split_summary"], indent=2))
        return

    tf = ensure_tensorflow()
    if tf is None:
        raise ModuleNotFoundError(
            "TensorFlow is not installed in this Python environment. "
            "Install TensorFlow, then re-run Part2_CNN_tuned.py without --init-only."
        )

    overall_results = []
    best_result = None

    for iteration, config in enumerate(configs, start=1):
        run_dir = runs_dir / f"iteration_{iteration:02d}"

        print(f"\n=== Iteration {iteration:02d}: {config['name']} ===")
        print(f"Config: {to_jsonable_config(config)}")
        print(
            "Split sizes:",
            f"train={len(split_data['train_paths'])},",
            f"val={len(split_data['val_paths'])}",
        )

        train_ds, val_ds = dataset_from_paths(
            tf=tf,
            split_data=split_data,
            img_height=config["img_height"],
            img_width=config["img_width"],
            batch_size=config["batch_size"],
            seed=123,
        )

        class_names = split_data["class_names"]
        autotune = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(256, seed=123).prefetch(buffer_size=autotune)
        val_ds = val_ds.cache().prefetch(buffer_size=autotune)

        tf.keras.utils.set_random_seed(123)
        model = build_model(tf=tf, num_classes=len(class_names), config=config)
        (run_dir / "model_summary.txt").write_text(model_summary_text(model), encoding="utf-8")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=config.get("early_stopping_patience", 8),
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=config.get("lr_plateau_patience", 4),
                min_lr=1e-6,
            ),
        ]

        start = time.time()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2,
        )
        elapsed_sec = time.time() - start

        eval_loss, eval_accuracy = model.evaluate(val_ds, verbose=0)

        history_data = history.history
        best_val_accuracy = max(history_data.get("val_accuracy", [0.0]))
        best_epoch = history_data.get("val_accuracy", [0.0]).index(best_val_accuracy) + 1
        train_accuracy_last = history_data.get("accuracy", [0.0])[-1]
        val_accuracy_last = history_data.get("val_accuracy", [0.0])[-1]

        write_history_csv(run_dir / "history.csv", history_data)
        plot_iteration_curves(history_data, run_dir / "val_accuracy_curve.png")
        metrics = {
            "iteration": iteration,
            "name": config["name"],
            "epochs_requested": epochs,
            "epochs_ran": len(history_data.get("loss", [])),
            "best_epoch": best_epoch,
            "train_accuracy_last": float(train_accuracy_last),
            "val_accuracy_last": float(val_accuracy_last),
            "best_val_accuracy": float(best_val_accuracy),
            "eval_loss": float(eval_loss),
            "eval_accuracy": float(eval_accuracy),
            "elapsed_seconds": round(elapsed_sec, 2),
        }
        write_json(run_dir / "metrics.json", metrics)

        row = {
            "iteration": iteration,
            "name": config["name"],
            "img_height": config["img_height"],
            "img_width": config["img_width"],
            "batch_size": config["batch_size"],
            "conv_filters": "-".join(str(x) for x in config["conv_filters"]),
            "head_type": config.get("head_type", "global_avg"),
            "dense_units": config["dense_units"],
            "dropout_rate": config["dropout_rate"],
            "use_batch_norm": config["use_batch_norm"],
            "use_augmentation": config["use_augmentation"],
            "learning_rate": config["learning_rate"],
            "best_val_accuracy": round(float(best_val_accuracy), 4),
            "eval_accuracy": round(float(eval_accuracy), 4),
            "eval_loss": round(float(eval_loss), 4),
            "epochs_ran": len(history_data.get("loss", [])),
            "elapsed_seconds": round(elapsed_sec, 2),
        }
        overall_results.append(row)

        if best_result is None or row["eval_accuracy"] > best_result["eval_accuracy"]:
            best_result = row

        print(
            f"Iteration {iteration:02d} done | "
            f"best_val_accuracy={row['best_val_accuracy']:.4f}, "
            f"eval_accuracy={row['eval_accuracy']:.4f}"
        )

    leaderboard_path = history_root / "leaderboard.csv"
    fieldnames = [
        "iteration",
        "name",
        "img_height",
        "img_width",
        "batch_size",
        "conv_filters",
        "head_type",
        "dense_units",
        "dropout_rate",
        "use_batch_norm",
        "use_augmentation",
        "learning_rate",
        "best_val_accuracy",
        "eval_accuracy",
        "eval_loss",
        "epochs_ran",
        "elapsed_seconds",
    ]
    with leaderboard_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(overall_results)
    plot_val_accuracy_by_iteration(
        overall_results, history_root / "val_accuracy_per_iteration.png"
    )

    if best_result is not None:
        write_json(history_root / "best_run.json", best_result)
        print("\nBest run summary:")
        print(json.dumps(best_result, indent=2))

    print(f"\nSaved tuning history to: {history_root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Tune CNN for Week 5 Part 2 (D1 dataset)")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="How many predefined configurations to run (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs per iteration (default: 20)",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Initialize tuing_history files/folders without training",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    tune_models(
        base_dir=base_dir,
        max_iterations=args.max_iterations,
        epochs=args.epochs,
        init_only=args.init_only,
    )


if __name__ == "__main__":
    main()

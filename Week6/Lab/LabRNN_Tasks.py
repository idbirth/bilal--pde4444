#!/usr/bin/env python3
"""Week 6 Lab: RNN/LSTM sentiment classification task completion.

This script is a completed Python version of the notebook `LabRNN-Tasks.ipynb`.
It performs:
1) Data loading and preprocessing
2) Text tokenization and sequence padding
3) LSTM model creation and training
4) Inference on a sample sentence
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


RANDOM_SEED = 42
MAX_WORDS = 5000
MAX_SEQUENCE_LEN = 200
EMBEDDING_VECTOR_LENGTH = 32
BATCH_SIZE = 64
EPOCHS = 5


def load_and_prepare_data(csv_path: Path):
    """Load dataset and complete preprocessing steps from the lab."""
    df = pd.read_csv(csv_path, names=["seq", "brand", "sentiment", "text"])

    # Select only target and text columns.
    data = df[["sentiment", "text"]].copy()

    print("All sentiments:")
    print(data["sentiment"].value_counts())

    # Keep only binary sentiment labels.
    data = data[data["sentiment"] != "Neutral"]
    data = data[data["sentiment"] != "Irrelevant"]

    print("\nBinary sentiments only:")
    print(data["sentiment"].value_counts())

    # Replace missing text values with blank space.
    data["text"] = data["text"].fillna(" ")

    # Convert labels to numeric values.
    sentiment_label = data["sentiment"].factorize()
    data["label"] = sentiment_label[0]
    label_names = sentiment_label[1]

    print("\nLabel mapping:")
    for idx, name in enumerate(label_names):
        print(f"  {idx}: {name}")

    return data, label_names


def preprocess_text(input_text):
    """Tokenize and pad text sequences."""
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(input_text)

    sequence_data = tokenizer.texts_to_sequences(input_text)
    sequence_data = pad_sequences(
        sequence_data,
        maxlen=MAX_SEQUENCE_LEN,
        padding="post",
        truncating="post",
    )

    return tokenizer, sequence_data


def build_model(vocab_size: int):
    """Build LSTM model based on the lab instructions."""
    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_VECTOR_LENGTH, input_length=MAX_SEQUENCE_LEN))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def predict_sentence(model, tokenizer, sentence: str, label_names):
    """Run inference for one sentence and decode the class label."""
    sentence_seq = tokenizer.texts_to_sequences([sentence])
    sentence_pad = pad_sequences(
        sentence_seq,
        maxlen=MAX_SEQUENCE_LEN,
        padding="post",
        truncating="post",
    )

    score = float(model.predict(sentence_pad, verbose=0)[0][0])
    predicted_class_id = int(score >= 0.5)
    predicted_label = label_names[predicted_class_id]

    return score, predicted_class_id, predicted_label


def main():
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    csv_path = Path(__file__).resolve().parent / "twitter_training.csv"

    data, label_names = load_and_prepare_data(csv_path)

    # Extract input sequence from text column.
    input_data = data["text"].values
    y = data["label"].values

    # Tokenization + sequence preprocessing.
    tokenizer, x = preprocess_text(input_data)

    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    model = build_model(vocab_size)

    print("\nModel summary:")
    model.summary()

    print("\nTraining...")
    history = model.fit(
        x,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1,
    )

    final_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"\nFinal training accuracy: {final_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")

    # Test sentence from the lab task.
    test_sentence = "the quality of this product is very bad"
    score, class_id, label = predict_sentence(model, tokenizer, test_sentence, label_names)

    print("\nTest sentence:")
    print(f"  {test_sentence}")
    print("Prediction:")
    print(f"  score={score:.4f}, class_id={class_id}, sentiment={label}")


if __name__ == "__main__":
    main()

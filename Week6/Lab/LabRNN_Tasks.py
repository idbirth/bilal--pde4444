#!/usr/bin/env python3
"""Week 6 Lab: RNN/LSTM sentiment classification task completion.

This script is a completed Python version of the notebook `LabRNN-Tasks.ipynb`.
It performs:
1) Data loading and preprocessing
2) Text tokenization and sequence padding
3) LSTM model creation and training
4) Inference on a sample sentence
"""

# Step 0: Import required libraries.
# - pathlib: to build a reliable path to the CSV file
# - numpy/tensorflow: numeric and deep learning operations
# - pandas: data loading and cleaning
# - keras modules: tokenization, padding, model layers, and training
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# Step 1: Define global settings/hyperparameters used in the full pipeline.
RANDOM_SEED = 42
MAX_WORDS = 5000
MAX_SEQUENCE_LEN = 200
EMBEDDING_VECTOR_LENGTH = 32
BATCH_SIZE = 64
EPOCHS = 5



def load_and_prepare_data(csv_path: Path):
    """Load dataset and complete preprocessing steps from the lab."""
    # Step 2.1: Read CSV with the exact column names requested in the lab task.
    df = pd.read_csv(csv_path, names=["seq", "brand", "sentiment", "text"])

    # Step 2.2: Keep only the two columns used for sentiment classification.
    data = df[["sentiment", "text"]].copy()

    # Step 2.3: Show class distribution before filtering.
    print("All sentiments:")
    print(data["sentiment"].value_counts())

    # Step 2.4: Remove Neutral and Irrelevant classes to make this a binary task.
    data = data[data["sentiment"] != "Neutral"]
    data = data[data["sentiment"] != "Irrelevant"]

    # Step 2.5: Show class distribution after filtering.
    print("\nBinary sentiments only:")
    print(data["sentiment"].value_counts())

    # Step 2.6: Replace missing text with a blank string so tokenizer can process all rows.
    data["text"] = data["text"].fillna(" ")

    # Step 2.7: Convert string labels (Positive/Negative) to integer labels (0/1).
    # factorize() returns:
    # - encoded integers for each row
    # - an index-like array containing the original label names
    sentiment_label = data["sentiment"].factorize()
    data["label"] = sentiment_label[0]
    label_names = sentiment_label[1]

    # Step 2.8: Print mapping so we know which numeric value means which sentiment.
    print("\nLabel mapping:")
    for idx, name in enumerate(label_names):
        print(f"  {idx}: {name}")

    return data, label_names


def preprocess_text(input_text):
    """Tokenize and pad text sequences."""
    # Step 3.1: Create tokenizer with vocabulary cap (5000 words) and OOV token.
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")

    # Step 3.2: Build the word index from training text.
    tokenizer.fit_on_texts(input_text)

    # Step 3.3: Convert each sentence into a sequence of word IDs.
    sequence_data = tokenizer.texts_to_sequences(input_text)

    # Step 3.4: Pad/truncate all sequences to a fixed length (200).
    # This makes input shape consistent for LSTM.
    sequence_data = pad_sequences(
        sequence_data,
        maxlen=MAX_SEQUENCE_LEN,
        padding="post",
        truncating="post",
    )

    return tokenizer, sequence_data


def build_model(vocab_size: int):
    """Build LSTM model based on the lab instructions."""
    # Step 4.1: Initialize a sequential model (layer-by-layer stack).
    model = Sequential()

    # Step 4.2: Embedding layer turns each token ID into a dense vector.
    model.add(Embedding(vocab_size, EMBEDDING_VECTOR_LENGTH, input_length=MAX_SEQUENCE_LEN))

    # Step 4.3: Spatial dropout drops whole embedding dimensions to reduce overfitting.
    model.add(SpatialDropout1D(0.2))

    # Step 4.4: LSTM captures sequential patterns in text.
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

    # Step 4.5: Extra dropout regularization before output layer.
    model.add(Dropout(0.2))

    # Step 4.6: Single sigmoid output neuron for binary classification.
    model.add(Dense(1, activation="sigmoid"))

    # Step 4.7: Compile with binary cross-entropy and accuracy metric.
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def predict_sentence(model, tokenizer, sentence: str, label_names):
    """Run inference for one sentence and decode the class label."""
    # Step 6.1: Apply the same preprocessing as training data.
    sentence_seq = tokenizer.texts_to_sequences([sentence])
    sentence_pad = pad_sequences(
        sentence_seq,
        maxlen=MAX_SEQUENCE_LEN,
        padding="post",
        truncating="post",
    )

    # Step 6.2: Predict probability score (0 to 1).
    score = float(model.predict(sentence_pad, verbose=0)[0][0])

    # Step 6.3: Convert probability to class ID using threshold 0.5.
    predicted_class_id = int(score >= 0.5)

    # Step 6.4: Convert class ID back to readable label.
    predicted_label = label_names[predicted_class_id]

    return score, predicted_class_id, predicted_label


def main():
    # Step 1.1: Set random seeds for reproducibility.
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Step 1.2: Build dataset path relative to this script's location.
    csv_path = Path(__file__).resolve().parent / "twitter_training.csv"

    # Step 2: Load and preprocess tabular data.
    data, label_names = load_and_prepare_data(csv_path)

    # Step 3: Extract features (text) and target labels.
    input_data = data["text"].values
    y = data["label"].values

    # Step 3.5: Convert text into padded numeric sequences.
    tokenizer, x = preprocess_text(input_data)

    # Step 4.8: Calculate effective vocabulary size for embedding input dimension.
    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)

    # Step 4.9: Build the LSTM model.
    model = build_model(vocab_size)

    # Step 4.10: Print model architecture.
    print("\nModel summary:")
    model.summary()

    # Step 5: Train model on the prepared dataset.
    print("\nTraining...")
    history = model.fit(
        x,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1,
    )

    # Step 5.1: Show final train/validation accuracy from the last epoch.
    final_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"\nFinal training accuracy: {final_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")

    # Step 6: Test the trained model with an example sentence from the lab.
    test_sentence = "the quality of this product is very bad"
    score, class_id, label = predict_sentence(model, tokenizer, test_sentence, label_names)

    # Step 6.5: Print human-readable prediction results.
    print("\nTest sentence:")
    print(f"  {test_sentence}")
    print("Prediction:")
    print(f"  score={score:.4f}, class_id={class_id}, sentiment={label}")


# Standard Python entry point guard.
if __name__ == "__main__":
    main()

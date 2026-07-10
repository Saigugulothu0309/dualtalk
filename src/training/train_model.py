import os
import pickle
import sys
from math import ceil

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.gestures.label_policy import canonicalize_label
from src.utils.feature_extraction import FEATURE_COUNT, extract_feature_matrix

DATA_PATH = os.path.join(BASE_DIR, "data", "hand_landmarks.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.pkl")
MIN_SAMPLES_PER_LABEL = 20
TEST_SIZE = 0.2


def load_dataset():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        sys.exit(1)

    try:
        return pd.read_csv(DATA_PATH)
    except Exception as exc:
        print(f"Error: Failed to read dataset from {DATA_PATH}: {exc}")
        sys.exit(1)


def canonicalize_labels(data):
    data = data.copy()
    data["label"] = data["label"].apply(canonicalize_label)
    return data


def get_label_counts(data):
    if "label" not in data.columns:
        print("Error: Dataset must contain a 'label' column.")
        sys.exit(1)

    label_counts = data["label"].value_counts().sort_index()
    if label_counts.empty:
        print("Error: Dataset is empty. Collect gesture samples before training.")
        sys.exit(1)

    return label_counts


def print_label_distribution(label_counts, title):
    print(title)
    for label, count in label_counts.items():
        print(f"{label}: {count}")


def warn_low_sample_counts(label_counts):
    low_sample_counts = label_counts[label_counts < MIN_SAMPLES_PER_LABEL]
    if low_sample_counts.empty:
        return

    print(f"Warning: Some classes have fewer than {MIN_SAMPLES_PER_LABEL} samples.")
    for label, count in low_sample_counts.items():
        print(f"Warning: {label} has only {count} samples.")


def balance_dataset(data):
    label_counts = get_label_counts(data)
    print_label_distribution(label_counts, "Training distribution:")
    warn_low_sample_counts(label_counts)

    max_count = label_counts.max()
    balanced_frames = []

    for label, count in label_counts.items():
        df_label = data[data["label"] == label]
        if count < max_count:
            df_label = resample(
                df_label,
                replace=True,
                n_samples=max_count,
                random_state=42,
            )
        balanced_frames.append(df_label)

    balanced_data = (
        pd.concat(balanced_frames)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    balanced_counts = get_label_counts(balanced_data)
    print_label_distribution(balanced_counts, "Balanced training distribution:")
    return balanced_data


def shuffle_dataset(data):
    return data.sample(frac=1, random_state=42).reset_index(drop=True)


def normalize_features(data):
    x_values = data.drop("label", axis=1).to_numpy(dtype=float)
    return extract_feature_matrix(x_values)


def get_stratify_labels(y_values):
    label_counts = y_values.value_counts()
    class_count = len(label_counts)
    test_count = ceil(len(y_values) * TEST_SIZE)
    train_count = len(y_values) - test_count

    if label_counts.min() >= 2 and test_count >= class_count and train_count >= class_count:
        return y_values

    print("Warning: Dataset is too small for stratified split; using shuffled split only.")
    return None


def train_model(data):
    data = shuffle_dataset(data)
    dataset_counts = get_label_counts(data)
    print_label_distribution(dataset_counts, "Dataset distribution:")

    y_values = data["label"]
    stratify_labels = get_stratify_labels(y_values)

    train_data, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=42,
        shuffle=True,
        stratify=stratify_labels,
    )

    balanced_train_data = balance_dataset(train_data)
    x_train = normalize_features(balanced_train_data)
    y_train = balanced_train_data["label"]
    x_test = normalize_features(test_data)
    y_test = test_data["label"]
    print(f"Feature count: {x_train.shape[1]} / {FEATURE_COUNT}")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy_score: {accuracy:.4f}")
    labels = list(model.classes_)
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    matrix_frame = pd.DataFrame(matrix, index=labels, columns=labels)
    print("confusion_matrix (rows=true, columns=predicted):")
    print(matrix_frame.to_string())

    return model


def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)
    print(f"Model trained and saved to {MODEL_PATH}")


def main():
    data = canonicalize_labels(load_dataset())
    model = train_model(data)
    save_model(model)


if __name__ == "__main__":
    main()

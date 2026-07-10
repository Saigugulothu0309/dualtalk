import csv
import os
from collections import Counter
from datetime import datetime


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GESTURE_TEST_LOG_PATH = os.path.join(BASE_DIR, "logs", "gesture_test.csv")
GESTURE_TEST_FIELDNAMES = [
    "timestamp",
    "expected_gesture",
    "predicted_gesture",
    "confidence",
    "correct",
]
TRAINED_TEST_GESTURES = (
    "YES",
    "HOLD",
    "WAIT",
    "HELP",
    "CALL",
    "COME HERE",
    "DRINK",
    "EAT",
    "BATHROOM / RESTROOM / TOILET",
    "THANK YOU",
    "THANKS",
    "DANGER",
    "OK / SUPER",
)
PREDICTION_LABEL_TO_TEST_GESTURE = {
    "bathroom": "BATHROOM / RESTROOM / TOILET",
    "bathroom/restrom/toilet": "BATHROOM / RESTROOM / TOILET",
    "bathroom/restroom/toilet": "BATHROOM / RESTROOM / TOILET",
    "call": "CALL",
    "come here": "COME HERE",
    "danger": "DANGER",
    "drink": "DRINK",
    "eat": "EAT",
    "help": "HELP",
    "hold": "HOLD",
    "ok": "OK / SUPER",
    "ok / super": "OK / SUPER",
    "ok/super": "OK / SUPER",
    "stop": "DANGER",
    "super": "OK / SUPER",
    "thank you": "THANK YOU",
    "thanks": "THANKS",
    "unsure": "UNSURE",
    "unknown": "UNKNOWN",
    "wait": "WAIT",
    "yes": "YES",
}
TEST_GESTURE_ALIASES = {
    "BATHROOM": "BATHROOM / RESTROOM / TOILET",
    "BATHROOM / RESTROOM / TOILET": "BATHROOM / RESTROOM / TOILET",
    "BATHROOM/RESTROM/TOILET": "BATHROOM / RESTROOM / TOILET",
    "BATHROOM/RESTROOM/TOILET": "BATHROOM / RESTROOM / TOILET",
    "CALL": "CALL",
    "COME HERE": "COME HERE",
    "DANGER": "DANGER",
    "DRINK": "DRINK",
    "EAT": "EAT",
    "HELP": "HELP",
    "HOLD": "HOLD",
    "OK / SUPER": "OK / SUPER",
    "OK SUPER": "OK / SUPER",
    "OK/SUPER": "OK / SUPER",
    "RESTROOM": "BATHROOM / RESTROOM / TOILET",
    "THANK YOU": "THANK YOU",
    "THANKS": "THANKS",
    "TOILET": "BATHROOM / RESTROOM / TOILET",
    "WAIT": "WAIT",
    "YES": "YES",
}
TEST_GESTURE_TO_APP_GESTURE = {
    "YES": "YES",
    "HOLD": "HOLD",
    "WAIT": "HOLD",
    "HELP": "HELP",
    "CALL": "COME HERE",
    "COME HERE": "COME HERE",
    "DRINK": "DRINK",
    "EAT": "EAT",
    "BATHROOM / RESTROOM / TOILET": "BATHROOM",
    "THANK YOU": "THANK YOU",
    "THANKS": "THANK YOU",
    "DANGER": "STOP",
    "OK / SUPER": None,
}


def normalize_test_gesture_label(label):
    if label is None:
        return None

    normalized = str(label).strip().replace("_", " ")
    normalized = " ".join(normalized.split()).upper()
    if not normalized:
        return None
    return TEST_GESTURE_ALIASES.get(normalized, normalized)


def format_test_prediction_label(label):
    if label is None:
        return "UNKNOWN"

    normalized = " ".join(str(label).strip().split()).lower()
    if not normalized:
        return "UNKNOWN"

    mapped_label = PREDICTION_LABEL_TO_TEST_GESTURE.get(normalized)
    if mapped_label is not None:
        return mapped_label

    return str(label).strip().upper()


def get_app_gesture_for_test_label(label):
    normalized_label = normalize_test_gesture_label(label)
    if normalized_label is None:
        return None
    return TEST_GESTURE_TO_APP_GESTURE.get(normalized_label)


class GestureValidationSession:
    def __init__(
        self,
        *,
        log_path=GESTURE_TEST_LOG_PATH,
        expected_gestures=TRAINED_TEST_GESTURES,
        start_gesture=None,
    ):
        self.log_path = log_path
        self.expected_gestures = tuple(expected_gestures)
        self.records = []
        self._initialized = False
        self.current_index = self.resolve_index(start_gesture)

    def resolve_index(self, gesture_label):
        if not self.expected_gestures:
            return 0
        if gesture_label is None:
            return 0

        normalized_label = normalize_test_gesture_label(gesture_label)
        if normalized_label is None:
            return 0
        try:
            return self.expected_gestures.index(normalized_label)
        except ValueError:
            return 0

    @property
    def current_expected_gesture(self):
        if not self.expected_gestures:
            return None
        return self.expected_gestures[self.current_index]

    @property
    def total_attempts(self):
        return len(self.records)

    def advance(self, step=1):
        if not self.expected_gestures:
            return None
        self.current_index = (self.current_index + step) % len(self.expected_gestures)
        return self.current_expected_gesture

    def count_for_gesture(self, gesture_label=None):
        expected_gesture = normalize_test_gesture_label(
            gesture_label or self.current_expected_gesture
        )
        total = 0
        correct = 0
        for record in self.records:
            if record["expected_gesture"] != expected_gesture:
                continue
            total += 1
            correct += int(bool(record["correct"]))
        return correct, total

    def accuracy_text_for_gesture(self, gesture_label=None):
        correct, total = self.count_for_gesture(gesture_label)
        return f"{correct}/{total}"

    def record_prediction(self, predicted_gesture, confidence, *, expected_gesture=None):
        normalized_expected = normalize_test_gesture_label(
            expected_gesture or self.current_expected_gesture
        )
        normalized_predicted = format_test_prediction_label(predicted_gesture)
        correct = normalized_expected == normalized_predicted
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "expected_gesture": normalized_expected,
            "predicted_gesture": normalized_predicted,
            "confidence": float(confidence or 0.0),
            "correct": correct,
        }
        self.records.append(record)
        self._write_record(record)
        return record

    def build_accuracy_summary(self):
        return [
            f"{gesture}: {self.accuracy_text_for_gesture(gesture)}"
            for gesture in self.expected_gestures
        ]

    def build_confusion_summary(self):
        confusion_counts = Counter()
        for record in self.records:
            if record["correct"]:
                continue
            confusion_counts[
                (record["expected_gesture"], record["predicted_gesture"])
            ] += 1

        return [
            f"{expected} -> {predicted}: {count}"
            for (expected, predicted), count in sorted(
                confusion_counts.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
        ]

    def _ensure_log_ready(self):
        if self._initialized:
            return

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        file_exists = (
            os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 0
        )
        if not file_exists:
            with open(self.log_path, "a", newline="", encoding="utf-8") as file_obj:
                writer = csv.DictWriter(
                    file_obj,
                    fieldnames=GESTURE_TEST_FIELDNAMES,
                )
                writer.writeheader()
        self._initialized = True

    def _write_record(self, record):
        self._ensure_log_ready()
        with open(self.log_path, "a", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=GESTURE_TEST_FIELDNAMES)
            writer.writerow(
                {
                    "timestamp": record["timestamp"],
                    "expected_gesture": record["expected_gesture"],
                    "predicted_gesture": record["predicted_gesture"],
                    "confidence": f"{record['confidence']:.4f}",
                    "correct": str(bool(record["correct"])).lower(),
                }
            )

import csv
import os
import sys

import cv2
import mediapipe as mp


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.gestures.label_policy import canonicalize_label

DATA_PATH = os.path.join(BASE_DIR, "data", "hand_landmarks.csv")
LANDMARK_COUNT = 21
MIN_DETECTION_CONFIDENCE = 0.7
VISIBILITY_MARGIN = 0.02
MAX_LANDMARK_JUMP = 0.15


def build_header():
    header = ["label"]
    for index in range(1, LANDMARK_COUNT + 1):
        header.extend([f"x{index}", f"y{index}", f"z{index}"])
    return header


def ensure_dataset_file():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) == 0:
        with open(DATA_PATH, "w", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file).writerow(build_header())


def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    values = []

    for landmark in landmarks:
        values.extend(
            [
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z,
            ]
        )

    max_abs = max(abs(value) for value in values) or 1.0
    return [value / max_abs for value in values]


def landmarks_to_points(landmarks):
    return [(landmark.x, landmark.y, landmark.z) for landmark in landmarks]


def is_hand_fully_visible(landmarks):
    if len(landmarks) < LANDMARK_COUNT:
        return False

    return all(
        VISIBILITY_MARGIN <= landmark.x <= 1.0 - VISIBILITY_MARGIN
        and VISIBILITY_MARGIN <= landmark.y <= 1.0 - VISIBILITY_MARGIN
        for landmark in landmarks
    )


def get_detection_confidence(results, hand_index):
    if results.multi_handedness is None or hand_index >= len(results.multi_handedness):
        return 0.0

    handedness = results.multi_handedness[hand_index]
    if handedness is None or not handedness.classification:
        return 0.0

    return handedness.classification[0].score


def get_best_hand(results):
    if not results.multi_hand_landmarks:
        return None, 0.0

    best_index = max(
        range(len(results.multi_hand_landmarks)),
        key=lambda hand_index: get_detection_confidence(results, hand_index),
    )
    return results.multi_hand_landmarks[best_index], get_detection_confidence(
        results,
        best_index,
    )


def get_max_landmark_jump(current_points, previous_points):
    if previous_points is None:
        return 0.0

    return max(
        (
            (current_x - previous_x) ** 2
            + (current_y - previous_y) ** 2
            + (current_z - previous_z) ** 2
        )
        ** 0.5
        for (current_x, current_y, current_z), (previous_x, previous_y, previous_z) in zip(
            current_points,
            previous_points,
        )
    )


def prompt_label():
    while True:
        label = canonicalize_label(input("Gesture label: "))
        if label:
            return label
        print("Please enter a label name.")


def prompt_sample_count():
    while True:
        value = input("Number of samples: ").strip()
        try:
            sample_count = int(value)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if sample_count > 0:
            return sample_count
        print("Please enter a number greater than 0.")


def draw_status(
    frame,
    label,
    saved_samples,
    target_samples,
    skipped_samples,
    quality_status,
):
    cv2.putText(
        frame,
        f"Label: {label}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Samples: {saved_samples}/{target_samples}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Skipped: {skipped_samples}",
        (10, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        quality_status,
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Press q to stop",
        (10, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def collect_samples(label, target_samples):
    ensure_dataset_file()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    saved_samples = 0
    skipped_samples = 0
    previous_landmark_points = None
    quality_status = "Waiting for hand"

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Unable to open webcam. Check your camera device.")

    try:
        with open(DATA_PATH, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)

            with mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_DETECTION_CONFIDENCE,
            ) as hands:
                while saved_samples < target_samples:
                    success, frame = capture.read()
                    if not success:
                        print("Could not read a frame from the webcam.")
                        break

                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb_frame)

                    hand_landmarks, detection_confidence = get_best_hand(results)

                    if hand_landmarks is not None:
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                        )

                        landmark_points = landmarks_to_points(hand_landmarks.landmark)

                        if detection_confidence < MIN_DETECTION_CONFIDENCE:
                            skipped_samples += 1
                            quality_status = (
                                f"Skipped: confidence {detection_confidence:.2f} < "
                                f"{MIN_DETECTION_CONFIDENCE:.2f}"
                            )
                        elif not is_hand_fully_visible(hand_landmarks.landmark):
                            skipped_samples += 1
                            quality_status = "Skipped: hand not fully visible"
                        else:
                            max_jump = get_max_landmark_jump(
                                landmark_points,
                                previous_landmark_points,
                            )
                            previous_landmark_points = landmark_points

                            if max_jump > MAX_LANDMARK_JUMP:
                                skipped_samples += 1
                                quality_status = (
                                    f"Skipped: unstable landmarks jump {max_jump:.2f}"
                                )
                            else:
                                row = [label] + normalize_landmarks(hand_landmarks.landmark)
                                writer.writerow(row)
                                csv_file.flush()
                                saved_samples += 1
                                quality_status = (
                                    f"Saved: confidence {detection_confidence:.2f}, "
                                    f"jump {max_jump:.2f}"
                                )
                    else:
                        quality_status = "Waiting for hand"

                    draw_status(
                        frame,
                        label,
                        saved_samples,
                        target_samples,
                        skipped_samples,
                        quality_status,
                    )
                    cv2.imshow("DualTalk - Collect Gesture Data", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return saved_samples


def main():
    print(f"Saving samples to: {DATA_PATH}")

    while True:
        label = prompt_label()
        target_samples = prompt_sample_count()
        saved_samples = collect_samples(label, target_samples)

        print(f"Saved {saved_samples}/{target_samples} samples for '{label}'.")

        collect_more = input("Collect another gesture? [y/N]: ").strip().lower()
        if collect_more not in {"y", "yes"}:
            break


if __name__ == "__main__":
    main()

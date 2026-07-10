import os
import pickle
import sys
from collections import deque

import cv2
import mediapipe as mp


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.gestures.gesture_rules import (
    detect_gesture,
    detect_no_motion,
    get_index_finger_tip_position,
    is_no_handshape,
    is_open_palm,
)
from src.gestures.label_policy import canonicalize_prediction_label
from src.utils.feature_extraction import FEATURE_COUNT, extract_features_from_landmarks

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.pkl")
CONFIDENCE_THRESHOLD = 0.75
STABLE_FRAME_COUNT = 5
WAVE_BUFFER_SIZE = 12
WAVE_MIN_HORIZONTAL_RANGE = 0.08
WAVE_MIN_TOTAL_MOTION = 0.16
WAVE_MIN_DIRECTION_CHANGES = 1
WAVE_MIN_DELTA = 0.01
PALM_CENTER_LANDMARKS = (0, 5, 9, 13, 17)
NO_BUFFER_SIZE = 8
NO_STABLE_REQUIRED_FRAMES = 3
NO_LATCH_FRAMES = STABLE_FRAME_COUNT + NO_STABLE_REQUIRED_FRAMES


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}; continuing with rule-based detection only")
        return None

    try:
        with open(MODEL_PATH, "rb") as model_file:
            model = pickle.load(model_file)
        model_feature_count = getattr(model, "n_features_in_", None)
        if model_feature_count is not None and model_feature_count != FEATURE_COUNT:
            print(
                f"Model expects {model_feature_count} features, but realtime extraction "
                f"now produces {FEATURE_COUNT}; retrain the model before using ML prediction"
            )
            return None
        print(f"Loaded model from {MODEL_PATH}")
        return model
    except Exception:
        print(f"Could not load model from {MODEL_PATH}; continuing with rule-based detection only")
        return None


def extract_features(landmarks):
    return extract_features_from_landmarks(landmarks)


def predict_model_candidate(model, landmarks):
    if model is None or not hasattr(model, "predict_proba"):
        return None, 0.0

    features = extract_features(landmarks)

    try:
        probabilities = model.predict_proba([features])[0]
        best_index = max(range(len(probabilities)), key=lambda i: probabilities[i])
        confidence = probabilities[best_index]
        return str(model.classes_[best_index]).strip(), confidence
    except Exception:
        return None, 0.0


def predict_with_model(model, landmarks):
    predicted_label, confidence = predict_model_candidate(model, landmarks)
    predicted_gesture = canonicalize_prediction_label(predicted_label)
    if confidence >= CONFIDENCE_THRESHOLD and predicted_gesture:
        return predicted_gesture, confidence
    try:
        return "UNKNOWN", float(confidence)
    except (TypeError, ValueError):
        return "UNKNOWN", 0.0


def get_palm_center_x(landmarks):
    return sum(landmarks[index].x for index in PALM_CENTER_LANDMARKS) / len(
        PALM_CENTER_LANDMARKS
    )


def detect_wave_motion(x_positions):
    if len(x_positions) < 6:
        return False

    deltas = [
        x_positions[index] - x_positions[index - 1]
        for index in range(1, len(x_positions))
    ]
    significant_signs = [
        1 if delta > 0 else -1
        for delta in deltas
        if abs(delta) >= WAVE_MIN_DELTA
    ]

    if len(significant_signs) < 2:
        return False

    direction_changes = sum(
        1
        for index in range(1, len(significant_signs))
        if significant_signs[index] != significant_signs[index - 1]
    )
    horizontal_range = max(x_positions) - min(x_positions)
    total_motion = sum(abs(delta) for delta in deltas)

    return (
        horizontal_range >= WAVE_MIN_HORIZONTAL_RANGE
        and total_motion >= WAVE_MIN_TOTAL_MOTION
        and direction_changes >= WAVE_MIN_DIRECTION_CHANGES
    )


def detect_no_for_hand(
    landmarks,
    hand_key,
    no_x_buffers,
    no_y_buffers,
    no_latches,
    no_stable_counts,
):
    if hand_key not in no_x_buffers:
        no_x_buffers[hand_key] = deque(maxlen=NO_BUFFER_SIZE)
        no_y_buffers[hand_key] = deque(maxlen=NO_BUFFER_SIZE)
        no_latches[hand_key] = 0
        no_stable_counts[hand_key] = 0

    index_x, index_y = get_index_finger_tip_position(landmarks)
    wrist_x = float(landmarks[0].x)
    if not is_no_handshape(landmarks):
        if (
            no_x_buffers[hand_key]
            or no_y_buffers[hand_key]
            or no_stable_counts[hand_key] > 0
            or no_latches[hand_key] > 0
        ):
            print("INDEX TIP X:", f"{index_x:.3f}")
            print("WRIST X:", f"{wrist_x:.3f}")
            print("NO MOTION:", False)
            print("STABLE NO COUNT:", 0)
        no_x_buffers[hand_key].clear()
        no_y_buffers[hand_key].clear()
        no_latches[hand_key] = 0
        no_stable_counts[hand_key] = 0
        return False

    no_x_buffers[hand_key].append(index_x - wrist_x)
    no_y_buffers[hand_key].append(index_y)
    motion_detected = detect_no_motion(
        no_x_buffers[hand_key],
        no_y_buffers[hand_key],
    )

    if motion_detected:
        no_stable_counts[hand_key] = min(
            NO_STABLE_REQUIRED_FRAMES,
            no_stable_counts[hand_key] + 1,
        )
    else:
        no_stable_counts[hand_key] = max(0, no_stable_counts[hand_key] - 1)

    print("INDEX TIP X:", f"{index_x:.3f}")
    print("WRIST X:", f"{wrist_x:.3f}")
    print("NO MOTION:", motion_detected)
    print("STABLE NO COUNT:", no_stable_counts[hand_key])

    if no_stable_counts[hand_key] >= NO_STABLE_REQUIRED_FRAMES:
        if no_latches[hand_key] == 0:
            print("NO STABLE")
            print("NO DETECTED")
        no_latches[hand_key] = NO_LATCH_FRAMES

    if no_latches[hand_key] > 0:
        no_latches[hand_key] -= 1
        return True
    return False


def update_stable_prediction(
    prediction_buffer,
    stable_prediction,
    current_gesture,
    current_confidence,
):
    prediction_buffer.append((current_gesture, current_confidence))

    if len(prediction_buffer) < STABLE_FRAME_COUNT:
        return stable_prediction

    gestures = [gesture for gesture, _ in prediction_buffer]
    if gestures[0] == "UNKNOWN" or any(gesture != gestures[0] for gesture in gestures):
        return stable_prediction

    average_confidence = sum(confidence for _, confidence in prediction_buffer) / len(
        prediction_buffer
    )
    return gestures[0], average_confidence


def speak_gesture(engine, gesture_text, last_spoken):
    return last_spoken


def get_display_gesture(gesture_text, confidence):
    if gesture_text == "No Hand":
        return gesture_text

    if confidence < CONFIDENCE_THRESHOLD:
        return "UNSURE"

    return gesture_text


def draw_prediction(frame, hand_label, gesture_text, confidence, y_offset):
    display_gesture = get_display_gesture(gesture_text, confidence)
    confidence_percent = confidence * 100
    text_color = (0, 255, 0)
    if display_gesture == "UNSURE":
        text_color = (0, 165, 255)

    cv2.putText(
        frame,
        f"{hand_label}: {display_gesture}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        text_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Confidence: {confidence_percent:.0f}%",
        (10, y_offset + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run_realtime():
    model = load_model()
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    engine = None
    last_spoken_by_hand = {}
    stable_predictions = {}
    prediction_buffers = {}
    wave_buffers = {}
    no_x_buffers = {}
    no_y_buffers = {}
    no_latches = {}
    no_stable_counts = {}

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Unable to open webcam. Check your camera device.")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)
            hand_predictions = []

            if result.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    if hand_landmarks is None or len(hand_landmarks.landmark) < 21:
                        continue

                    hand_label = f"Hand {idx + 1}"
                    if result.multi_handedness is not None and idx < len(result.multi_handedness):
                        handedness = result.multi_handedness[idx]
                        if handedness and handedness.classification:
                            classification = handedness.classification[0]
                            score = classification.score
                            if score < 0.5:
                                continue
                            hand_label = classification.label

                    landmarks = hand_landmarks.landmark
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    hand_key = hand_label
                    if hand_key not in prediction_buffers:
                        prediction_buffers[hand_key] = deque(maxlen=STABLE_FRAME_COUNT)
                        stable_predictions[hand_key] = ("UNKNOWN", 0.0)
                        wave_buffers[hand_key] = deque(maxlen=WAVE_BUFFER_SIZE)

                    if is_open_palm(landmarks):
                        wave_buffers[hand_key].append(get_palm_center_x(landmarks))
                    else:
                        wave_buffers[hand_key].clear()

                    if is_open_palm(landmarks) and detect_wave_motion(wave_buffers[hand_key]):
                        gesture_text = "WAVE"
                    elif detect_no_for_hand(
                        landmarks,
                        hand_key,
                        no_x_buffers,
                        no_y_buffers,
                        no_latches,
                        no_stable_counts,
                    ):
                        gesture_text = "NO"
                    else:
                        gesture_text = detect_gesture(landmarks)

                    if gesture_text == "UNKNOWN":
                        gesture_text, confidence = predict_with_model(model, landmarks)
                    else:
                        confidence = 1.0

                    stable_predictions[hand_key] = update_stable_prediction(
                        prediction_buffers[hand_key],
                        stable_predictions[hand_key],
                        gesture_text,
                        confidence,
                    )
                    gesture_text, confidence = stable_predictions[hand_key]

                    if gesture_text != "UNKNOWN":
                        spoken_text = f"{hand_label} {gesture_text}"
                        last_spoken_by_hand[hand_key] = speak_gesture(
                            engine,
                            spoken_text,
                            last_spoken_by_hand.get(hand_key, ""),
                        )
                    hand_predictions.append((hand_label, gesture_text, confidence))

            if not hand_predictions:
                draw_prediction(frame, "Hands", "No Hand", 0.0, 50)
            else:
                for prediction_index, (hand_label, gesture_text, confidence) in enumerate(
                    hand_predictions
                ):
                    draw_prediction(
                        frame,
                        hand_label,
                        gesture_text,
                        confidence,
                        50 + prediction_index * 80,
                    )
            cv2.imshow("DualTalk - Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        hands.close()
        capture.release()
        cv2.destroyAllWindows()


def main():
    run_realtime()


if __name__ == "__main__":
    main()

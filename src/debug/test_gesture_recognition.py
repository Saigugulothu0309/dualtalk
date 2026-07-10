import os
import sys
import time
from collections import deque

import cv2
import mediapipe as mp


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import get as config_get
from src.gestures.gesture_rules import detect_gesture, is_open_palm
from src.inference.run_realtime import (
    detect_no_for_hand,
    WAVE_BUFFER_SIZE,
    detect_wave_motion,
    get_palm_center_x,
    load_model,
    NO_BUFFER_SIZE,
    predict_with_model,
)


WINDOW_NAME = "DualTalk - Gesture Recognition Test"
STABLE_FRAME_TARGET = 15
HAND_CONFIDENCE_THRESHOLD = 0.5


def get_hand_label_and_score(result, hand_index):
    if result.multi_handedness is None or hand_index >= len(result.multi_handedness):
        return f"Hand {hand_index + 1}", 1.0

    handedness = result.multi_handedness[hand_index]
    if handedness is None or not handedness.classification:
        return f"Hand {hand_index + 1}", 1.0

    classification = handedness.classification[0]
    return classification.label, classification.score


def detect_current_gesture(
    model,
    result,
    wave_buffers,
    no_x_buffers,
    no_y_buffers,
    no_latches,
    no_stable_counts,
):
    if not result.multi_hand_landmarks:
        wave_buffers.clear()
        no_x_buffers.clear()
        no_y_buffers.clear()
        no_latches.clear()
        no_stable_counts.clear()
        return None, 0.0

    best_gesture = None
    best_confidence = 0.0

    for hand_index, hand_landmarks in enumerate(result.multi_hand_landmarks):
        if hand_landmarks is None or len(hand_landmarks.landmark) < 21:
            continue

        hand_label, hand_score = get_hand_label_and_score(result, hand_index)
        if hand_score < HAND_CONFIDENCE_THRESHOLD:
            continue

        landmarks = hand_landmarks.landmark
        hand_key = hand_label
        if hand_key not in wave_buffers:
            wave_buffers[hand_key] = deque(maxlen=WAVE_BUFFER_SIZE)
        if hand_key not in no_x_buffers:
            no_x_buffers[hand_key] = deque(maxlen=NO_BUFFER_SIZE)
            no_y_buffers[hand_key] = deque(maxlen=NO_BUFFER_SIZE)
            no_latches[hand_key] = 0
            no_stable_counts[hand_key] = 0

        if is_open_palm(landmarks):
            wave_buffers[hand_key].append(get_palm_center_x(landmarks))
        else:
            wave_buffers[hand_key].clear()

        if is_open_palm(landmarks) and detect_wave_motion(wave_buffers[hand_key]):
            detected_gesture = "WAVE"
            confidence = 1.0
        elif detect_no_for_hand(
            landmarks,
            hand_key,
            no_x_buffers,
            no_y_buffers,
            no_latches,
            no_stable_counts,
        ):
            detected_gesture = "NO"
            confidence = 1.0
        else:
            detected_gesture = detect_gesture(landmarks)
            if detected_gesture == "UNKNOWN":
                detected_gesture, confidence = predict_with_model(model, landmarks)
            else:
                confidence = 1.0

        if (
            best_gesture is None
            or confidence > best_confidence
            or (best_gesture == "UNKNOWN" and detected_gesture != "UNKNOWN")
        ):
            best_gesture = detected_gesture
            best_confidence = confidence

    if best_gesture is None:
        wave_buffers.clear()
        return None, 0.0

    return best_gesture, best_confidence


def run_gesture_test():
    model = load_model()
    mp_hands = mp.solutions.hands
    wave_buffers = {}
    no_x_buffers = {}
    no_y_buffers = {}
    no_latches = {}
    no_stable_counts = {}

    last_gesture = None
    stable_gesture = None
    frame_count = 0

    capture = cv2.VideoCapture(int(config_get("camera.index", 0)))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(config_get("camera.width", 1280)))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config_get("camera.height", 720)))
    capture.set(cv2.CAP_PROP_FPS, int(config_get("camera.fps", 30)))
    if not capture.isOpened():
        raise RuntimeError("Unable to open webcam. Check your camera device.")

    with mp_hands.Hands(
        max_num_hands=int(config_get("detection.max_num_hands", 2)),
        min_detection_confidence=float(
            config_get("detection.detection_confidence", 0.7)
        ),
        min_tracking_confidence=float(
            config_get("detection.tracking_confidence", 0.7)
        ),
    ) as hands:
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    print("Could not read a frame from the webcam.")
                    time.sleep(0.1)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)

                detected_gesture, confidence = detect_current_gesture(
                    model,
                    result,
                    wave_buffers,
                    no_x_buffers,
                    no_y_buffers,
                    no_latches,
                    no_stable_counts,
                )

                if detected_gesture is None:
                    print("NO HAND")
                    last_gesture = None
                    stable_gesture = None
                    frame_count = 0
                else:
                    if detected_gesture != last_gesture:
                        if last_gesture is not None:
                            print("GESTURE CHANGED")
                        last_gesture = detected_gesture
                        stable_gesture = None
                        frame_count = 1
                    else:
                        frame_count = min(frame_count + 1, STABLE_FRAME_TARGET)

                    if frame_count >= STABLE_FRAME_TARGET:
                        if stable_gesture != detected_gesture:
                            print(f"{detected_gesture} CONFIRMED")
                        stable_gesture = detected_gesture
                    else:
                        stable_gesture = None

                print("RAW:", detected_gesture)
                print("STABLE:", stable_gesture)
                print("COUNT:", frame_count)

                display_gesture = detected_gesture if detected_gesture is not None else "No Hand"
                display_confidence = f"{confidence * 100:.0f}%"
                display_count = f"{frame_count}/{STABLE_FRAME_TARGET}"

                cv2.putText(
                    frame,
                    f"Gesture: {display_gesture}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Confidence: {display_confidence}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Stable: {display_count}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()


def main():
    run_gesture_test()


if __name__ == "__main__":
    main()

import argparse
import os
import sys
import time
from collections import deque

import cv2
import mediapipe as mp


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.communication.sender import (
    EMPTY_VALUE,
    HAND_CONFIDENCE_THRESHOLD,
    WINDOW_NAME,
    open_camera,
    show_sender_frame,
)
from src.config import get as config_get
from src.gestures.gesture_rules import detect_gesture
from src.inference.run_realtime import (
    STABLE_FRAME_COUNT,
    load_model,
    predict_model_candidate,
    update_stable_prediction,
)
from src.processing.placement_engine import detect_zone
from src.processing.smart_sentence_builder import SmartSentenceBuilder
from src.testing.gesture_validation import (
    GESTURE_TEST_LOG_PATH,
    GestureValidationSession,
    format_test_prediction_label,
    get_app_gesture_for_test_label,
)


HAND_STATE_TTL_SECONDS = float(config_get("detection.hand_state_ttl_seconds", 1.2))
DEFAULT_CONFIDENCE_THRESHOLD = float(
    config_get("detection.confidence_threshold", 0.75)
)


def clear_stale_prediction_state(
    prediction_buffers,
    stable_predictions,
    hand_last_seen_at,
    current_time,
):
    stale_keys = [
        hand_key
        for hand_key, last_seen_at in hand_last_seen_at.items()
        if (current_time - last_seen_at) >= HAND_STATE_TTL_SECONDS
    ]
    for hand_key in stale_keys:
        prediction_buffers.pop(hand_key, None)
        stable_predictions.pop(hand_key, None)
        hand_last_seen_at.pop(hand_key, None)


def get_hand_label_and_score(result, hand_index):
    if result.multi_handedness is None or hand_index >= len(result.multi_handedness):
        return f"Hand {hand_index + 1}", 1.0

    handedness = result.multi_handedness[hand_index]
    if handedness is None or not handedness.classification:
        return f"Hand {hand_index + 1}", 1.0

    classification = handedness.classification[0]
    return classification.label, classification.score


def predict_validation_gesture_for_hand(
    model,
    landmarks,
    hand_key,
    prediction_buffers,
    stable_predictions,
):
    if hand_key not in prediction_buffers:
        prediction_buffers[hand_key] = deque(maxlen=STABLE_FRAME_COUNT)
        stable_predictions[hand_key] = ("UNKNOWN", 0.0)

    raw_label, confidence = predict_model_candidate(model, landmarks)
    if raw_label is None:
        fallback_gesture = detect_gesture(landmarks)
        if fallback_gesture != "UNKNOWN":
            raw_label = fallback_gesture
            confidence = 1.0
        else:
            raw_label = "UNKNOWN"
            confidence = 0.0

    display_gesture = format_test_prediction_label(raw_label)
    stable_predictions[hand_key] = update_stable_prediction(
        prediction_buffers[hand_key],
        stable_predictions[hand_key],
        display_gesture,
        float(confidence or 0.0),
    )
    return display_gesture, float(confidence or 0.0), stable_predictions[hand_key]


def empty_trace(placement=None):
    return {
        "gesture": None,
        "placement": placement,
        "zone": placement,
        "intent": None,
        "sentence": None,
    }


def build_trace_for_prediction(predicted_gesture, placement, builder):
    app_gesture = get_app_gesture_for_test_label(predicted_gesture)
    if app_gesture is None:
        return empty_trace(placement)

    builder.clear()
    return builder.build_sentence_trace(app_gesture, placement=placement)


def update_validation_ui_state(ui_state, session):
    ui_state["expected_gesture"] = session.current_expected_gesture or EMPTY_VALUE
    ui_state["detection_count"] = session.count_for_gesture()[1]
    ui_state["accuracy_text"] = session.accuracy_text_for_gesture()
    ui_state["gesture_progress"] = (
        f"{session.current_index + 1}/{len(session.expected_gestures)}"
        if session.expected_gestures
        else "0/0"
    )
    ui_state["session_count"] = session.total_attempts


def reset_live_prediction(ui_state):
    ui_state["gesture"] = "READY"
    ui_state["confidence"] = 0.0
    ui_state["intent"] = EMPTY_VALUE
    ui_state["sentence"] = EMPTY_VALUE


def handle_validation_keypress(key, session, ui_state):
    if key in {ord("q"), ord("Q")}:
        return True, False

    gesture_changed = False
    if key in {ord("n"), ord("N")}:
        session.advance(1)
        gesture_changed = True
    elif key in {ord("b"), ord("B")}:
        session.advance(-1)
        gesture_changed = True

    if gesture_changed:
        update_validation_ui_state(ui_state, session)
        reset_live_prediction(ui_state)
        print(f"CURRENT TEST GESTURE: {session.current_expected_gesture}")

    return False, gesture_changed


def print_detection_block(record, intent, sentence):
    print("GESTURE DETECTED:", record["predicted_gesture"])
    print("CONFIDENCE:", f"{record['confidence']:.2f}")
    print("INTENT:", intent or EMPTY_VALUE)
    print("SENTENCE:", sentence or EMPTY_VALUE)
    print("EXPECTED:", record["expected_gesture"])
    print("CORRECT:", "YES" if record["correct"] else "NO")


def print_validation_summary(session):
    print("VALIDATION SUMMARY")
    for summary_line in session.build_accuracy_summary():
        print(summary_line)

    confusion_lines = session.build_confusion_summary()
    if confusion_lines:
        print("CONFUSED GESTURES")
        for confusion_line in confusion_lines:
            print(confusion_line)
    else:
        print("CONFUSED GESTURES")
        print("None recorded.")


def run_gesture_validation_mode(
    camera_index,
    *,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    start_gesture=None,
    log_path=GESTURE_TEST_LOG_PATH,
):
    session = GestureValidationSession(
        log_path=log_path,
        start_gesture=start_gesture,
    )
    ui_state = {
        "test_mode": True,
        "expected_gesture": session.current_expected_gesture or EMPTY_VALUE,
        "gesture": "READY",
        "confidence": 0.0,
        "intent": EMPTY_VALUE,
        "sentence": EMPTY_VALUE,
        "detection_count": 0,
        "accuracy_text": "0/0",
        "gesture_progress": "0/0",
        "session_count": 0,
        "threshold": float(confidence_threshold),
        "speech_enabled": False,
        "pilot_enabled": False,
        "metrics_visible": False,
        "connection_status": "TEST MODE",
        "camera_lost": False,
        "startup_message": "TEST MODE",
    }
    update_validation_ui_state(ui_state, session)

    model = load_model()
    if model is None:
        print("Warning: model not loaded. Validation will be limited to rule-based gestures.")

    capture = open_camera(camera_index)
    if capture is None or not capture.isOpened():
        raise RuntimeError("Unable to open webcam. Check your camera device.")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    prediction_buffers = {}
    stable_predictions = {}
    hand_last_seen_at = {}
    active_detection_signature = None
    trace_builder = SmartSentenceBuilder(
        timeout_seconds=float(config_get("sentence_builder.timeout_seconds", 2.0))
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)
    print(f"CURRENT TEST GESTURE: {session.current_expected_gesture}")

    try:
        with mp_hands.Hands(
            max_num_hands=int(config_get("detection.max_num_hands", 2)),
            min_detection_confidence=float(
                config_get("detection.detection_confidence", 0.7)
            ),
            min_tracking_confidence=float(
                config_get("detection.tracking_confidence", 0.7)
            ),
        ) as hands:
            while True:
                success, frame = capture.read()
                if not success:
                    ui_state["camera_lost"] = True
                    reset_live_prediction(ui_state)
                    show_sender_frame(None, ui_state)
                    should_quit, _ = handle_validation_keypress(
                        cv2.waitKey(1) & 0xFF,
                        session,
                        ui_state,
                    )
                    if should_quit:
                        break
                    time.sleep(0.1)
                    continue

                ui_state["camera_lost"] = False
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)
                frame_started_at = time.perf_counter()

                best_current = None
                best_stable = None

                if result.multi_hand_landmarks:
                    for hand_index, hand_landmarks in enumerate(
                        result.multi_hand_landmarks
                    ):
                        if hand_landmarks is None or len(hand_landmarks.landmark) < 21:
                            continue

                        hand_label, hand_score = get_hand_label_and_score(
                            result,
                            hand_index,
                        )
                        if hand_score < HAND_CONFIDENCE_THRESHOLD:
                            continue

                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                        )
                        hand_last_seen_at[hand_label] = frame_started_at
                        current_gesture, current_confidence, stable_prediction = (
                            predict_validation_gesture_for_hand(
                                model,
                                hand_landmarks.landmark,
                                hand_label,
                                prediction_buffers,
                                stable_predictions,
                            )
                        )
                        hand_zone = detect_zone(hand_landmarks.landmark)
                        stable_gesture, stable_confidence = stable_prediction
                        stable_matches_current = (
                            stable_gesture == current_gesture
                            and stable_gesture not in {"UNKNOWN", "UNSURE"}
                        )
                        current_candidate = {
                            "gesture": current_gesture,
                            "confidence": current_confidence,
                            "zone": hand_zone,
                        }
                        stable_candidate = {
                            "gesture": stable_gesture,
                            "confidence": stable_confidence,
                            "zone": hand_zone,
                            "stable_matches_current": stable_matches_current,
                        }

                        if (
                            best_current is None
                            or current_confidence >= best_current["confidence"]
                        ):
                            best_current = current_candidate
                        if (
                            best_stable is None
                            or stable_confidence >= best_stable["confidence"]
                        ):
                            best_stable = stable_candidate

                clear_stale_prediction_state(
                    prediction_buffers,
                    stable_predictions,
                    hand_last_seen_at,
                    frame_started_at,
                )

                if best_current is None:
                    reset_live_prediction(ui_state)
                    active_detection_signature = None
                else:
                    ui_state["gesture"] = best_current["gesture"]
                    ui_state["confidence"] = best_current["confidence"]
                    trace = build_trace_for_prediction(
                        best_current["gesture"],
                        best_current["zone"],
                        trace_builder,
                    )
                    ui_state["intent"] = trace["intent"] or EMPTY_VALUE
                    ui_state["sentence"] = trace["sentence"] or EMPTY_VALUE

                    stable_is_ready = (
                        best_stable is not None
                        and best_stable["stable_matches_current"]
                        and best_stable["confidence"] >= confidence_threshold
                    )
                    if stable_is_ready:
                        detection_signature = (
                            session.current_expected_gesture,
                            best_stable["gesture"],
                        )
                        if detection_signature != active_detection_signature:
                            record = session.record_prediction(
                                best_stable["gesture"],
                                best_stable["confidence"],
                            )
                            stable_trace = build_trace_for_prediction(
                                best_stable["gesture"],
                                best_stable["zone"],
                                trace_builder,
                            )
                            update_validation_ui_state(ui_state, session)
                            print_detection_block(
                                record,
                                stable_trace["intent"],
                                stable_trace["sentence"],
                            )
                            active_detection_signature = detection_signature
                    else:
                        active_detection_signature = None

                show_sender_frame(frame, ui_state)
                should_quit, gesture_changed = handle_validation_keypress(
                    cv2.waitKey(1) & 0xFF,
                    session,
                    ui_state,
                )
                if gesture_changed:
                    prediction_buffers.clear()
                    stable_predictions.clear()
                    hand_last_seen_at.clear()
                    active_detection_signature = None
                if should_quit:
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print_validation_summary(session)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DualTalk gesture validation mode."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=int(config_get("camera.index", 0)),
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold for logging a detected gesture.",
    )
    parser.add_argument(
        "--start-gesture",
        default=None,
        help="Optional gesture label to start on, for example WAIT or COME_HERE.",
    )
    parser.add_argument(
        "--log",
        default=GESTURE_TEST_LOG_PATH,
        help="CSV path for gesture validation logs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_gesture_validation_mode(
        args.camera,
        confidence_threshold=args.threshold,
        start_gesture=args.start_gesture,
        log_path=args.log,
    )


if __name__ == "__main__":
    main()

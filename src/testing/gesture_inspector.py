import argparse
import os
import sys
import time

import cv2
import numpy as np


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import get as config_get
from src.gestures.gesture_rules import FINGER_DEBUG_ORDER, get_finger_states
from src.utils.hand_tracker import (
    annotate_frame,
    create_hand_detector,
    draw_landmarks_on_frame,
    extract_landmark_coordinates,
)


WINDOW_NAME = "DualTalk - Gesture Inspector"
HAND_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_PRINT_INTERVAL_SECONDS = 0.25
PANEL_WIDTH = 430


def get_hand_label_and_score(result, hand_index):
    if result.multi_handedness is None or hand_index >= len(result.multi_handedness):
        return f"Hand {hand_index + 1}", 1.0

    handedness = result.multi_handedness[hand_index]
    if handedness is None or not handedness.classification:
        return f"Hand {hand_index + 1}", 1.0

    classification = handedness.classification[0]
    return classification.label, classification.score


def select_best_hand(result):
    if not result.multi_hand_landmarks:
        return None

    best_hand = None
    for hand_index, hand_landmarks in enumerate(result.multi_hand_landmarks):
        if hand_landmarks is None or len(hand_landmarks.landmark) < 21:
            continue

        hand_label, hand_score = get_hand_label_and_score(result, hand_index)
        if hand_score < HAND_CONFIDENCE_THRESHOLD:
            continue

        if best_hand is None or hand_score > best_hand["score"]:
            best_hand = {
                "index": hand_index,
                "label": hand_label,
                "score": hand_score,
                "landmarks": hand_landmarks,
            }

    return best_hand


def print_debug_block(hand_label, hand_score, finger_states, landmarks):
    print("=" * 44)
    print(f"Hand: {hand_label} ({hand_score:.2f})")
    for finger_name in FINGER_DEBUG_ORDER:
        print(f"{finger_name}: {finger_states[finger_name]}")
    print("Landmarks:")
    for landmark_index, landmark in enumerate(landmarks):
        print(
            f"  {landmark_index:02d}: "
            f"x={float(landmark.x):.3f} "
            f"y={float(landmark.y):.3f} "
            f"z={float(landmark.z):.3f}"
        )


def draw_panel_line(panel, text, y, *, scale=0.48, color=(230, 235, 245), thickness=1):
    cv2.putText(
        panel,
        text,
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def build_debug_view(frame, hand_info, finger_states):
    panel = np.zeros((frame.shape[0], PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = (16, 20, 28)

    draw_panel_line(
        panel,
        "Gesture Inspector",
        32,
        scale=0.8,
        color=(120, 220, 255),
        thickness=2,
    )
    draw_panel_line(panel, "Q Quit", 58, scale=0.48, color=(180, 190, 210))

    if hand_info is None:
        draw_panel_line(
            panel,
            "No hand detected",
            102,
            scale=0.65,
            color=(0, 210, 255),
            thickness=2,
        )
        draw_panel_line(
            panel,
            "Show one hand to inspect finger states and landmarks.",
            132,
            scale=0.45,
            color=(210, 210, 210),
        )
        return np.hstack((frame, panel))

    draw_panel_line(
        panel,
        f"Tracking: {hand_info['label']} ({hand_info['score']:.2f})",
        96,
        scale=0.58,
        color=(0, 210, 255),
        thickness=2,
    )

    cursor_y = 132
    for finger_name in FINGER_DEBUG_ORDER:
        draw_panel_line(
            panel,
            f"{finger_name}: {finger_states[finger_name]}",
            cursor_y,
            scale=0.54,
            color=(255, 255, 255),
            thickness=2,
        )
        cursor_y += 26

    cursor_y += 12
    draw_panel_line(
        panel,
        "Landmarks (normalized x, y, z)",
        cursor_y,
        scale=0.5,
        color=(180, 190, 210),
    )
    cursor_y += 24

    for landmark_index, landmark in enumerate(hand_info["landmarks"].landmark):
        draw_panel_line(
            panel,
            (
                f"{landmark_index:02d}: "
                f"{float(landmark.x):.3f}, "
                f"{float(landmark.y):.3f}, "
                f"{float(landmark.z):.3f}"
            ),
            cursor_y,
            scale=0.44,
            color=(225, 225, 225),
        )
        cursor_y += 22

    draw_panel_line(
        panel,
        "Point IDs are drawn on the camera view.",
        panel.shape[0] - 24,
        scale=0.42,
        color=(160, 170, 190),
    )
    return np.hstack((frame, panel))


def run_gesture_inspector(camera_index, print_interval_seconds):
    capture = cv2.VideoCapture(camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(config_get("camera.width", 1280)))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config_get("camera.height", 720)))
    capture.set(cv2.CAP_PROP_FPS, int(config_get("camera.fps", 30)))
    if not capture.isOpened():
        raise RuntimeError("Unable to open webcam. Check your camera device.")

    last_print_at = 0.0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        with create_hand_detector(
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
                    print("Could not read a frame from the webcam.")
                    time.sleep(0.1)
                    if cv2.waitKey(1) & 0xFF in {ord("q"), ord("Q")}:
                        break
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)

                hand_info = select_best_hand(result)
                finger_states = None
                if hand_info is not None:
                    draw_landmarks_on_frame(frame, hand_info["landmarks"])
                    landmark_points = extract_landmark_coordinates(
                        hand_info["landmarks"],
                        frame.shape,
                    )
                    annotate_frame(frame, landmark_points)
                    finger_states = get_finger_states(hand_info["landmarks"].landmark)

                    current_time = time.time()
                    if current_time - last_print_at >= max(0.0, print_interval_seconds):
                        print_debug_block(
                            hand_info["label"],
                            hand_info["score"],
                            finger_states,
                            hand_info["landmarks"].landmark,
                        )
                        last_print_at = current_time

                debug_view = build_debug_view(frame, hand_info, finger_states)
                cv2.imshow(WINDOW_NAME, debug_view)
                if cv2.waitKey(1) & 0xFF in {ord("q"), ord("Q")}:
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the DualTalk gesture inspector."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=int(config_get("camera.index", 0)),
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=DEFAULT_PRINT_INTERVAL_SECONDS,
        help="Seconds between console landmark dumps while a hand is visible.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_gesture_inspector(args.camera, args.print_interval)


if __name__ == "__main__":
    main()

from typing import Any, List, Tuple

import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def create_hand_detector(
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.7,
    min_tracking_confidence: float = 0.7,
) -> mp_hands.Hands:
    """Create a MediaPipe hand detector with the desired settings."""
    return mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def extract_landmark_coordinates(
    hand_landmarks: Any,
    image_shape: Tuple[int, int, int],
) -> List[Tuple[int, int]]:
    """Convert normalized MediaPipe landmark coordinates to pixel coordinates."""
    height, width, _ = image_shape
    return [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in hand_landmarks.landmark
    ]


def draw_landmarks_on_frame(frame: Any, hand_landmarks: Any) -> None:
    """Draw the hand landmarks and connections on the frame."""
    mp_draw.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
        mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
    )


def annotate_frame(frame: Any, landmark_points: List[Tuple[int, int]]) -> None:
    """Draw landmark circles and ids on the frame."""
    for idx, (x, y) in enumerate(landmark_points):
        cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)
        cv2.putText(
            frame,
            str(idx),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def log_landmark_coordinates(landmark_points: List[Tuple[int, int]], hand_index: int) -> None:
    """Print the 21 landmark coordinates for one detected hand."""
    print(f"Hand {hand_index} landmarks:")
    for idx, (x, y) in enumerate(landmark_points):
        print(f"  {idx}: ({x}, {y})")


def process_frame(frame: Any, hands: mp_hands.Hands) -> Any:
    """Process a single webcam frame and return the annotated result."""
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks, start=1):
            landmark_points = extract_landmark_coordinates(hand_landmarks, frame.shape)
            log_landmark_coordinates(landmark_points, hand_index)
            draw_landmarks_on_frame(frame, hand_landmarks)
            annotate_frame(frame, landmark_points)

    return frame


def run_hand_tracker() -> None:
    """Run the webcam loop and show detected hand landmarks in real time."""
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Unable to open webcam. Check your camera device.")

    try:
        with create_hand_detector() as hands:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                annotated_frame = process_frame(frame, hands)
                cv2.imshow("Hand Tracker", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_hand_tracker()

import unittest
from dataclasses import dataclass

from src.gestures.gesture_rules import (
    detect_gesture,
    detect_no_motion,
    get_finger_states,
    is_no_handshape,
)


@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0


def build_landmarks():
    return [Landmark(0.5, 0.5, 0.0) for _ in range(21)]


class NoGestureRuleTests(unittest.TestCase):
    def test_no_handshape_requires_index_up_and_other_fingers_folded(self):
        landmarks = build_landmarks()
        landmarks[8] = Landmark(0.50, 0.20)
        landmarks[6] = Landmark(0.50, 0.32)
        landmarks[5] = Landmark(0.50, 0.42)
        landmarks[12] = Landmark(0.55, 0.70)
        landmarks[16] = Landmark(0.60, 0.72)
        landmarks[20] = Landmark(0.65, 0.74)
        landmarks[9] = Landmark(0.55, 0.48)
        landmarks[13] = Landmark(0.60, 0.50)
        landmarks[17] = Landmark(0.65, 0.52)

        self.assertTrue(is_no_handshape(landmarks))

    def test_no_motion_detects_left_right_index_motion(self):
        x_positions = [0.44, 0.49, 0.43, 0.50, 0.42, 0.48, 0.44]
        y_positions = [0.24, 0.25, 0.24, 0.25, 0.24, 0.25, 0.24]

        self.assertTrue(detect_no_motion(x_positions, y_positions))

    def test_no_motion_rejects_small_horizontal_range(self):
        x_positions = [0.44, 0.45, 0.44, 0.45, 0.44, 0.45]
        y_positions = [0.24, 0.24, 0.25, 0.24, 0.25, 0.24]

        self.assertFalse(detect_no_motion(x_positions, y_positions))

    def test_yes_rule_still_works(self):
        landmarks = build_landmarks()
        landmarks[4] = Landmark(0.45, 0.20)
        landmarks[3] = Landmark(0.45, 0.32)
        landmarks[8] = Landmark(0.50, 0.70)
        landmarks[5] = Landmark(0.50, 0.42)
        landmarks[12] = Landmark(0.56, 0.72)
        landmarks[9] = Landmark(0.56, 0.46)

        self.assertEqual(detect_gesture(landmarks), "YES")

    def test_hold_rule_still_works(self):
        landmarks = build_landmarks()
        landmarks[8] = Landmark(0.45, 0.20)
        landmarks[12] = Landmark(0.50, 0.22)
        landmarks[16] = Landmark(0.55, 0.24)
        landmarks[20] = Landmark(0.60, 0.26)
        landmarks[5] = Landmark(0.45, 0.45)
        landmarks[9] = Landmark(0.50, 0.47)
        landmarks[13] = Landmark(0.55, 0.49)
        landmarks[17] = Landmark(0.60, 0.51)

        self.assertEqual(detect_gesture(landmarks), "HOLD")

    def test_finger_states_report_open_for_open_palm(self):
        landmarks = build_landmarks()
        landmarks[0] = Landmark(0.50, 0.72)
        landmarks[2] = Landmark(0.41, 0.60)
        landmarks[3] = Landmark(0.33, 0.54)
        landmarks[4] = Landmark(0.25, 0.48)
        landmarks[5] = Landmark(0.41, 0.52)
        landmarks[6] = Landmark(0.39, 0.38)
        landmarks[8] = Landmark(0.37, 0.14)
        landmarks[9] = Landmark(0.49, 0.50)
        landmarks[10] = Landmark(0.49, 0.34)
        landmarks[12] = Landmark(0.49, 0.08)
        landmarks[13] = Landmark(0.57, 0.52)
        landmarks[14] = Landmark(0.59, 0.38)
        landmarks[16] = Landmark(0.63, 0.14)
        landmarks[17] = Landmark(0.65, 0.55)
        landmarks[18] = Landmark(0.69, 0.44)
        landmarks[20] = Landmark(0.75, 0.23)

        self.assertEqual(
            get_finger_states(landmarks),
            {
                "Thumb": "OPEN",
                "Index": "OPEN",
                "Middle": "OPEN",
                "Ring": "OPEN",
                "Pinky": "OPEN",
            },
        )

    def test_finger_states_report_closed_for_fist(self):
        landmarks = build_landmarks()
        landmarks[0] = Landmark(0.50, 0.70)
        landmarks[2] = Landmark(0.42, 0.60)
        landmarks[3] = Landmark(0.43, 0.56)
        landmarks[4] = Landmark(0.47, 0.58)
        landmarks[5] = Landmark(0.41, 0.52)
        landmarks[6] = Landmark(0.43, 0.46)
        landmarks[8] = Landmark(0.48, 0.56)
        landmarks[9] = Landmark(0.49, 0.50)
        landmarks[10] = Landmark(0.50, 0.44)
        landmarks[12] = Landmark(0.52, 0.56)
        landmarks[13] = Landmark(0.57, 0.52)
        landmarks[14] = Landmark(0.56, 0.46)
        landmarks[16] = Landmark(0.55, 0.58)
        landmarks[17] = Landmark(0.64, 0.55)
        landmarks[18] = Landmark(0.61, 0.50)
        landmarks[20] = Landmark(0.58, 0.61)

        self.assertEqual(
            get_finger_states(landmarks),
            {
                "Thumb": "CLOSED",
                "Index": "CLOSED",
                "Middle": "CLOSED",
                "Ring": "CLOSED",
                "Pinky": "CLOSED",
            },
        )


if __name__ == "__main__":
    unittest.main()

import unittest
from dataclasses import dataclass

from src.processing.intent_builder import IntentBuilder
from src.processing.intent_engine import IntentEngine
from src.processing.placement_engine import (
    CENTER_ZONE,
    CHEST_ZONE,
    HEAD_ZONE,
    LEFT_ZONE,
    RIGHT_ZONE,
    detect_zone,
)


@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0


def build_landmarks(wrist_x: float, wrist_y: float):
    return [Landmark(wrist_x, wrist_y) for _ in range(21)]


class PlacementEngineTests(unittest.TestCase):
    def test_detect_zone_classifies_head(self):
        self.assertEqual(detect_zone(build_landmarks(0.50, 0.20)), HEAD_ZONE)

    def test_detect_zone_classifies_chest(self):
        self.assertEqual(detect_zone(build_landmarks(0.50, 0.45)), CHEST_ZONE)

    def test_detect_zone_classifies_left(self):
        self.assertEqual(detect_zone(build_landmarks(0.20, 0.45)), LEFT_ZONE)

    def test_detect_zone_classifies_right(self):
        self.assertEqual(detect_zone(build_landmarks(0.80, 0.45)), RIGHT_ZONE)

    def test_detect_zone_classifies_center(self):
        self.assertEqual(detect_zone(build_landmarks(0.50, 0.80)), CENTER_ZONE)


class IntentBuilderPlacementTests(unittest.TestCase):
    def test_help_in_chest_zone_maps_to_need_help(self):
        trace = IntentBuilder().build_sentence_trace("HELP", placement=CHEST_ZONE)
        self.assertEqual(trace["zone"], CHEST_ZONE)
        self.assertEqual(trace["intent"], "help")
        self.assertEqual(trace["sentence"], "I need help.")

    def test_help_in_head_zone_maps_to_emergency(self):
        trace = IntentBuilder().build_sentence_trace("HELP", placement=HEAD_ZONE)
        self.assertEqual(trace["intent"], "emergency")
        self.assertEqual(trace["sentence"], "This is an emergency.")

    def test_point_in_left_zone_maps_to_go_left(self):
        trace = IntentBuilder().build_sentence_trace("POINT", placement=LEFT_ZONE)
        self.assertEqual(trace["intent"], "go_left")
        self.assertEqual(trace["sentence"], "Go left.")

    def test_point_in_right_zone_maps_to_go_right(self):
        trace = IntentBuilder().build_sentence_trace("POINT", placement=RIGHT_ZONE)
        self.assertEqual(trace["intent"], "go_right")
        self.assertEqual(trace["sentence"], "Go right.")


class IntentEnginePlacementTests(unittest.TestCase):
    def test_intent_engine_uses_placement_aware_intent(self):
        result = IntentEngine().process_gesture("HELP", placement=HEAD_ZONE)
        self.assertIsNotNone(result)
        self.assertEqual(result.zone, HEAD_ZONE)
        self.assertEqual(result.intent, "emergency")
        self.assertEqual(result.sentence, "This is an emergency.")


class IntentBuilderGestureExpansionTests(unittest.TestCase):
    def test_help_without_special_placement_maps_to_help(self):
        trace = IntentBuilder().build_sentence_trace("HELP", placement=CENTER_ZONE)
        self.assertEqual(trace["intent"], "help")
        self.assertEqual(trace["sentence"], "I need help.")

    def test_thank_you_maps_to_gratitude(self):
        trace = IntentBuilder().build_sentence_trace("THANK YOU")
        self.assertEqual(trace["intent"], "gratitude")
        self.assertEqual(trace["sentence"], "Thank you.")

    def test_stop_maps_to_stop_intent(self):
        trace = IntentBuilder().build_sentence_trace("STOP")
        self.assertEqual(trace["intent"], "stop")
        self.assertEqual(trace["sentence"], "Please stop.")

    def test_come_here_maps_to_request_intent(self):
        trace = IntentBuilder().build_sentence_trace("COME HERE")
        self.assertEqual(trace["intent"], "request")
        self.assertEqual(trace["sentence"], "Come here.")


if __name__ == "__main__":
    unittest.main()

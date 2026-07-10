import time

from src.processing.gesture_profile import get_gesture_profile
from src.processing.placement_engine import (
    resolve_placement_intent,
    resolve_placement_sentence,
    normalize_zone,
)


_PROFILE = get_gesture_profile()
GESTURE_TO_INTENT = _PROFILE.gesture_to_intent
INTENT_TO_SENTENCE = _PROFILE.intent_to_sentence
COMBINED_INTENT_TO_SENTENCE = _PROFILE.intent_sequences_to_sentence
NO_HAND_GESTURES = set(_PROFILE.no_hand_gestures)


class IntentBuilder:
    """
    Builds sentences from stable gesture intent instead of direct gesture-word mapping.
    """

    def __init__(self):
        self.last_gesture = None
        self.last_placement = None
        self.current_intent = None
        self.current_sentence = None

    def build_sentence(self, gesture, placement=None):
        trace = self.build_sentence_trace(gesture, placement=placement)
        return trace["sentence"]

    def build_sentence_trace(self, gesture, placement=None):
        normalized_gesture = self.normalize_gesture(gesture)
        normalized_placement = normalize_zone(placement)
        if normalized_gesture is None:
            self.clear()
            self._debug(gesture, normalized_placement, None, None)
            now = time.perf_counter()
            return {
                "gesture": None,
                "placement": normalized_placement,
                "zone": normalized_placement,
                "intent": None,
                "sentence": None,
                "intent_started_at": now,
                "intent_completed_at": now,
                "sentence_started_at": now,
                "sentence_completed_at": now,
            }

        intent_started_at = time.perf_counter()
        current_intent = self._resolve_intent(
            normalized_gesture,
            normalized_placement,
        )
        intent_completed_at = time.perf_counter()
        previous_intent = self.current_intent
        sentence_started_at = time.perf_counter()
        sentence = self._resolve_sentence(previous_intent, current_intent)
        if sentence is None and current_intent is not None:
            sentence = self._resolve_sentence_for_intent(current_intent)
        sentence_completed_at = time.perf_counter()

        self.last_gesture = normalized_gesture
        self.last_placement = normalized_placement
        self.current_intent = current_intent
        self.current_sentence = sentence
        self._debug(normalized_gesture, normalized_placement, current_intent, sentence)
        return {
            "gesture": normalized_gesture,
            "placement": normalized_placement,
            "zone": normalized_placement,
            "intent": current_intent,
            "sentence": sentence,
            "intent_started_at": intent_started_at,
            "intent_completed_at": intent_completed_at,
            "sentence_started_at": sentence_started_at,
            "sentence_completed_at": sentence_completed_at,
        }

    def clear(self):
        self.last_gesture = None
        self.last_placement = None
        self.current_intent = None
        self.current_sentence = None

    @staticmethod
    def normalize_gesture(gesture):
        if gesture is None:
            return None

        normalized = str(gesture).strip().upper()
        if not normalized or normalized in NO_HAND_GESTURES:
            return None
        return normalized

    @staticmethod
    def _resolve_sentence(previous_intent, current_intent):
        if current_intent is None:
            return None

        combined_sentence = COMBINED_INTENT_TO_SENTENCE.get(
            (previous_intent, current_intent)
        )
        if combined_sentence is not None:
            return combined_sentence

        return INTENT_TO_SENTENCE.get(current_intent)

    @staticmethod
    def _resolve_intent(gesture, placement):
        placement_intent = resolve_placement_intent(gesture, placement)
        if placement_intent is not None:
            return placement_intent
        return GESTURE_TO_INTENT.get(gesture)

    @staticmethod
    def _resolve_sentence_for_intent(intent):
        sentence = INTENT_TO_SENTENCE.get(intent)
        if sentence is not None:
            return sentence
        return resolve_placement_sentence(intent)

    @staticmethod
    def _debug(gesture, placement, intent, sentence):
        print("GESTURE DETECTED:", gesture)
        print("ZONE:", placement)
        print("INTENT:", intent)
        print("FINAL SENTENCE:", sentence)


_global_builder = IntentBuilder()


def build_sentence(gesture, placement=None):
    return _global_builder.build_sentence(gesture, placement=placement)

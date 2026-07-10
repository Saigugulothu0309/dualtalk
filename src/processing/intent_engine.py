import os
from dataclasses import dataclass

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "Missing dependency: PyYAML. Install using 'pip install pyyaml'"
    ) from exc

from src.processing.placement_engine import (
    normalize_zone,
    resolve_placement_intent,
    resolve_placement_sentence,
)


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GESTURE_CONFIG_DIR = os.path.join(BASE_DIR, "config", "gestures")
DEFAULT_PROFILE_LANGUAGE = (
    str(os.getenv("DUALTALK_GESTURE_PROFILE", "asl")).strip().lower() or "asl"
)
NO_HAND_GESTURES = {"", "NO HAND", "NO_HAND", "NOHAND"}
UNSTABLE_GESTURES = {"UNKNOWN", "UNSURE"}
INTENT_TO_SENTENCE = {
    "greeting": "Hello.",
    "pause": "Please wait.",
    "affirmative": "Yes.",
    "negative": "No.",
    "help": "I need help.",
    "gratitude": "Thank you.",
    "stop": "Please stop.",
    "request": "Come here.",
    "need_help": "I need help.",
    "need_water": "I need water.",
    "need_food": "I need food.",
    "need_bathroom": "I need the bathroom.",
    "emergency": "This is an emergency.",
    "go_left": "Go left.",
    "go_right": "Go right.",
}


def _normalize_gesture_map(raw_mapping):
    if not isinstance(raw_mapping, dict):
        return {}

    normalized = {}
    for gesture, intent in raw_mapping.items():
        if gesture is None or intent is None:
            continue

        gesture_key = str(gesture).strip().upper()
        intent_value = str(intent).strip().lower()
        if gesture_key and intent_value:
            normalized[gesture_key] = intent_value
    return normalized


def _load_profile_file(language):
    profile_path = os.path.join(GESTURE_CONFIG_DIR, f"{language}.yaml")
    if not os.path.exists(profile_path):
        return None

    with open(profile_path, "r", encoding="utf-8") as file_obj:
        loaded = yaml.load(file_obj, Loader=yaml.BaseLoader) or {}

    if not isinstance(loaded, dict):
        return None
    return loaded


def _load_gesture_profile():
    selected_language = DEFAULT_PROFILE_LANGUAGE
    loaded = _load_profile_file(selected_language)

    if loaded is None and selected_language != "asl":
        selected_language = "asl"
        loaded = _load_profile_file(selected_language)

    if loaded is None:
        print("LOADED GESTURE PROFILE:", selected_language)
        return selected_language, {}

    profile_language = str(
        loaded.get("language") or loaded.get("name") or selected_language
    ).strip().lower() or selected_language
    raw_mapping = loaded.get("gestures")
    if not isinstance(raw_mapping, dict):
        raw_mapping = loaded.get("gesture_to_intent", {})

    print("LOADED GESTURE PROFILE:", profile_language)
    return profile_language, _normalize_gesture_map(raw_mapping)


PROFILE_LANGUAGE, GESTURE_TO_INTENT = _load_gesture_profile()


@dataclass(frozen=True)
class IntentResult:
    gesture: str
    zone: str | None
    intent: str
    sentence: str | None


class IntentEngine:
    def __init__(self):
        self.clear()

    def process_gesture(self, gesture, placement=None, is_stable=True):
        normalized_gesture = self.normalize_gesture(gesture)
        normalized_zone = normalize_zone(placement)
        if normalized_gesture is None:
            self.clear()
            self._debug(None, normalized_zone, None)
            return None

        if not is_stable or normalized_gesture in UNSTABLE_GESTURES:
            self.clear()
            self._debug(normalized_gesture, normalized_zone, None)
            return None

        self.last_gesture = normalized_gesture
        self.last_placement = normalized_zone

        if (
            (normalized_gesture, normalized_zone) == self.active_input_signature
            and self.active_result is not None
        ):
            self.current_intent = self.active_result.intent
            self.current_sentence = self.active_result.sentence
            self._debug(
                self.active_result.gesture,
                self.active_result.zone,
                self.active_result.intent,
            )
            return self.active_result

        intent = self._resolve_intent(normalized_gesture, normalized_zone)
        self.active_input_signature = (normalized_gesture, normalized_zone)

        if intent is None:
            self.active_result = None
            self.current_intent = None
            self.current_sentence = None
            self._debug(normalized_gesture, normalized_zone, None)
            return None

        result = self._build_result(normalized_gesture, normalized_zone, intent)
        self.active_result = result
        self.current_intent = result.intent
        self.current_sentence = result.sentence
        self._debug(result.gesture, result.zone, result.intent)
        return result

    def build_sentence(self, gesture, placement=None, is_stable=True):
        result = self.process_gesture(
            gesture,
            placement=placement,
            is_stable=is_stable,
        )
        if result is None:
            return None
        return result.sentence

    def resolve_intent(self, gesture, placement=None, is_stable=True):
        result = self.process_gesture(
            gesture,
            placement=placement,
            is_stable=is_stable,
        )
        if result is None:
            return None
        return result.intent

    def clear(self):
        self.last_gesture = None
        self.last_placement = None
        self.current_intent = None
        self.current_sentence = None
        self.active_input_signature = None
        self.active_result = None

    @staticmethod
    def normalize_gesture(gesture):
        if gesture is None:
            return None

        normalized = str(gesture).strip().upper()
        if not normalized or normalized in NO_HAND_GESTURES:
            return None
        return normalized

    @staticmethod
    def _resolve_intent(gesture, placement):
        placement_intent = resolve_placement_intent(gesture, placement)
        if placement_intent is not None:
            return placement_intent
        return GESTURE_TO_INTENT.get(gesture)

    @staticmethod
    def _build_result(gesture, zone, intent):
        return IntentResult(
            gesture=gesture,
            zone=zone,
            intent=intent,
            sentence=INTENT_TO_SENTENCE.get(intent) or resolve_placement_sentence(intent),
        )

    @staticmethod
    def _debug(gesture, zone, intent):
        print("GESTURE DETECTED:", gesture)
        print("ZONE:", zone)
        print("INTENT:", intent)


_global_engine = IntentEngine()


def process_gesture(gesture, placement=None, is_stable=True):
    return _global_engine.process_gesture(
        gesture,
        placement=placement,
        is_stable=is_stable,
    )


def resolve_intent(gesture, placement=None, is_stable=True):
    return _global_engine.resolve_intent(
        gesture,
        placement=placement,
        is_stable=is_stable,
    )


def build_sentence(gesture, placement=None, is_stable=True):
    return _global_engine.build_sentence(
        gesture,
        placement=placement,
        is_stable=is_stable,
    )


def clear():
    _global_engine.clear()

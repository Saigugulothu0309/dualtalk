import os
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "Missing dependency: PyYAML. Install using 'pip install pyyaml'"
    ) from exc

from src.config import get as config_get


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_PROFILE_NAME = "hospital"
DEFAULT_PROFILES_DIR = os.path.join("config", "gestures")
DEFAULT_NO_HAND_GESTURES = frozenset({"", "NO HAND", "NO_HAND", "NOHAND"})
DEFAULT_UNSTABLE_GESTURES = frozenset({"UNKNOWN", "UNSURE"})
DEFAULT_PROFILE_DATA = {
    "name": DEFAULT_PROFILE_NAME,
    "description": "Default DualTalk hospital quick-communication profile.",
    "gesture_to_intent": {
        "HOLD": "pause",
        "WAVE": "greeting",
        "YES": "affirmative",
        "NO": "negative",
        "HELP": "help",
        "ME": "need_help",
        "WAIT": "pause",
        "THANKS": "gratitude",
        "THANK YOU": "gratitude",
        "STOP": "stop",
        "COME HERE": "request",
        "DRINK": "need_water",
        "EAT": "need_food",
        "BATHROOM": "need_bathroom",
    },
    "intent_to_sentence": {
        "pause": "Please wait.",
        "greeting": "Hello.",
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
    },
    "gesture_sequences_to_intent": {
        "HELP": {
            "DRINK": "need_water",
            "EAT": "need_food",
            "BATHROOM": "need_bathroom",
        }
    },
    "intent_sequences_to_sentence": {
        "help": {
            "need_water": "I need some water.",
            "need_food": "I need something to eat.",
            "need_bathroom": "I need the bathroom.",
            "affirmative": "I understand.",
            "negative": "I cannot.",
        },
        "need_help": {
            "need_water": "I need some water.",
            "need_food": "I need something to eat.",
            "need_bathroom": "I need to use the bathroom.",
            "affirmative": "I understand.",
            "negative": "I cannot.",
        }
    },
    "vocabulary": {
        "help": "help",
        "need_water": "water",
        "need_food": "food",
        "need_bathroom": "bathroom",
        "affirmative": "yes",
        "negative": "no",
        "gratitude": "thank you",
    },
    "sentence_templates": {
        "need_water": "I need some water.",
        "need_food": "I need something to eat.",
        "need_bathroom": "I need to use the bathroom.",
        "affirmative gratitude": "Yes, thank you.",
        "pause": "Please wait for a moment.",
    },
    "polite_responses": {
        "gratitude": "Thank you.",
        "affirmative": "Yes, thank you.",
        "negative": "No, thank you.",
    },
    "emergency_phrases": {
        "emergency": "This is an emergency.",
        "help": "I need help.",
    },
}
PROFILE_SECTION_KEYS = {
    "gesture_to_intent",
    "intent_to_sentence",
    "gesture_sequences_to_intent",
    "intent_sequences_to_sentence",
    "vocabulary",
    "sentence_templates",
    "polite_responses",
    "emergency_phrases",
    "no_hand_gestures",
    "unstable_gestures",
}


@dataclass(frozen=True)
class GestureProfile:
    name: str
    description: str
    gesture_to_intent: dict[str, str]
    intent_to_sentence: dict[str, str]
    gesture_sequences_to_intent: dict[tuple[str, str], str]
    intent_sequences_to_sentence: dict[tuple[str, str], str]
    vocabulary: dict[str, str]
    sentence_templates: dict[str, str]
    polite_responses: dict[str, str]
    emergency_phrases: dict[str, str]
    no_hand_gestures: frozenset[str]
    unstable_gestures: frozenset[str]


def _merge_dicts(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_gesture_mapping(raw_mapping) -> dict[str, str]:
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


def _normalize_intent_mapping(raw_mapping) -> dict[str, str]:
    if not isinstance(raw_mapping, dict):
        return {}

    normalized = {}
    for intent, sentence in raw_mapping.items():
        if intent is None or sentence is None:
            continue
        intent_key = str(intent).strip().lower()
        sentence_value = str(sentence).strip()
        if intent_key and sentence_value:
            normalized[intent_key] = sentence_value
    return normalized


def _normalize_sequence_map(
    raw_mapping,
    *,
    outer_case: str,
    inner_case: str,
    value_case: str,
) -> dict[tuple[str, str], str]:
    if not isinstance(raw_mapping, dict):
        return {}

    flattened = {}
    for outer_key, nested_mapping in raw_mapping.items():
        if outer_key is None or not isinstance(nested_mapping, dict):
            continue

        normalized_outer = str(outer_key).strip()
        normalized_outer = (
            normalized_outer.upper()
            if outer_case == "upper"
            else normalized_outer.lower()
        )
        if not normalized_outer:
            continue

        for inner_key, value in nested_mapping.items():
            if inner_key is None or value is None:
                continue

            normalized_inner = str(inner_key).strip()
            normalized_inner = (
                normalized_inner.upper()
                if inner_case == "upper"
                else normalized_inner.lower()
            )
            normalized_value = str(value).strip()
            if value_case == "upper":
                normalized_value = normalized_value.upper()
            elif value_case == "lower":
                normalized_value = normalized_value.lower()

            if normalized_inner and normalized_value:
                flattened[(normalized_outer, normalized_inner)] = normalized_value

    return flattened


def _normalize_set(raw_values, *, case: str) -> frozenset[str]:
    if not isinstance(raw_values, (list, tuple, set, frozenset)):
        return frozenset()

    normalized = set()
    for value in raw_values:
        if value is None:
            continue
        item = str(value).strip()
        item = item.upper() if case == "upper" else item.lower()
        normalized.add(item)
    return frozenset(normalized)


def _resolve_profiles_dir(profiles_dir: str | None) -> str:
    raw_path = profiles_dir or str(
        config_get("gestures.profiles_dir", DEFAULT_PROFILES_DIR)
    )
    return raw_path if os.path.isabs(raw_path) else os.path.join(BASE_DIR, raw_path)


def _load_yaml_profile(profile_path: str) -> dict:
    try:
        with open(profile_path, "r", encoding="utf-8") as file_obj:
            loaded = yaml.safe_load(file_obj) or {}
    except OSError as exc:
        print(f"Warning: could not read gesture profile '{profile_path}': {exc}")
        return {}
    except yaml.YAMLError as exc:
        print(f"Warning: failed to parse gesture profile '{profile_path}': {exc}")
        return {}

    if not isinstance(loaded, dict):
        print(f"Warning: gesture profile '{profile_path}' must contain a mapping.")
        return {}
    return loaded


def _build_profile(profile_name: str | None, profiles_dir: str | None) -> GestureProfile:
    selected_name = str(
        profile_name or config_get("gestures.profile", DEFAULT_PROFILE_NAME)
    ).strip() or DEFAULT_PROFILE_NAME
    resolved_dir = _resolve_profiles_dir(profiles_dir)
    selected_path = os.path.join(resolved_dir, f"{selected_name}.yaml")

    loaded_profile = {}
    if os.path.exists(selected_path):
        loaded_profile = _load_yaml_profile(selected_path)
    else:
        fallback_path = os.path.join(resolved_dir, f"{DEFAULT_PROFILE_NAME}.yaml")
        if selected_name != DEFAULT_PROFILE_NAME and os.path.exists(fallback_path):
            print(
                f"Warning: gesture profile '{selected_name}' not found at "
                f"'{selected_path}'. Falling back to '{DEFAULT_PROFILE_NAME}'."
            )
            loaded_profile = _load_yaml_profile(fallback_path)
        else:
            print(
                f"Warning: gesture profile '{selected_name}' not found at "
                f"'{selected_path}'. Using built-in defaults."
            )

    merged_profile = _merge_dicts(DEFAULT_PROFILE_DATA, loaded_profile)
    for section_key in PROFILE_SECTION_KEYS:
        if section_key in loaded_profile:
            merged_profile[section_key] = deepcopy(loaded_profile[section_key])

    return GestureProfile(
        name=str(merged_profile.get("name", selected_name)).strip() or selected_name,
        description=str(merged_profile.get("description", "")).strip(),
        gesture_to_intent=_normalize_gesture_mapping(
            merged_profile.get("gesture_to_intent")
        ),
        intent_to_sentence=_normalize_intent_mapping(
            merged_profile.get("intent_to_sentence")
        ),
        gesture_sequences_to_intent=_normalize_sequence_map(
            merged_profile.get("gesture_sequences_to_intent"),
            outer_case="upper",
            inner_case="upper",
            value_case="lower",
        ),
        intent_sequences_to_sentence=_normalize_sequence_map(
            merged_profile.get("intent_sequences_to_sentence"),
            outer_case="lower",
            inner_case="lower",
            value_case="keep",
        ),
        vocabulary=_normalize_mapping(
            merged_profile.get("vocabulary"),
            key_case="lower",
        ),
        sentence_templates=_normalize_mapping(
            merged_profile.get("sentence_templates"),
            key_case="lower",
        ),
        polite_responses=_normalize_mapping(
            merged_profile.get("polite_responses"),
            key_case="lower",
        ),
        emergency_phrases=_normalize_mapping(
            merged_profile.get("emergency_phrases"),
            key_case="lower",
        ),
        no_hand_gestures=_normalize_set(
            merged_profile.get("no_hand_gestures", DEFAULT_NO_HAND_GESTURES),
            case="upper",
        )
        or DEFAULT_NO_HAND_GESTURES,
        unstable_gestures=_normalize_set(
            merged_profile.get("unstable_gestures", DEFAULT_UNSTABLE_GESTURES),
            case="upper",
        )
        or DEFAULT_UNSTABLE_GESTURES,
    )


def _normalize_mapping(raw_mapping, *, key_case: str = "lower") -> dict[str, str]:
    if not isinstance(raw_mapping, dict):
        return {}

    normalized = {}
    for key, value in raw_mapping.items():
        if key is None or value is None:
            continue
        normalized_key = str(key).strip()
        normalized_key = (
            normalized_key.upper() if key_case == "upper" else normalized_key.lower()
        )
        normalized_value = str(value).strip()
        if normalized_key and normalized_value:
            normalized[normalized_key] = normalized_value
    return normalized


@lru_cache(maxsize=16)
def get_gesture_profile(
    profile_name: str | None = None,
    profiles_dir: str | None = None,
) -> GestureProfile:
    return _build_profile(profile_name, profiles_dir)

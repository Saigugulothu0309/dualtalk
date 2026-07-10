from __future__ import annotations

from dataclasses import dataclass


WRIST_LANDMARK_INDEX = 0

HEAD_ZONE = "HEAD"
CHEST_ZONE = "CHEST"
LEFT_ZONE = "LEFT"
RIGHT_ZONE = "RIGHT"
CENTER_ZONE = "CENTER"

LEFT_ZONE_MAX_X = 0.35
RIGHT_ZONE_MIN_X = 0.65
HEAD_ZONE_MAX_Y = 0.28
CHEST_ZONE_MAX_Y = 0.62

PLACEMENT_INTENT_MAP = {
    ("HELP", CHEST_ZONE): "help",
    ("HELP", HEAD_ZONE): "emergency",
    ("POINT", LEFT_ZONE): "go_left",
    ("POINT", RIGHT_ZONE): "go_right",
}

PLACEMENT_INTENT_TO_SENTENCE = {
    "emergency": "This is an emergency.",
    "go_left": "Go left.",
    "go_right": "Go right.",
}


@dataclass(frozen=True)
class PlacementResult:
    zone: str
    wrist_x: float
    wrist_y: float


def detect_placement(landmarks) -> PlacementResult | None:
    if landmarks is None:
        return None

    try:
        wrist = landmarks[WRIST_LANDMARK_INDEX]
    except (IndexError, KeyError, TypeError):
        return None

    wrist_x = _coerce_coordinate(getattr(wrist, "x", None))
    wrist_y = _coerce_coordinate(getattr(wrist, "y", None))
    if wrist_x is None or wrist_y is None:
        return None

    if wrist_x <= LEFT_ZONE_MAX_X:
        zone = LEFT_ZONE
    elif wrist_x >= RIGHT_ZONE_MIN_X:
        zone = RIGHT_ZONE
    elif wrist_y <= HEAD_ZONE_MAX_Y:
        zone = HEAD_ZONE
    elif wrist_y <= CHEST_ZONE_MAX_Y:
        zone = CHEST_ZONE
    else:
        zone = CENTER_ZONE

    return PlacementResult(zone=zone, wrist_x=wrist_x, wrist_y=wrist_y)


def detect_zone(landmarks) -> str | None:
    placement = detect_placement(landmarks)
    if placement is None:
        return None
    return placement.zone


def normalize_zone(zone) -> str | None:
    if zone is None:
        return None

    normalized = str(zone).strip().upper()
    if not normalized:
        return None
    if normalized.endswith("_ZONE"):
        normalized = normalized[:-5]

    if normalized in {
        HEAD_ZONE,
        CHEST_ZONE,
        LEFT_ZONE,
        RIGHT_ZONE,
        CENTER_ZONE,
    }:
        return normalized
    return None


def resolve_placement_intent(gesture, zone) -> str | None:
    normalized_gesture = normalize_gesture(gesture)
    normalized_zone = normalize_zone(zone)
    if normalized_gesture is None or normalized_zone is None:
        return None
    return PLACEMENT_INTENT_MAP.get((normalized_gesture, normalized_zone))


def resolve_placement_sentence(intent) -> str | None:
    normalized_intent = normalize_intent(intent)
    if normalized_intent is None:
        return None
    return PLACEMENT_INTENT_TO_SENTENCE.get(normalized_intent)


def normalize_gesture(gesture) -> str | None:
    if gesture is None:
        return None

    normalized = str(gesture).strip().upper()
    if not normalized:
        return None
    return normalized


def normalize_intent(intent) -> str | None:
    if intent is None:
        return None

    normalized = str(intent).strip().lower()
    if not normalized:
        return None
    return normalized


def _coerce_coordinate(value) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None

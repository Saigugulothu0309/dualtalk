# ---------------------------------------------------------------------------
# LABEL_REPLACEMENTS – applied at training time via canonicalize_label().
# These normalize noisy / legacy CSV label strings before the model is fit.
# Keys are lower-cased CSV values; values are the canonical training label.
# ---------------------------------------------------------------------------
LABEL_REPLACEMENTS = {
    # Legacy / alternate spellings kept for backward-compatibility
    "hello": "hold",
    "stop": "hold",
    "no/stop": "hold",
    "two": "peace",
    "2": "peace",
    # Normalise the bathroom spelling variants found in the CSV
    "bathroom": "bathroom/restrom/toilet",
    "restroom": "bathroom/restrom/toilet",
    "toilet": "bathroom/restrom/toilet",
    # Both "call" and "come here" map to the same training class
    "call": "come here",
    # Normalise thanks → thank you so the model trains one class
    "thanks": "thank you",
    # ok and super are the same gesture
    "super": "ok/super",
    "ok": "ok/super",
}

# ---------------------------------------------------------------------------
# PREDICTION_LABEL_TO_GESTURE – applied at inference time via
# canonicalize_prediction_label().  Maps the model's raw class string to the
# display label shown in the UI.  Every class the model can emit must have
# an entry here, otherwise the gesture will be silently dropped.
# ---------------------------------------------------------------------------
PREDICTION_LABEL_TO_GESTURE = {
    "bathroom/restrom/toilet": "BATHROOM",
    "come here": "COME HERE",
    "danger": "DANGER",
    "drink": "DRINK",
    "eat": "EAT",
    "help": "HELP",
    "hold": "HOLD",
    "ok/super": "OK",
    "thank you": "THANK YOU",
    "wait": "WAIT",
    "yes": "YES",
    # Legacy aliases kept so an older model still works
    "call": "COME HERE",
    "thanks": "THANK YOU",
}


def canonicalize_label(label: str) -> str:
    """Normalise a raw CSV label for training (lowercased, de-aliased)."""
    normalized_label = str(label).strip().lower()
    return LABEL_REPLACEMENTS.get(normalized_label, normalized_label)


def canonicalize_prediction_label(label) -> str | None:
    """Map a model prediction class to a human-readable gesture string.

    Returns *None* when the class is unknown so the caller can fall back to
    rule-based detection gracefully.
    """
    if label is None:
        return None
    normalized_label = str(label).strip().lower()
    return PREDICTION_LABEL_TO_GESTURE.get(normalized_label)

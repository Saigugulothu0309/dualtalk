import time

from src.processing.intent_builder import (
    COMBINED_INTENT_TO_SENTENCE,
    GESTURE_TO_INTENT,
    INTENT_TO_SENTENCE,
    NO_HAND_GESTURES,
)


CONTEXT_TIMEOUT_SECONDS = 2.0
MAX_INTENT_HISTORY = 2
UNSTABLE_GESTURES = {"UNKNOWN", "UNSURE"}
CONTEXT_SENTENCE_MAP = COMBINED_INTENT_TO_SENTENCE
KNOWN_INTENTS = set(INTENT_TO_SENTENCE) | {
    intent
    for intent_pair in CONTEXT_SENTENCE_MAP
    for intent in intent_pair
}


class ContextEngine:
    """
    Builds a context-aware sentence from the latest stable intents.

    The engine keeps a rolling two-intent history so combinations like
    need_help -> need_water can resolve to a more natural sentence.
    """

    def __init__(self, timeout_seconds=CONTEXT_TIMEOUT_SECONDS):
        self.timeout_seconds = float(timeout_seconds)
        self.clear()

    def process_gesture(self, gesture, is_stable=True, now=None):
        current_time = self._resolve_time(now)
        normalized_gesture = self.normalize_gesture(gesture)
        if normalized_gesture is None:
            return self._handle_no_hand(current_time)

        self.no_hand_since = None
        if not is_stable or normalized_gesture in UNSTABLE_GESTURES:
            return None

        intent = GESTURE_TO_INTENT.get(normalized_gesture)
        if intent is None:
            return None
        return self._store_intent(intent, current_time)

    def process_intent(self, intent, is_stable=True, now=None):
        current_time = self._resolve_time(now)
        normalized_intent = self.normalize_intent(intent)
        if normalized_intent is None:
            return self._handle_no_hand(current_time)

        self.no_hand_since = None
        if not is_stable:
            return None
        return self._store_intent(normalized_intent, current_time)

    def check_timeout(self, now=None):
        return self._handle_no_hand(self._resolve_time(now))

    def clear(self):
        self.intent_history = []
        self.current_sentence = None
        self.last_intent = None
        self.last_update_time = 0.0
        self.no_hand_since = None

    def get_history(self):
        return list(self.intent_history)

    @staticmethod
    def normalize_gesture(gesture):
        if gesture is None:
            return None

        normalized = str(gesture).strip().upper()
        if not normalized or normalized in NO_HAND_GESTURES:
            return None
        return normalized

    @staticmethod
    def normalize_intent(intent):
        if intent is None:
            return None

        normalized = str(intent).strip().lower()
        if not normalized:
            return None
        if normalized not in KNOWN_INTENTS:
            return None
        return normalized

    @staticmethod
    def _resolve_time(now):
        if now is None:
            return time.time()
        return float(now)

    def _handle_no_hand(self, current_time):
        if self.no_hand_since is None:
            self.no_hand_since = current_time
            return None

        if current_time - self.no_hand_since < self.timeout_seconds:
            return None

        if self.intent_history or self.current_sentence is not None:
            self.clear()
            self.no_hand_since = current_time
            self._debug(self.intent_history, self.current_sentence)
        return None

    def _store_intent(self, intent, current_time):
        if intent != self.last_intent:
            self.intent_history.append(intent)
            if len(self.intent_history) > MAX_INTENT_HISTORY:
                self.intent_history = self.intent_history[-MAX_INTENT_HISTORY:]
            self.last_intent = intent

        self.last_update_time = current_time
        self.current_sentence = self._resolve_context_sentence()
        self._debug(self.intent_history, self.current_sentence)
        return self.current_sentence

    def _resolve_context_sentence(self):
        if len(self.intent_history) >= 2:
            history_pair = tuple(self.intent_history[-2:])
            combined_sentence = CONTEXT_SENTENCE_MAP.get(history_pair)
            if combined_sentence is not None:
                return combined_sentence

        if not self.intent_history:
            return None
        return INTENT_TO_SENTENCE.get(self.intent_history[-1])

    @staticmethod
    def _debug(history, sentence):
        print("INTENT HISTORY:", history)
        print("CONTEXT SENTENCE:", sentence)


_global_engine = ContextEngine()


def process_gesture(gesture, is_stable=True, now=None):
    return _global_engine.process_gesture(gesture, is_stable=is_stable, now=now)


def process_intent(intent, is_stable=True, now=None):
    return _global_engine.process_intent(intent, is_stable=is_stable, now=now)


def check_timeout(now=None):
    return _global_engine.check_timeout(now=now)


def clear():
    _global_engine.clear()


def get_history():
    return _global_engine.get_history()

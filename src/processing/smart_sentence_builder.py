from src.processing.intent_builder import IntentBuilder


class SmartSentenceBuilder(IntentBuilder):
    """
    Backwards-compatible wrapper around the intent-based sentence builder.
    """

    def __init__(self, timeout_seconds=2.0):
        super().__init__()
        self.timeout_seconds = timeout_seconds

    def check_timeout(self):
        self.clear()
        print("GESTURE DETECTED:", None)
        print("ZONE:", None)
        print("INTENT:", None)
        print("FINAL SENTENCE:", None)
        return None


# Global instance for ease of use
_global_builder = SmartSentenceBuilder()


def build_sentence(gesture, placement=None):
    """
    Module-level convenience function using the intent-based global builder state.
    """
    return _global_builder.build_sentence(gesture, placement=placement)

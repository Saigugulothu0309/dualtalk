from src.processing.intent_builder import IntentBuilder


class SentenceBuilder(IntentBuilder):
    def __init__(self, max_buffer_size=3):
        super().__init__()
        self.max_buffer_size = max_buffer_size
        self.gesture_buffer = []

    def add_gesture(self, gesture, placement=None):
        normalized_gesture = self.normalize_gesture(gesture)
        self.gesture_buffer = [normalized_gesture] if normalized_gesture else []
        return super().build_sentence(gesture, placement=placement)

    def build_sentence(self, gesture=None, placement=None):
        return self.add_gesture(gesture, placement=placement)

    def flush(self):
        sentence = self.current_sentence
        self.clear()
        return sentence

    def clear(self):
        self.gesture_buffer.clear()
        super().clear()

    @staticmethod
    def normalize_gesture(gesture):
        return IntentBuilder.normalize_gesture(gesture)


def build_sentence_from_gestures(gestures):
    builder = SentenceBuilder()
    latest_sentence = None
    for gesture in gestures:
        sentence = builder.add_gesture(gesture)
        if sentence is not None:
            latest_sentence = sentence
    return latest_sentence

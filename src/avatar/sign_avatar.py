WORD_TO_SIGN = {
    "hello": ["Wave hand side to side"],
    "yes": ["Thumb up"],
    "no": ["Shake fist left and right"],
    "water": ["W shape -> tap chin twice"],
    "help": ["Flat hand -> lift with other fist"],
    "please": ["Flat hand, circular motion on chest"],
    "wait": ["Both hands open, wave fingers"],
    "thank": ["Flat hand from chin, move forward"],
    "bathroom": ["T hand -> shake twice"],
    "need": ["Hook index finger, bend down"],
    "i": ["Point to chest"],
    "want": ["Clawed hands, pull toward body"],
}


class SignAvatar:
    """
    Converts a received text sentence into a sequence of sign descriptions.
    Each description is displayed on the sign user's screen panel.
    """

    def __init__(self, display_seconds_per_sign: float = 1.5):
        self.display_seconds = display_seconds_per_sign
        self._sequence: list[str] = []
        self._index = 0
        self._timer = 0.0
        self._active = False

    def set_sentence(self, sentence: str):
        words = sentence.lower().split()
        self._sequence = []
        for word in words:
            clean = word.strip(".,!?")
            signs = WORD_TO_SIGN.get(clean)
            if signs:
                self._sequence.extend(signs)
            elif clean:
                self._sequence.append(f"[Fingerspell: {clean.upper()}]")
        self._index = 0
        self._timer = 0.0
        self._active = bool(self._sequence)

    def update(self, delta_seconds: float) -> str | None:
        if not self._active or not self._sequence:
            return None

        self._timer += delta_seconds
        if self._timer >= self.display_seconds:
            self._timer = 0.0
            self._index += 1
            if self._index >= len(self._sequence):
                self._active = False
                return None

        return self._sequence[self._index]

    def is_active(self) -> bool:
        return self._active

    def progress(self) -> float:
        if not self._sequence:
            return 0.0
        return min(1.0, self._index / max(1, len(self._sequence) - 1))

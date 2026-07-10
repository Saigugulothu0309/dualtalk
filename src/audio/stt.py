import threading

import speech_recognition as sr


class SpeechToText:
    """
    Continuous speech-to-text listener.
    Calls callback(text) for every recognized utterance.
    Runs in a daemon thread and never blocks the caller.
    """

    def __init__(
        self,
        language: str = "en-US",
        timeout: float = 5.0,
        phrase_time_limit: float = 10.0,
    ):
        self.recognizer = sr.Recognizer()
        self.language = language
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self._running = False
        self._thread: threading.Thread | None = None
        self.request_error_detected = False
        self.last_error: Exception | None = None

    def listen_once(self) -> str | None:
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.timeout,
                        phrase_time_limit=self.phrase_time_limit,
                    )
                    return self.recognizer.recognize_google(
                        audio,
                        language=self.language,
                    )
                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    return None
                except sr.RequestError as exc:
                    self.request_error_detected = True
                    self.last_error = exc
                    return None
        except OSError as exc:
            self.last_error = exc
            return None

    def listen_continuous(self, callback):
        self._running = True

        def _loop():
            while self._running:
                text = self.listen_once()
                if self.request_error_detected or self.last_error is not None:
                    self._running = False
                    break
                if text and text.strip():
                    callback(text.strip())

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

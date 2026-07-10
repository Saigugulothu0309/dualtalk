import time
import unittest

from src.audio.tts import TTS


class RecordingEngine:
    def __init__(self, events, *, fail_on_run=False):
        self.events = events
        self.fail_on_run = fail_on_run
        self.properties = {}

    def setProperty(self, key, value):
        self.properties[key] = value

    def say(self, text):
        self.events.append(("say", text))

    def runAndWait(self):
        self.events.append(("run", None))
        if self.fail_on_run:
            self.fail_on_run = False
            raise RuntimeError("engine failed")

    def stop(self):
        self.events.append(("stop", None))


class TTSWorkerTests(unittest.TestCase):
    def test_tts_speaks_multiple_sentences(self):
        events = []
        tts = TTS(
            enabled=True,
            engine_factory=lambda: RecordingEngine(events),
        )
        self.addCleanup(tts.shutdown)

        self.assertTrue(tts.speak("Hello."))
        self.assertTrue(tts.speak("Yes."))

        self._wait_until(lambda: self._count(events, "say") >= 2)

        spoken = [value for action, value in events if action == "say"]
        self.assertEqual(spoken[:2], ["Hello.", "Yes."])
        self.assertFalse(tts.busy)

    def test_tts_resets_busy_after_engine_failure(self):
        events = []
        factory_calls = {"count": 0}

        def factory():
            factory_calls["count"] += 1
            return RecordingEngine(
                events,
                fail_on_run=factory_calls["count"] == 1,
            )

        tts = TTS(enabled=True, engine_factory=factory)
        self.addCleanup(tts.shutdown)

        self.assertTrue(tts.speak("First"))
        self._wait_until(lambda: factory_calls["count"] >= 2)
        self._wait_until(lambda: not tts.busy)

        self.assertTrue(tts.speak("Second"))
        self._wait_until(lambda: self._count(events, "say") >= 2)

        spoken = [value for action, value in events if action == "say"]
        self.assertEqual(spoken[:2], ["First", "Second"])
        self.assertFalse(tts.busy)

    @staticmethod
    def _count(events, action):
        return sum(1 for event_action, _ in events if event_action == action)

    @staticmethod
    def _wait_until(predicate, timeout=2.0, interval=0.01):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return
            time.sleep(interval)
        raise AssertionError("Timed out waiting for worker state")


if __name__ == "__main__":
    unittest.main()

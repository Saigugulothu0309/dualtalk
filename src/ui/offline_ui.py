import threading

import cv2
import numpy as np

from src.audio import SpeechToText, TTS
from src.communication.sender import get_hand_label_and_score, predict_gesture_for_hand
from src.config import get as config_get
from src.inference.run_realtime import load_model
from src.processing.placement_engine import detect_zone
from src.processing.smart_sentence_builder import SmartSentenceBuilder
from src.utils.hand_tracker import create_hand_detector, draw_landmarks_on_frame


class OfflineUI:
    """
    No WebSocket. No network. Runs entirely on one device.

    Left half: sign language user with live camera feed and gesture detection.
    Right half: normal user with sign-sentence display and speech or typed input.
    """

    WINDOW = "DualTalk - Offline (Face-to-Face)"
    W = int(config_get("ui.offline_width", 1280))
    H = int(config_get("ui.offline_height", 720))

    def __init__(self):
        self.sign_sentence = ""
        self.normal_text = ""
        self.tts_enabled = bool(config_get("tts.enabled", True))
        self.last_spoken = ""
        self.current_gesture = "No Hand"

        self._builder = SmartSentenceBuilder(
            timeout_seconds=float(config_get("sentence_builder.timeout_seconds", 2.0))
        )
        self._tts = TTS(
            rate=int(config_get("tts.rate", 150)),
            volume=float(config_get("tts.volume", 1.0)),
        )
        if not self.tts_enabled:
            self._tts.disable()

        self._stt = SpeechToText(
            language=str(config_get("stt.language", "en-US")),
            timeout=float(config_get("stt.timeout", 5)),
            phrase_time_limit=float(config_get("stt.phrase_time_limit", 10)),
        )
        self._prediction_buffers = {}
        self._stable_predictions = {}
        self._wave_buffers = {}
        self._no_x_buffers = {}
        self._no_y_buffers = {}
        self._no_latches = {}
        self._no_stable_counts = {}
        self._typed_buffer = ""
        self._use_typed_input = False
        self._running = False
        self._lock = threading.Lock()

    def run(self):
        model = load_model()
        camera_index = int(config_get("camera.index", 0))
        capture = cv2.VideoCapture(camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(config_get("camera.width", 1280)))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config_get("camera.height", 720)))
        capture.set(cv2.CAP_PROP_FPS, int(config_get("camera.fps", 30)))
        if not capture.isOpened():
            raise RuntimeError("Unable to open webcam. Check your camera device.")

        self._running = True
        try:
            self._stt.listen_continuous(self._on_speech)
        except Exception:
            self._use_typed_input = True

        try:
            with create_hand_detector(
                max_num_hands=int(config_get("detection.max_num_hands", 2)),
                min_detection_confidence=float(
                    config_get("detection.detection_confidence", 0.7)
                ),
                min_tracking_confidence=float(
                    config_get("detection.tracking_confidence", 0.7)
                ),
            ) as hands:
                while self._running:
                    success, frame = capture.read()
                    if not success:
                        print("Could not read a frame from the webcam.")
                        break

                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb_frame)
                    detected_gesture = None
                    detected_zone = None
                    self.current_gesture = "No Hand"

                    if result.multi_hand_landmarks:
                        for hand_index, hand_landmarks in enumerate(
                            result.multi_hand_landmarks
                        ):
                            if hand_landmarks is None or len(hand_landmarks.landmark) < 21:
                                continue

                            draw_landmarks_on_frame(frame, hand_landmarks)
                            hand_label, hand_score = get_hand_label_and_score(
                                result,
                                hand_index,
                            )
                            if hand_score < 0.5:
                                continue

                            hand_gesture, _ = predict_gesture_for_hand(
                                model,
                                hand_landmarks.landmark,
                                hand_label,
                                self._prediction_buffers,
                                self._stable_predictions,
                                self._wave_buffers,
                                self._no_x_buffers,
                                self._no_y_buffers,
                                self._no_latches,
                                self._no_stable_counts,
                            )
                            if hand_gesture and hand_gesture != "UNKNOWN":
                                detected_gesture = hand_gesture
                                detected_zone = detect_zone(hand_landmarks.landmark)
                                self.current_gesture = hand_gesture

                    print("DETECTED:", detected_gesture)
                    sentence = self._builder.build_sentence(
                        detected_gesture,
                        placement=detected_zone,
                    )
                    if sentence is not None:
                        print("SENTENCE:", sentence)
                    display_sentence = sentence if sentence is not None else ""

                    if sentence is not None:
                        with self._lock:
                            self.sign_sentence = sentence
                        if self.tts_enabled and sentence != self.last_spoken:
                            self.last_spoken = sentence
                            self._tts.speak(sentence)

                    if self._stt.request_error_detected or self._stt.last_error is not None:
                        self._use_typed_input = True

                    canvas = self._render(frame)
                    cv2.imshow(self.WINDOW, canvas)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("m"):
                        self.tts_enabled = self._tts.toggle()
                    elif key == ord("c"):
                        self._clear_history()
                    elif self._use_typed_input and key != 255:
                        self._handle_typed_key(key)
        finally:
            self._running = False
            self._stt.stop()
            capture.release()
            cv2.destroyAllWindows()

    def _render(self, camera_frame) -> np.ndarray:
        canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        canvas[:] = (16, 18, 24)

        body_height = self.H - 60
        left_width = self.W // 2
        right_x = left_width

        left_frame = cv2.resize(camera_frame, (left_width, body_height))
        canvas[:body_height, :left_width] = left_frame
        canvas[:body_height, right_x:] = (26, 28, 36)

        with self._lock:
            sign_sentence = self.sign_sentence
            normal_text = self.normal_text

        cv2.line(canvas, (left_width, 0), (left_width, body_height), (80, 80, 80), 2)
        cv2.putText(
            canvas,
            "SIGN USER",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "NORMAL USER",
            (right_x + 20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"Gesture: {self.current_gesture}",
            (20, body_height - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Sentence: {sign_sentence}",
            (20, body_height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        sign_display = sign_sentence.upper() if sign_sentence else "WAITING..."
        sign_lines = self._wrap_text(sign_display, self.W // 2 - 60, 1.3, 3)
        for index, line in enumerate(sign_lines[:3]):
            cv2.putText(
                canvas,
                line,
                (right_x + 30, 170 + index * 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            "You said:",
            (right_x + 30, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
        normal_display = normal_text or "Listening..."
        normal_lines = self._wrap_text(normal_display, self.W // 2 - 60, 0.8, 2)
        for index, line in enumerate(normal_lines[:3]):
            cv2.putText(
                canvas,
                line,
                (right_x + 30, 405 + index * 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )

        if self._use_typed_input:
            typed_prompt = self._typed_buffer or "Type here and press Enter"
            cv2.putText(
                canvas,
                f"Type input: {typed_prompt}",
                (right_x + 30, body_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (160, 200, 255),
                2,
                cv2.LINE_AA,
            )

        canvas[body_height:, :] = (38, 38, 38)
        footer = "[Q: quit]  [M: mute TTS]  [C: clear history]"
        cv2.putText(
            canvas,
            footer,
            (20, self.H - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def _on_speech(self, text: str):
        with self._lock:
            self.normal_text = text

    def _handle_typed_key(self, key: int):
        if key == 13:
            text = self._typed_buffer.strip()
            if text:
                self._on_speech(text)
            self._typed_buffer = ""
            return

        if key == 8:
            self._typed_buffer = self._typed_buffer[:-1]
            return

        if 32 <= key <= 126:
            self._typed_buffer += chr(key)

    def _clear_history(self):
        with self._lock:
            self.sign_sentence = ""
            self.normal_text = ""
        self._typed_buffer = ""
        self.last_spoken = ""

    @staticmethod
    def _wrap_text(text: str, max_width: int, scale: float, thickness: int):
        text = str(text or "").strip()
        if not text:
            return []

        words = text.split()
        lines = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            text_width, _ = cv2.getTextSize(
                candidate,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                thickness,
            )[0]
            if text_width <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines


def main():
    OfflineUI().run()


if __name__ == "__main__":
    main()

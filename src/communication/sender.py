import argparse
import asyncio
import csv
import json
import os
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

try:
    import websockets
except ImportError as exc:
    raise SystemExit("Missing dependency: install it with `pip install websockets`.") from exc


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.audio import TTS
from src.config import get as config_get
from src.debug.metrics_engine import MetricsEngine
from src.gestures.gesture_rules import detect_gesture
from src.inference.run_realtime import (
    NO_BUFFER_SIZE,
    STABLE_FRAME_COUNT,
    WAVE_BUFFER_SIZE,
    detect_no_for_hand,
    detect_wave_motion,
    get_display_gesture,
    get_palm_center_x,
    is_open_palm,
    load_model,
    predict_with_model,
    update_stable_prediction,
)
from src.processing.placement_engine import detect_zone
from src.processing.smart_sentence_builder import SmartSentenceBuilder
from src.processing.translator import Translator
from src.security import resolve_crypto


DEFAULT_SERVER_URL = str(config_get("server.url", "ws://127.0.0.1:8765"))
HAND_CONFIDENCE_THRESHOLD = 0.5
SEND_CHANGE_DELAY_SECONDS = float(
    config_get("communication.send_debounce_seconds", 0.2)
)
RECONNECT_DELAY_SECONDS = float(
    config_get("communication.reconnect_delay_seconds", 2.0)
)
KEEPALIVE_INTERVAL_SECONDS = float(
    config_get("communication.keepalive_interval_seconds", 8.0)
)
KEEPALIVE_TIMEOUT_SECONDS = float(
    config_get("communication.keepalive_timeout_seconds", 8.0)
)
CAMERA_RETRY_DELAY_SECONDS = float(
    config_get("camera.retry_delay_seconds", 1.5)
)
TEMP_HAND_LOSS_GRACE_SECONDS = float(
    config_get("detection.temp_hand_loss_grace_seconds", 0.45)
)
STALE_GESTURE_RESET_SECONDS = float(
    config_get("detection.stale_gesture_reset_seconds", 1.0)
)
HAND_RESET_GRACE_SECONDS = max(STALE_GESTURE_RESET_SECONDS, 1.5)
HAND_STATE_TTL_SECONDS = float(
    config_get("detection.hand_state_ttl_seconds", 1.2)
)
WINDOW_NAME = "DualTalk - Gesture Sender"
PILOT_LOG_PATH = os.path.join(BASE_DIR, "logs", "pilot_session.csv")
METRICS_LOG_PATH = os.path.join(BASE_DIR, "logs", "metrics.csv")
EMPTY_VALUE = "-"
INITIALIZING_MESSAGE = "INITIALIZING..."
DEMO_PANEL_WIDTH = 360
DISPLAY_FRAME_WIDTH = 960
DISPLAY_FRAME_HEIGHT = 540
DEFAULT_CAMERA_WIDTH = int(config_get("camera.width", 1280))
DEFAULT_CAMERA_HEIGHT = int(config_get("camera.height", 720))
DEFAULT_PILOT_MODE = True
DEFAULT_METRICS_VISIBLE = False


class QuietSentenceBuilder(SmartSentenceBuilder):
    def check_timeout(self):
        return super().check_timeout()


class PilotSessionLogger:
    FIELDNAMES = [
        "timestamp",
        "gesture",
        "intent",
        "sentence",
        "speech_enabled",
        "latency_ms",
    ]

    def __init__(self, path: str = PILOT_LOG_PATH):
        self.path = path
        self._initialized = False

    def enable(self):
        if self._initialized:
            return

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        file_exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        if not file_exists:
            with open(self.path, "a", newline="", encoding="utf-8") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=self.FIELDNAMES)
                writer.writeheader()
        self._initialized = True

    def log_event(
        self,
        gesture: str,
        intent: str,
        sentence: str,
        speech_enabled: bool,
        latency_ms: float,
    ):
        if not self._initialized:
            self.enable()

        with open(self.path, "a", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=self.FIELDNAMES)
            writer.writerow(
                {
                    "timestamp": datetime.now().astimezone().isoformat(
                        timespec="milliseconds"
                    ),
                    "gesture": gesture,
                    "intent": intent,
                    "sentence": sentence,
                    "speech_enabled": str(bool(speech_enabled)),
                    "latency_ms": f"{latency_ms:.2f}",
                }
            )


def decode_message(raw_message, crypto=None) -> str | None:
    if raw_message is None:
        return None
    if crypto and isinstance(raw_message, bytes):
        try:
            return crypto.decrypt(raw_message)
        except Exception:
            return None
    if isinstance(raw_message, str):
        return raw_message
    return raw_message.decode("utf-8", errors="ignore")


def encode_message(message, crypto=None):
    serialized = json.dumps(message)
    return crypto.encrypt(serialized) if crypto else serialized


def build_communication_message(
    sentence,
    speech_enabled,
    *,
    gesture,
    intent,
    message_id,
    sample_id,
):
    return {
        "type": "communication",
        "message_id": str(message_id),
        "sample_id": str(sample_id),
        "gesture": str(gesture or ""),
        "intent": str(intent or ""),
        "sentence": str(sentence),
        "speech_enabled": bool(speech_enabled),
    }


def build_state_update_message(speech_enabled):
    return {
        "type": "state_update",
        "speech_enabled": bool(speech_enabled),
    }


def parse_control_message(raw_message):
    if raw_message is None:
        return None

    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8", errors="ignore")

    try:
        payload = json.loads(raw_message)
    except (TypeError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    message_type = str(payload.get("type", "")).strip().lower()
    if message_type != "delivery_ack":
        return None

    message_id = str(payload.get("message_id", "")).strip()
    if not message_id:
        return None

    return {
        "type": message_type,
        "message_id": message_id,
    }


def open_camera(camera_index):
    capture = cv2.VideoCapture(camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, int(config_get("camera.fps", 30)))
    if not capture.isOpened():
        capture.release()
        return None
    return capture


def clear_stale_hand_state(
    prediction_buffers,
    stable_predictions,
    wave_buffers,
    no_x_buffers,
    no_y_buffers,
    no_latches,
    no_stable_counts,
    hand_last_seen_at,
    current_time,
):
    stale_keys = [
        hand_key
        for hand_key, last_seen_at in hand_last_seen_at.items()
        if (current_time - last_seen_at) >= HAND_STATE_TTL_SECONDS
    ]
    for hand_key in stale_keys:
        prediction_buffers.pop(hand_key, None)
        stable_predictions.pop(hand_key, None)
        wave_buffers.pop(hand_key, None)
        no_x_buffers.pop(hand_key, None)
        no_y_buffers.pop(hand_key, None)
        no_latches.pop(hand_key, None)
        no_stable_counts.pop(hand_key, None)
        hand_last_seen_at.pop(hand_key, None)


def reset_sender_state(
    sentence_builder,
    ui_state,
):
    sentence_builder.clear()
    ui_state["gesture"] = "READY"
    ui_state["intent"] = EMPTY_VALUE
    ui_state["sentence"] = EMPTY_VALUE


def apply_live_metrics(ui_state, metrics_engine):
    live_metrics = metrics_engine.get_live_snapshot()
    ui_state["fps"] = live_metrics.fps
    ui_state["latency_ms"] = live_metrics.latency_ms
    ui_state["cpu_percent"] = live_metrics.cpu_percent
    ui_state["ram_percent"] = live_metrics.ram_percent


def finalize_pending_events(pending_events, metrics_engine, pilot_logger):
    for message_id in list(pending_events):
        pending_event = pending_events.pop(message_id)
        record = metrics_engine.finalize_local(sample_id=pending_event["sample_id"])
        if pilot_logger is not None and pending_event.get("pilot_enabled"):
            pilot_logger.log_event(
                gesture=pending_event["gesture"],
                intent=pending_event["intent"],
                sentence=pending_event["sentence"],
                speech_enabled=pending_event["speech_enabled"],
                latency_ms=record.total_ms or 0.0,
            )


def wrap_text(text, max_width, font, scale, thickness):
    text = str(text or "").strip()
    if not text:
        return []

    words = text.split()
    lines = []
    current_line = words[0]
    for word in words[1:]:
        candidate = f"{current_line} {word}"
        text_width, _ = cv2.getTextSize(candidate, font, scale, thickness)[0]
        if text_width <= max_width:
            current_line = candidate
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines


def draw_text_lines(frame, lines, x, y, font, scale, color, thickness, line_height):
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + index * line_height),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def resize_frame_to_fit(frame, target_width, target_height):
    frame_height, frame_width = frame.shape[:2]
    if frame_height <= 0 or frame_width <= 0:
        return frame

    scale = min(target_width / frame_width, target_height / frame_height, 1.0)
    if scale >= 0.999:
        return frame

    resized_width = max(1, int(frame_width * scale))
    resized_height = max(1, int(frame_height * scale))
    return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def get_hand_label_and_score(result, hand_index):
    if result.multi_handedness is None or hand_index >= len(result.multi_handedness):
        return f"Hand {hand_index + 1}", 1.0

    handedness = result.multi_handedness[hand_index]
    if handedness is None or not handedness.classification:
        return f"Hand {hand_index + 1}", 1.0

    classification = handedness.classification[0]
    return classification.label, classification.score


def predict_gesture_for_hand(
    model,
    landmarks,
    hand_key,
    prediction_buffers,
    stable_predictions,
    wave_buffers,
    no_x_buffers,
    no_y_buffers,
    no_latches,
    no_stable_counts,
):
    if hand_key not in prediction_buffers:
        prediction_buffers[hand_key] = deque(maxlen=STABLE_FRAME_COUNT)
        stable_predictions[hand_key] = ("UNKNOWN", 0.0)
        wave_buffers[hand_key] = deque(maxlen=WAVE_BUFFER_SIZE)
        no_x_buffers[hand_key] = deque(maxlen=NO_BUFFER_SIZE)
        no_y_buffers[hand_key] = deque(maxlen=NO_BUFFER_SIZE)
        no_latches[hand_key] = 0
        no_stable_counts[hand_key] = 0

    if is_open_palm(landmarks):
        wave_buffers[hand_key].append(get_palm_center_x(landmarks))
    else:
        wave_buffers[hand_key].clear()

    if is_open_palm(landmarks) and detect_wave_motion(wave_buffers[hand_key]):
        gesture_text = "WAVE"
    elif detect_no_for_hand(
        landmarks,
        hand_key,
        no_x_buffers,
        no_y_buffers,
        no_latches,
        no_stable_counts,
    ):
        gesture_text = "NO"
    else:
        gesture_text = detect_gesture(landmarks)

    if gesture_text == "UNKNOWN":
        gesture_text, confidence = predict_with_model(model, landmarks)
    else:
        confidence = 1.0

    stable_predictions[hand_key] = update_stable_prediction(
        prediction_buffers[hand_key],
        stable_predictions[hand_key],
        gesture_text,
        confidence,
    )
    return stable_predictions[hand_key]


def draw_status_block(frame, label, value, x, y, width):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame,
        label,
        (x, y),
        font,
        0.58,
        (170, 190, 210),
        1,
        cv2.LINE_AA,
    )
    value_lines = wrap_text(value, width, font, 0.72, 2)[:2] or [EMPTY_VALUE]
    draw_text_lines(
        frame,
        value_lines,
        x,
        y + 26,
        font,
        0.72,
        (255, 255, 255),
        2,
        26,
    )
    return y + 34 + len(value_lines) * 26


def render_demo_frame(camera_frame, ui_state):
    if camera_frame is None:
        frame = np.zeros((DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, 3), dtype=np.uint8)
        frame[:] = (12, 14, 18)
    else:
        frame = camera_frame.copy()

    frame_height, frame_width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (18, 18),
        (min(DEMO_PANEL_WIDTH, frame_width - 18), frame_height - 18),
        (14, 16, 22),
        -1,
    )
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    panel_width = min(DEMO_PANEL_WIDTH - 32, frame_width - 64)
    panel_x = 32
    cursor_y = 48

    cv2.putText(
        frame,
        "DualTalk Validation" if ui_state.get("test_mode") else "DualTalk MVP",
        (panel_x, cursor_y),
        font,
        0.82,
        (180, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cursor_y += 34

    if ui_state.get("test_mode"):
        cursor_y = draw_status_block(
            frame,
            "Expected Gesture",
            str(ui_state.get("expected_gesture", EMPTY_VALUE)),
            panel_x,
            cursor_y,
            panel_width,
        )
        cursor_y = draw_status_block(
            frame,
            "Current Gesture",
            str(ui_state.get("gesture", "READY")),
            panel_x,
            cursor_y + 6,
            panel_width,
        )
        cursor_y = draw_status_block(
            frame,
            "Confidence",
            f"{float(ui_state.get('confidence', 0.0)) * 100:.0f}%",
            panel_x,
            cursor_y + 6,
            panel_width,
        )
        cursor_y = draw_status_block(
            frame,
            "Intent",
            str(ui_state.get("intent", EMPTY_VALUE)),
            panel_x,
            cursor_y + 6,
            panel_width,
        )
        cursor_y = draw_status_block(
            frame,
            "Generated Sentence",
            str(ui_state.get("sentence", EMPTY_VALUE)),
            panel_x,
            cursor_y + 6,
            panel_width,
        )
        cursor_y = draw_status_block(
            frame,
            "Detection Count",
            str(ui_state.get("detection_count", 0)),
            panel_x,
            cursor_y + 6,
            panel_width,
        )
        cursor_y = draw_status_block(
            frame,
            "Current Accuracy",
            str(ui_state.get("accuracy_text", "0/0")),
            panel_x,
            cursor_y + 6,
            panel_width,
        )
        draw_status_block(
            frame,
            "Progress",
            (
                f"{ui_state.get('gesture_progress', '0/0')}  "
                f"Session {ui_state.get('session_count', 0)}"
            ),
            panel_x,
            cursor_y + 6,
            panel_width,
        )
    else:
        cursor_y = draw_status_block(
            frame,
            "Connection",
            str(ui_state.get("connection_status", "CONNECTING")),
            panel_x,
            cursor_y,
            panel_width,
        )
        cursor_y = draw_status_block(
            frame,
            "Speech",
            "ON" if ui_state.get("speech_enabled") else "OFF",
            panel_x,
            cursor_y + 6,
            panel_width,
        )
        cursor_y = draw_status_block(
            frame,
            "Gesture",
            str(ui_state.get("gesture", "READY")),
            panel_x,
            cursor_y + 6,
            panel_width,
        )
        cursor_y = draw_status_block(
            frame,
            "Intent",
            str(ui_state.get("intent", EMPTY_VALUE)),
            panel_x,
            cursor_y + 6,
            panel_width,
        )
        cursor_y = draw_status_block(
            frame,
            "Sentence",
            str(ui_state.get("sentence", EMPTY_VALUE)),
            panel_x,
            cursor_y + 6,
            panel_width,
        )
        cursor_y = draw_status_block(
            frame,
            "FPS",
            f"{ui_state.get('fps', 0.0):.1f}",
            panel_x,
            cursor_y + 6,
            panel_width,
        )
        cursor_y = draw_status_block(
            frame,
            "Latency",
            f"{ui_state.get('latency_ms', 0.0):.1f} ms",
            panel_x,
            cursor_y + 6,
            panel_width,
        )
        if ui_state.get("metrics_visible", True):
            cursor_y = draw_status_block(
                frame,
                "CPU",
                f"{ui_state.get('cpu_percent', 0.0):.1f} %",
                panel_x,
                cursor_y + 6,
                panel_width,
            )
            draw_status_block(
                frame,
                "RAM",
                f"{ui_state.get('ram_percent', 0.0):.1f} %",
                panel_x,
                cursor_y + 6,
                panel_width,
            )

    startup_message = str(ui_state.get("startup_message", "")).strip()
    if startup_message:
        banner_width = min(360, frame_width - panel_x - 40)
        banner_height = 56
        banner_x = frame_width - banner_width - 28
        banner_y = 28
        cv2.rectangle(
            frame,
            (banner_x, banner_y),
            (banner_x + banner_width, banner_y + banner_height),
            (26, 30, 38),
            -1,
        )
        cv2.putText(
            frame,
            startup_message,
            (banner_x + 18, banner_y + 36),
            font,
            0.82,
            (0, 210, 255),
            2,
            cv2.LINE_AA,
        )

    if ui_state.get("test_mode"):
        mode_text = (
            f"Test Mode  Threshold {float(ui_state.get('threshold', 0.0)) * 100:.0f}%"
        )
        shortcut_text = "N Next  B Back  Q Quit"
    else:
        mode_text = (
            f"Pilot {'ON' if ui_state.get('pilot_enabled') else 'OFF'}"
            f"  Metrics {'ON' if ui_state.get('metrics_visible') else 'OFF'}"
        )
        shortcut_text = "V Speech  P Pilot  M Metrics  Q Quit"
    cv2.putText(
        frame,
        mode_text,
        (panel_x, frame_height - 56),
        font,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        shortcut_text,
        (panel_x, frame_height - 30),
        font,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    if ui_state.get("camera_lost"):
        warning_text = "CAMERA LOST"
        text_size = cv2.getTextSize(warning_text, font, 1.8, 4)[0]
        text_x = max((frame_width - text_size[0]) // 2, panel_x + 20)
        text_y = frame_height // 2
        cv2.putText(
            frame,
            warning_text,
            (text_x, text_y),
            font,
            1.8,
            (0, 80, 255),
            4,
            cv2.LINE_AA,
        )

    return frame


def show_sender_frame(camera_frame, ui_state):
    display_frame = render_demo_frame(camera_frame, ui_state)
    cv2.imshow(
        WINDOW_NAME,
        resize_frame_to_fit(display_frame, DISPLAY_FRAME_WIDTH, DISPLAY_FRAME_HEIGHT),
    )


def handle_sender_keypress(key, ui_state, pilot_logger, speech=None):
    if key in {ord("q"), ord("Q")}:
        return True

    if key in {ord("v"), ord("V")}:
        target_state = not ui_state.get("speech_enabled", False)
        if speech is None:
            speech_enabled = target_state
        else:
            speech_enabled = speech.enable() if target_state else speech.disable()
        ui_state["speech_enabled"] = speech_enabled
        ui_state["pending_state_sync"] = True
        print("SPEECH TOGGLED: ON" if speech_enabled else "SPEECH TOGGLED: OFF")
    elif key in {ord("p"), ord("P")}:
        ui_state["pilot_enabled"] = not ui_state.get("pilot_enabled", False)
        if ui_state["pilot_enabled"] and pilot_logger is not None:
            pilot_logger.enable()
        print("PILOT MODE: ON" if ui_state["pilot_enabled"] else "PILOT MODE: OFF")
    elif key in {ord("m"), ord("M")}:
        ui_state["metrics_visible"] = not ui_state.get("metrics_visible", True)
        print("METRICS: ON" if ui_state["metrics_visible"] else "METRICS: OFF")

    return False


async def wait_with_status(ui_state, pilot_logger, duration_seconds):
    deadline = time.time() + max(0.0, duration_seconds)
    while time.time() < deadline:
        show_sender_frame(None, ui_state)
        if handle_sender_keypress(cv2.waitKey(1) & 0xFF, ui_state, pilot_logger):
            return True
        await asyncio.sleep(0.05)
    return False


async def keep_websocket_alive(websocket, label):
    try:
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL_SECONDS)
            pong_waiter = await websocket.ping()
            await asyncio.wait_for(pong_waiter, timeout=KEEPALIVE_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"{label} keepalive error: {exc}")
        try:
            await websocket.close()
        except Exception:
            pass
        raise OSError(f"{label} keepalive failed: {exc}") from exc


async def receive_remote_text(
    websocket,
    metrics_engine,
    ui_state,
    pending_events,
    pilot_logger,
    crypto=None,
):
    async for raw_message in websocket:
        text = decode_message(raw_message, crypto)
        if text is None:
            continue
        control_message = parse_control_message(text)
        if control_message is None:
            continue
        if control_message["type"] != "delivery_ack":
            continue

        record = metrics_engine.mark_delivery(control_message["message_id"])
        if record is None:
            continue

        apply_live_metrics(ui_state, metrics_engine)
        pending_event = pending_events.pop(control_message["message_id"], None)
        if pending_event is None:
            continue
        if pending_event.get("pilot_enabled") and pilot_logger is not None:
            pilot_logger.log_event(
                gesture=pending_event["gesture"],
                intent=pending_event["intent"],
                sentence=pending_event["sentence"],
                speech_enabled=pending_event["speech_enabled"],
                latency_ms=record.total_ms or 0.0,
            )


async def send_state_update(websocket, ui_state, crypto=None, *, force=False):
    if not force and not ui_state.get("pending_state_sync", False):
        return

    payload = encode_message(
        build_state_update_message(ui_state.get("speech_enabled", False)),
        crypto=crypto,
    )
    try:
        await websocket.send(payload)
    except Exception as exc:
        ui_state["connection_status"] = "DISCONNECTED"
        raise OSError(f"Failed to send state update: {exc}") from exc

    ui_state["pending_state_sync"] = False
    print("STATE SENT")


async def connect_sender_websocket(server_url):
    return await websockets.connect(
        server_url,
        compression=None,
        ping_interval=20,
        ping_timeout=20,
        max_queue=1,
    )


async def stream_gestures(
    server_url,
    camera_index,
    translator,
    crypto=None,
    ui_state=None,
    pilot_logger=None,
    metrics_engine=None,
):
    _ = translator
    metrics_engine = metrics_engine or MetricsEngine(
        csv_path=METRICS_LOG_PATH,
        print_metrics=False,
    )
    ui_state = ui_state or {}
    ui_state.setdefault("gesture", "READY")
    ui_state.setdefault("intent", EMPTY_VALUE)
    ui_state.setdefault("sentence", EMPTY_VALUE)
    ui_state.setdefault("speech_enabled", False)
    ui_state.setdefault("pilot_enabled", DEFAULT_PILOT_MODE)
    ui_state.setdefault("connection_status", "CONNECTING")
    ui_state.setdefault("fps", 0.0)
    ui_state.setdefault("latency_ms", 0.0)
    ui_state.setdefault("cpu_percent", 0.0)
    ui_state.setdefault("ram_percent", 0.0)
    ui_state.setdefault("camera_lost", False)
    ui_state.setdefault("metrics_visible", DEFAULT_METRICS_VISIBLE)
    ui_state.setdefault("pending_state_sync", False)
    ui_state.setdefault("startup_message", "")

    model = None
    model_loaded = False
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    prediction_buffers = {}
    stable_predictions = {}
    wave_buffers = {}
    no_x_buffers = {}
    no_y_buffers = {}
    no_latches = {}
    no_stable_counts = {}
    hand_last_seen_at = {}
    last_processed_gesture = None
    last_sent_signature = None
    last_visible_gesture = "READY"
    hand_loss_started_at = None
    pending_events = {}
    sentence_builder = QuietSentenceBuilder(
        timeout_seconds=float(config_get("sentence_builder.timeout_seconds", 2.0))
    )
    speech = TTS(
        rate=int(config_get("tts.rate", 150)),
        volume=float(config_get("tts.volume", 1.0)),
        enabled=ui_state.get("speech_enabled", False),
    )

    ui_state["connection_status"] = "CONNECTING"
    ui_state["startup_message"] = INITIALIZING_MESSAGE
    ui_state["gesture"] = INITIALIZING_MESSAGE
    ui_state["intent"] = EMPTY_VALUE
    ui_state["sentence"] = INITIALIZING_MESSAGE

    receive_task = None
    keepalive_task = None
    websocket = None
    websocket_task = asyncio.create_task(connect_sender_websocket(server_url))
    model_task = asyncio.create_task(asyncio.to_thread(load_model))
    cap = open_camera(camera_index)
    last_camera_retry_at = time.time() if cap is not None else 0.0
    user_requested_stop = False
    initial_frame = None
    if cap is not None and cap.isOpened():
        success, initial_frame = cap.read()
        if success:
            initial_frame = cv2.flip(initial_frame, 1)
        else:
            initial_frame = None

    try:
        apply_live_metrics(ui_state, metrics_engine)
        show_sender_frame(initial_frame, ui_state)
        cv2.waitKey(1)

        with mp_hands.Hands(
            max_num_hands=int(config_get("detection.max_num_hands", 2)),
            min_detection_confidence=float(
                config_get("detection.detection_confidence", 0.7)
            ),
            min_tracking_confidence=float(
                config_get("detection.tracking_confidence", 0.7)
            ),
        ) as hands:
            while True:
                if not model_loaded and model_task.done():
                    model = model_task.result()
                    model_loaded = True

                if websocket is None and websocket_task.done():
                    websocket = websocket_task.result()
                    ui_state["connection_status"] = "CONNECTED"
                    ui_state["startup_message"] = ""
                    if ui_state.get("gesture") == INITIALIZING_MESSAGE:
                        reset_sender_state(sentence_builder, ui_state)
                    if receive_task is not None:
                        receive_task.cancel()
                        try:
                            await receive_task
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            pass
                    receive_task = asyncio.create_task(
                        receive_remote_text(
                            websocket,
                            metrics_engine,
                            ui_state,
                            pending_events,
                            pilot_logger,
                            crypto=crypto,
                        )
                    )
                    keepalive_task = asyncio.create_task(
                        keep_websocket_alive(websocket, "Sender")
                    )
                    await send_state_update(websocket, ui_state, crypto=crypto, force=True)

                if websocket is None and websocket_task.done():
                    raise OSError("Sender websocket task ended before connecting.")

                if websocket is not None and receive_task is not None and receive_task.done():
                    task_exception = receive_task.exception()
                    if task_exception is not None:
                        raise task_exception
                    raise OSError("Sender receiver task ended.")

                if keepalive_task is not None and keepalive_task.done():
                    task_exception = keepalive_task.exception()
                    if task_exception is not None:
                        raise task_exception
                    raise OSError("Sender keepalive task ended.")

                frame_started_at = time.perf_counter()
                metrics_engine.tick_frame(frame_started_at=frame_started_at)
                apply_live_metrics(ui_state, metrics_engine)

                if cap is None or not cap.isOpened():
                    if cap is not None and not cap.isOpened():
                        cap.release()
                        cap = None
                    camera_retry_time = time.time()
                    if (
                        cap is None
                        and (camera_retry_time - last_camera_retry_at)
                        >= CAMERA_RETRY_DELAY_SECONDS
                    ):
                        last_camera_retry_at = camera_retry_time
                        cap = open_camera(camera_index)

                    ui_state["camera_lost"] = True
                    ui_state["gesture"] = "CAMERA LOST"
                    if hand_loss_started_at is None:
                        hand_loss_started_at = frame_started_at
                    if (
                        hand_loss_started_at is not None
                        and (frame_started_at - hand_loss_started_at)
                        >= HAND_RESET_GRACE_SECONDS
                    ):
                        reset_sender_state(sentence_builder, ui_state)
                        prediction_buffers.clear()
                        stable_predictions.clear()
                        wave_buffers.clear()
                        no_x_buffers.clear()
                        no_y_buffers.clear()
                        no_latches.clear()
                        no_stable_counts.clear()
                        hand_last_seen_at.clear()
                        last_processed_input = None
                        last_sent_signature = None
                        last_visible_gesture = "READY"
                        ui_state["gesture"] = "CAMERA LOST"
                    show_sender_frame(None, ui_state)
                    if handle_sender_keypress(
                        cv2.waitKey(1) & 0xFF,
                        ui_state,
                        pilot_logger,
                        speech=speech,
                    ):
                        user_requested_stop = True
                        break
                    if websocket is not None:
                        await send_state_update(websocket, ui_state, crypto=crypto)
                    await asyncio.sleep(0.1)
                    continue

                success, frame = cap.read()
                if not success:
                    ui_state["camera_lost"] = True
                    ui_state["gesture"] = "CAMERA LOST"
                    cap.release()
                    cap = None
                    show_sender_frame(None, ui_state)
                    if handle_sender_keypress(
                        cv2.waitKey(1) & 0xFF,
                        ui_state,
                        pilot_logger,
                        speech=speech,
                    ):
                        user_requested_stop = True
                        break
                    if websocket is not None:
                        await send_state_update(websocket, ui_state, crypto=crypto)
                    await asyncio.sleep(0.1)
                    continue

                ui_state["camera_lost"] = False
                frame = cv2.flip(frame, 1)

                if websocket is None:
                    ui_state["connection_status"] = "CONNECTING"
                    ui_state["startup_message"] = INITIALIZING_MESSAGE
                    ui_state["gesture"] = INITIALIZING_MESSAGE
                    ui_state["intent"] = EMPTY_VALUE
                    ui_state["sentence"] = INITIALIZING_MESSAGE
                    show_sender_frame(frame, ui_state)
                    if handle_sender_keypress(
                        cv2.waitKey(1) & 0xFF,
                        ui_state,
                        pilot_logger,
                        speech=speech,
                    ):
                        user_requested_stop = True
                        break
                    await asyncio.sleep(0.01)
                    continue

                ui_state["startup_message"] = ""
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)
                detected_gesture = None
                detected_zone = None
                detected_confidence = 0.0
                display_gesture = "READY"

                if result.multi_hand_landmarks:
                    for hand_index, hand_landmarks in enumerate(
                        result.multi_hand_landmarks
                    ):
                        if hand_landmarks is None or len(hand_landmarks.landmark) < 21:
                            continue

                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                        )

                        hand_label, hand_score = get_hand_label_and_score(
                            result,
                            hand_index,
                        )
                        if hand_score < HAND_CONFIDENCE_THRESHOLD:
                            continue
                        hand_last_seen_at[hand_label] = frame_started_at
                        hand_zone = detect_zone(hand_landmarks.landmark)

                        hand_gesture, confidence = predict_gesture_for_hand(
                            model,
                            hand_landmarks.landmark,
                            hand_label,
                            prediction_buffers,
                            stable_predictions,
                            wave_buffers,
                            no_x_buffers,
                            no_y_buffers,
                            no_latches,
                            no_stable_counts,
                        )

                        candidate_display_gesture = get_display_gesture(
                            hand_gesture,
                            confidence,
                        )
                        if (
                            candidate_display_gesture != "UNSURE"
                            and hand_gesture != "UNKNOWN"
                            and confidence >= detected_confidence
                        ):
                            detected_gesture = hand_gesture
                            detected_zone = hand_zone
                            detected_confidence = confidence
                            display_gesture = candidate_display_gesture
                        elif detected_gesture is None:
                            detected_gesture = candidate_display_gesture
                            detected_zone = hand_zone
                            detected_confidence = confidence
                            display_gesture = candidate_display_gesture

                clear_stale_hand_state(
                    prediction_buffers,
                    stable_predictions,
                    wave_buffers,
                    no_x_buffers,
                    no_y_buffers,
                    no_latches,
                    no_stable_counts,
                    hand_last_seen_at,
                    frame_started_at,
                )

                if detected_gesture is None:
                    if hand_loss_started_at is None:
                        hand_loss_started_at = frame_started_at
                    if (
                        last_processed_gesture is not None
                        and (frame_started_at - hand_loss_started_at)
                        < TEMP_HAND_LOSS_GRACE_SECONDS
                    ):
                        ui_state["gesture"] = last_visible_gesture
                    else:
                        ui_state["gesture"] = "READY"
                    if (
                        hand_loss_started_at is not None
                        and (frame_started_at - hand_loss_started_at)
                        >= HAND_RESET_GRACE_SECONDS
                    ):
                        reset_sender_state(sentence_builder, ui_state)
                        prediction_buffers.clear()
                        stable_predictions.clear()
                        wave_buffers.clear()
                        no_x_buffers.clear()
                        no_y_buffers.clear()
                        no_latches.clear()
                        no_stable_counts.clear()
                        hand_last_seen_at.clear()
                        last_processed_gesture = None
                        last_sent_signature = None
                        last_visible_gesture = "READY"
                else:
                    hand_loss_started_at = None
                    ui_state["gesture"] = display_gesture
                    last_visible_gesture = display_gesture
                    
                    if (
                        detected_gesture != "UNSURE"
                        and detected_gesture != "UNKNOWN"
                    ):
                        if detected_gesture == last_processed_gesture:
                            print("SAME GESTURE SKIPPED")
                        else:
                            print("NEW GESTURE DETECTED")
                            last_processed_gesture = detected_gesture
                            
                            sample_id = metrics_engine.start_sample(
                                gesture=detected_gesture,
                                frame_started_at=frame_started_at,
                            )
                            metrics_engine.mark_detection(
                                gesture=detected_gesture,
                                sample_id=sample_id,
                            )
                            trace = sentence_builder.build_sentence_trace(
                                detected_gesture,
                                placement=detected_zone,
                            )
                            metrics_engine.mark_intent(
                                intent=trace["intent"],
                                sample_id=sample_id,
                                intent_started_at=trace["intent_started_at"],
                                intent_completed_at=trace["intent_completed_at"],
                            )
                            metrics_engine.mark_sentence(
                                sentence=trace["sentence"],
                                sample_id=sample_id,
                                sentence_started_at=trace["sentence_started_at"],
                                sentence_completed_at=trace["sentence_completed_at"],
                            )
                            apply_live_metrics(ui_state, metrics_engine)

                            intent = trace["intent"] or EMPTY_VALUE
                            sentence = trace["sentence"]
                            ui_state["intent"] = intent
                            ui_state["sentence"] = sentence or EMPTY_VALUE

                            if not sentence:
                                metrics_engine.discard_sample(sample_id=sample_id)
                            else:
                                event_signature = (
                                    detected_gesture,
                                    trace["zone"],
                                    intent,
                                    sentence,
                                )
                                if event_signature == last_sent_signature:
                                    metrics_engine.discard_sample(sample_id=sample_id)
                                else:
                                    message_id = metrics_engine.mark_send(sample_id=sample_id)
                                    payload = encode_message(
                                        build_communication_message(
                                            sentence,
                                            ui_state.get("speech_enabled", False),
                                            gesture=detected_gesture,
                                            intent=trace["intent"],
                                            message_id=message_id,
                                            sample_id=sample_id,
                                        ),
                                        crypto=crypto,
                                    )
                                    try:
                                        await websocket.send(payload)
                                    except Exception as exc:
                                        ui_state["connection_status"] = "DISCONNECTED"
                                        raise OSError(f"Failed to send message: {exc}") from exc

                                    pending_events[message_id] = {
                                        "sample_id": sample_id,
                                        "gesture": detected_gesture,
                                        "intent": intent,
                                        "sentence": sentence,
                                        "speech_enabled": ui_state.get("speech_enabled", False),
                                        "pilot_enabled": ui_state.get("pilot_enabled", False),
                                    }
                                    last_sent_signature = event_signature
                                    apply_live_metrics(ui_state, metrics_engine)
                                    print(f"GESTURE DETECTED: {detected_gesture}")
                                    print(f"INTENT: {intent}")
                                    print(f"FINAL SENTENCE: {sentence}")
                                    print(f"SENDER SENT: {sentence}")

                                    if ui_state.get("speech_enabled") and speech.speak(sentence):
                                        print(f"SPEAKING SENTENCE: {sentence}")

                                    await asyncio.sleep(SEND_CHANGE_DELAY_SECONDS)

                ui_state["connection_status"] = "CONNECTED"
                show_sender_frame(frame, ui_state)

                if handle_sender_keypress(
                    cv2.waitKey(1) & 0xFF,
                    ui_state,
                    pilot_logger,
                    speech=speech,
                ):
                    user_requested_stop = True
                    break
                if websocket is not None:
                    await send_state_update(websocket, ui_state, crypto=crypto)

                await asyncio.sleep(0)
    finally:
        finalize_pending_events(pending_events, metrics_engine, pilot_logger)
        if websocket_task is not None and not websocket_task.done():
            websocket_task.cancel()
            try:
                await websocket_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        if model_task is not None and not model_task.done():
            model_task.cancel()
        if keepalive_task is not None:
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        if receive_task is not None:
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        if cap is not None:
            cap.release()
        speech.shutdown()
        if websocket is not None:
            try:
                await websocket.close()
            except Exception:
                pass

    return user_requested_stop


async def run_with_reconnect(server_url, camera_index, translator, crypto=None):
    ui_state = {
        "gesture": "READY",
        "intent": EMPTY_VALUE,
        "sentence": EMPTY_VALUE,
        "speech_enabled": False,
        "pilot_enabled": DEFAULT_PILOT_MODE,
        "connection_status": "CONNECTING",
        "fps": 0.0,
        "latency_ms": 0.0,
        "cpu_percent": 0.0,
        "ram_percent": 0.0,
        "camera_lost": False,
        "metrics_visible": DEFAULT_METRICS_VISIBLE,
        "startup_message": INITIALIZING_MESSAGE,
    }
    pilot_logger = PilotSessionLogger()
    metrics_engine = MetricsEngine(csv_path=METRICS_LOG_PATH, print_metrics=False)
    if ui_state["pilot_enabled"]:
        pilot_logger.enable()

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, DISPLAY_FRAME_WIDTH, DISPLAY_FRAME_HEIGHT)
        while True:
            ui_state["connection_status"] = "CONNECTING"
            ui_state["camera_lost"] = False
            ui_state["startup_message"] = INITIALIZING_MESSAGE

            try:
                user_requested_stop = await stream_gestures(
                    server_url,
                    camera_index,
                    translator=translator,
                    crypto=crypto,
                    ui_state=ui_state,
                    pilot_logger=pilot_logger,
                    metrics_engine=metrics_engine,
                )
                if user_requested_stop:
                    break

                ui_state["connection_status"] = "RECONNECTING"
                ui_state["startup_message"] = INITIALIZING_MESSAGE
                print(f"Sender reconnecting in {RECONNECT_DELAY_SECONDS:.0f}s...")
                if await wait_with_status(
                    ui_state,
                    pilot_logger,
                    RECONNECT_DELAY_SECONDS,
                ):
                    break
            except (OSError, RuntimeError, websockets.exceptions.ConnectionClosed) as exc:
                ui_state["connection_status"] = "RECONNECTING"
                ui_state["camera_lost"] = isinstance(exc, RuntimeError)
                ui_state["startup_message"] = INITIALIZING_MESSAGE
                print(f"Sender disconnected: {exc}")
                print(f"Retrying in {RECONNECT_DELAY_SECONDS:.0f}s...")
                if await wait_with_status(
                    ui_state,
                    pilot_logger,
                    RECONNECT_DELAY_SECONDS,
                ):
                    break
    finally:
        metrics_engine.close()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Send detected gesture text to a server.")
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER_URL,
        help="WebSocket server URL, for example ws://192.168.1.20:8765",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=int(config_get("camera.index", 0)),
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--language",
        default=str(config_get("language.default", "en")),
        help="Translation target language name or code, for example Hindi or te.",
    )
    parser.add_argument(
        "--encrypt",
        action="store_true",
        default=bool(config_get("communication.encrypted", False)),
        help="Encrypt outgoing messages with a shared Fernet key.",
    )
    parser.add_argument(
        "--key",
        default=config_get("communication.encryption_key"),
        help="Shared Fernet key, optionally prefixed with DUALTALK_KEY=",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    translator = Translator(str(config_get("language.default", "en")))
    translator.set_language(args.language)
    crypto = resolve_crypto(
        args.encrypt,
        key_str=args.key,
        generate_if_missing=args.encrypt and not args.key,
    )
    try:
        asyncio.run(
            run_with_reconnect(
                args.server,
                args.camera,
                translator=translator,
                crypto=crypto,
            )
        )
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("Sender stopped safely")


if __name__ == "__main__":
    main()

import argparse
import asyncio
import base64
import json
import mimetypes
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import mediapipe as mp

try:
    import websockets
except ImportError as exc:
    raise SystemExit("Missing dependency: install it with `pip install websockets`.") from exc

from src.audio import TTS
from src.communication.receiver import build_delivery_ack_message, decode_message
from src.communication.sender import (
    build_communication_message,
    clear_stale_hand_state,
    get_hand_label_and_score,
    open_camera,
    predict_gesture_for_hand,
    resize_frame_to_fit,
)
from src.config import get as config_get
from src.inference.run_realtime import get_display_gesture, load_model
from src.processing.placement_engine import detect_zone
from src.processing.smart_sentence_builder import SmartSentenceBuilder
from src.security import resolve_crypto


BASE_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = BASE_DIR / "web"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"
INDEX_HTML_PATH = TEMPLATES_DIR / "index.html"
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8000
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8766
DEFAULT_SERVER_URL = str(config_get("server.url", "ws://127.0.0.1:8765"))
DEFAULT_REMOTE_FRAME_TIMEOUT_SECONDS = 2.5
DEFAULT_VIDEO_FRAME_INTERVAL_SECONDS = 0.08
DEFAULT_CAMERA_IDLE_SLEEP_SECONDS = 0.08
DEFAULT_CAMERA_RETRY_DELAY_SECONDS = float(
    config_get("camera.retry_delay_seconds", 1.5)
)
TEMP_HAND_LOSS_GRACE_SECONDS = float(
    config_get("detection.temp_hand_loss_grace_seconds", 0.45)
)
STALE_GESTURE_RESET_SECONDS = float(
    config_get("detection.stale_gesture_reset_seconds", 1.0)
)
HAND_RESET_GRACE_SECONDS = max(STALE_GESTURE_RESET_SECONDS, 1.5)
HAND_CONFIDENCE_THRESHOLD = 0.5
CHAT_HISTORY_LIMIT = 80
EMPTY_VALUE = "-"


def now_timestamp():
    return time.time()


def now_time_label():
    return datetime.now().strftime("%H:%M")


def parse_json_message(raw_message):
    if isinstance(raw_message, bytes):
        try:
            raw_message = raw_message.decode("utf-8")
        except UnicodeDecodeError:
            return None

    if not isinstance(raw_message, str):
        return None

    try:
        return json.loads(raw_message)
    except json.JSONDecodeError:
        return None


def build_state_update_message(
    speech_enabled,
    *,
    camera_enabled=None,
    ai_enabled=None,
    role=None,
):
    payload = {
        "type": "state_update",
        "speech_enabled": bool(speech_enabled),
    }
    if camera_enabled is not None:
        payload["camera_enabled"] = bool(camera_enabled)
    if ai_enabled is not None:
        payload["ai_enabled"] = bool(ai_enabled)
    if role is not None:
        payload["role"] = str(role)
    return payload


def parse_state_update_message(payload):
    if not isinstance(payload, dict):
        return None
    if str(payload.get("type", "")).strip().lower() != "state_update":
        return None
    return {
        "speech_enabled": bool(payload.get("speech_enabled", False)),
        "camera_enabled": bool(payload.get("camera_enabled", False)),
        "ai_enabled": bool(payload.get("ai_enabled", False)),
        "role": str(payload.get("role", "")).strip().lower() or None,
    }


def encode_jpeg(frame, *, target_width=640, target_height=480, quality=72):
    if frame is None:
        return None
    resized = resize_frame_to_fit(frame, target_width, target_height)
    success, encoded = cv2.imencode(
        ".jpg",
        resized,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not success:
        return None
    return encoded.tobytes()


def normalize_role(role):
    normalized = str(role or "").strip().lower()
    if normalized in {"signer", "receiver"}:
        return normalized
    return "signer"


def normalize_room_code(room_code):
    cleaned = str(room_code or "").strip().upper()
    return cleaned or None


@dataclass
class SnapshotMessage:
    direction: str
    text: str
    time: str
    source: str


class FrameStore:
    def __init__(self):
        self._condition = threading.Condition()
        self._frame = None
        self._sequence = 0

    def set_frame(self, frame_bytes):
        with self._condition:
            self._frame = frame_bytes
            self._sequence += 1
            self._condition.notify_all()

    def clear(self):
        with self._condition:
            self._frame = None
            self._sequence += 1
            self._condition.notify_all()

    def wait_for_frame(self, last_sequence, timeout=1.0):
        with self._condition:
            if self._sequence <= last_sequence:
                self._condition.wait(timeout=timeout)
            return self._sequence, self._frame

    def snapshot(self):
        with self._condition:
            return self._sequence, self._frame


class DualTalkSession:
    def __init__(self, loop, server_url, ui_session_id=None, crypto=None):
        self.loop = loop
        self.server_url = server_url
        self.ui_session_id = str(ui_session_id or uuid.uuid4().hex)
        self.crypto = crypto
        self.outgoing_queue = asyncio.Queue(maxsize=8)
        self.stop_event = asyncio.Event()
        self.remote_cleanup_task = None
        self.communication_task = None
        self._push_scheduled = False
        self._thread_lock = threading.Lock()
        self._last_local_frame_sent_at = 0.0
        self.local_frames = FrameStore()
        self.remote_frames = FrameStore()
        self.tts = TTS(
            rate=int(config_get("tts.rate", 150)),
            volume=float(config_get("tts.volume", 1.0)),
            enabled=False,
        )
        self.messages = []
        self.state = {
            "room_code": None,
            "pending_room_code": None,
            "role": "signer",
            "server_connected": False,
            "camera_on": True,
            "speech_on": False,
            "ai_on": True,
            "local_stream_available": False,
            "remote_stream_available": False,
            "remote_connected": False,
            "remote_role": "receiver",
            "translation": {
                "gesture": None,
                "intent": None,
                "sentence": None,
                "placeholder": "Join a room to start signing.",
            },
            "last_error": None,
            "remote_status": {
                "speech_enabled": False,
                "camera_enabled": False,
                "ai_enabled": False,
            },
        }
        self.remote_frame_last_at = None
        self.worker = CameraPipelineWorker(self)

    async def start(self):
        self.worker.start()
        self.communication_task = asyncio.create_task(self.communication_loop())
        self.remote_cleanup_task = asyncio.create_task(self.remote_frame_watchdog_loop())
        self.request_state_push()

    async def stop(self):
        self.stop_event.set()
        self.worker.stop()
        if self.communication_task is not None:
            self.communication_task.cancel()
            try:
                await self.communication_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        if self.remote_cleanup_task is not None:
            self.remote_cleanup_task.cancel()
            try:
                await self.remote_cleanup_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self.tts.shutdown()

    async def communication_loop(self):
        while not self.stop_event.is_set():
            try:
                async with websockets.connect(
                    self.server_url,
                    compression=None,
                    ping_interval=20,
                    ping_timeout=20,
                    max_queue=2,
                ) as websocket:
                    self._set_server_connected(True)
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "worker_register",
                                "session_id": self.ui_session_id,
                            }
                        )
                    )
                    await self._send_resume_state(websocket)
                    self.request_state_push()

                    receiver_task = asyncio.create_task(
                        self.receive_server_messages(websocket)
                    )
                    sender_task = asyncio.create_task(
                        self.flush_outgoing_messages(websocket)
                    )

                    done, pending = await asyncio.wait(
                        {receiver_task, sender_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    for task in done:
                        if task.cancelled():
                            continue
                        exception = task.exception()
                        if exception is not None:
                            raise exception
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._set_server_connected(False)
                self.push_toast(
                    f"Communication server unavailable: {exc}",
                    level="error",
                )
                await asyncio.sleep(
                    float(config_get("communication.reconnect_delay_seconds", 2.0))
                )
            finally:
                self._set_server_connected(False)

    async def _send_resume_state(self, websocket):
        room_code, role, camera_on, speech_on, ai_on = self.get_connection_snapshot()
        resume_room_code = self.get_resume_room_code()
        if resume_room_code:
            await websocket.send(
                json.dumps(
                    {
                        "type": "join_room",
                        "room_code": resume_room_code,
                        "role": role,
                    }
                )
            )
        await websocket.send(
            json.dumps(
                build_state_update_message(
                    speech_on,
                    camera_enabled=camera_on,
                    ai_enabled=ai_on,
                    role=role,
                )
            )
        )

    async def flush_outgoing_messages(self, websocket):
        while not self.stop_event.is_set():
            payload = await self.outgoing_queue.get()
            serialized = payload
            if not isinstance(payload, (bytes, str)):
                serialized = json.dumps(payload)
            payload_type = None
            if isinstance(payload, dict):
                payload_type = payload.get("type")
            if self.crypto and isinstance(serialized, str) and payload_type not in {
                "create_room",
                "join_room",
                "leave_room",
                "worker_register",
                "snapshot",
                "toast",
                "local_frame",
                "remote_frame",
            }:
                serialized = self.crypto.encrypt(serialized)
            await websocket.send(serialized)

    async def receive_server_messages(self, websocket):
        async for raw_message in websocket:
            decoded = decode_message(raw_message, self.crypto)
            payload = parse_json_message(decoded)
            if isinstance(payload, dict):
                await self.handle_server_payload(payload)
                continue

            text = str(decoded or "").strip()
            if not text:
                continue
            self.add_message("received", text, source="chat")
            self._maybe_speak(text)

    async def handle_server_payload(self, payload):
        message_type = str(payload.get("type", "")).strip().lower()
        if message_type == "ui_action":
            await self.handle_ui_action(payload)
            return

        if message_type == "room_created":
            room_code = normalize_room_code(payload.get("room_code"))
            self._apply_room_join(room_code, participants=payload.get("participants"))
            self.add_message(
                "system",
                f"Room {room_code} created. Share the code to invite someone.",
                source="system",
            )
            self.push_toast(f"Room created: {room_code}")
            return

        if message_type == "room_joined":
            room_code = normalize_room_code(payload.get("room_code"))
            self._apply_room_join(room_code, participants=payload.get("participants"))
            self.add_message(
                "system",
                f"Joined room {room_code} as {self.get_role_label()}.",
                source="system",
            )
            self.push_toast(f"Joined room {room_code}")
            return

        if message_type == "room_left":
            self._clear_room_state(reset_messages=True)
            self.push_toast("Left room")
            return

        if message_type == "room_error":
            self._clear_room_state(reset_messages=True)
            message = str(payload.get("message", "Room unavailable.")).strip()
            self.push_toast(message, level="error")
            return

        if message_type == "peer_joined":
            participant = payload.get("participant") or {}
            self._update_remote_participant(participant, connected=True)
            self.add_message("system", "Another user joined the room.", source="system")
            self.request_state_push()
            return

        if message_type == "peer_left":
            self._set_remote_connected(False)
            self.add_message("system", "The other user left the room.", source="system")
            self.push_toast("The other user disconnected.")
            return

        if message_type == "video_frame":
            encoded = str(payload.get("image", "")).strip()
            if not encoded:
                return
            previous_available = False
            with self._thread_lock:
                previous_available = bool(self.state["remote_stream_available"])
                self.state["remote_stream_available"] = True
            self.remote_frame_last_at = now_timestamp()
            try:
                remote_jpeg = base64.b64decode(encoded)
                self.remote_frames.set_frame(remote_jpeg)
            except Exception:
                pass
            self.queue_payload_from_thread(
                self._build_ui_payload(
                    {
                        "type": "remote_frame",
                        "image": encoded,
                        "role": payload.get("role"),
                    }
                )
            )
            if not previous_available:
                self.request_state_push()
            return

        state_update = parse_state_update_message(payload)
        if state_update is not None:
            self._apply_remote_state_update(state_update)
            return

        if message_type == "delivery_ack":
            return

        if message_type == "communication":
            message_id = str(payload.get("message_id", "")).strip()
            text = str(
                payload.get("sentence")
                or payload.get("text")
                or payload.get("gesture")
                or ""
            ).strip()
            if not text:
                return
            self.add_message("received", text, source="translation")
            self._maybe_speak(text)
            if message_id:
                await self.queue_payload(build_delivery_ack_message(message_id))
            return

    async def queue_payload(self, payload):
        try:
            await self.outgoing_queue.put(payload)
        except asyncio.CancelledError:
            raise

    def queue_payload_from_thread(self, payload):
        if self.loop.is_closed():
            return

        def _enqueue():
            payload_type = payload.get("type") if isinstance(payload, dict) else None
            if payload_type == "video_frame" and self.outgoing_queue.full():
                return
            try:
                self.outgoing_queue.put_nowait(payload)
            except asyncio.QueueFull:
                return

        try:
            self.loop.call_soon_threadsafe(_enqueue)
        except RuntimeError:
            pass

    def _build_ui_payload(self, payload):
        enriched_payload = dict(payload)
        enriched_payload["session_id"] = self.ui_session_id
        return enriched_payload

    async def handle_ui_socket(self, websocket):
        raise RuntimeError(
            "Browser UI now connects directly to the main communication server."
        )

    async def handle_ui_action(self, payload):
        action = str(payload.get("action", "")).strip().lower()
        if action == "create_room":
            await self.create_room(role=payload.get("role"))
            return
        if action == "join_room":
            await self.join_room(payload.get("room_code"), role=payload.get("role"))
            return
        if action == "leave_room":
            await self.leave_room()
            return
        if action == "send_chat":
            await self.send_chat_message(payload.get("text"))
            return
        if action == "toggle_camera":
            await self.set_camera_enabled(not self.is_camera_enabled())
            return
        if action == "toggle_speech":
            await self.set_speech_enabled(not self.is_speech_enabled())
            return
        if action == "toggle_ai":
            await self.set_ai_enabled(not self.is_ai_enabled())
            return

    async def create_room(self, role=None):
        await self.leave_room(notify_remote=False)
        self.set_role(role or self.get_role())
        await self.queue_payload(
            {
                "type": "create_room",
                "role": self.get_role(),
            }
        )
        self.push_toast("Creating room...")

    async def join_room(self, room_code, role=None):
        normalized = normalize_room_code(room_code)
        if not normalized:
            self.push_toast("Enter a room code first.", level="error")
            return
        await self.leave_room(notify_remote=False)
        self.set_role(role or self.get_role())
        with self._thread_lock:
            self.state["pending_room_code"] = normalized
        await self.queue_payload(
            {
                "type": "join_room",
                "room_code": normalized,
                "role": self.get_role(),
            }
        )
        self.push_toast(f"Joining {normalized}...")
        self.request_state_push()

    async def leave_room(self, notify_remote=True):
        room_code = self.get_room_code()
        if room_code and notify_remote:
            await self.queue_payload({"type": "leave_room"})
        self._clear_room_state(reset_messages=True)
        self.request_state_push()

    async def send_chat_message(self, text):
        cleaned = str(text or "").strip()
        if not cleaned:
            return
        if not self.get_room_code():
            self.push_toast("Join a room before sending messages.", level="error")
            return

        payload = build_communication_message(
            cleaned,
            self.is_speech_enabled(),
            gesture="CHAT",
            intent="manual",
            message_id=uuid.uuid4().hex,
            sample_id="web-chat",
        )
        await self.queue_payload(payload)
        self.add_message("sent", cleaned, source="chat")

    async def set_camera_enabled(self, enabled):
        enabled = bool(enabled)
        with self._thread_lock:
            self.state["camera_on"] = enabled
            if not enabled:
                self.state["local_stream_available"] = False
        if not enabled:
            self.local_frames.clear()
            self.worker.reset_translation_state()
        await self.broadcast_local_state_update()
        self.push_toast("Camera on" if enabled else "Camera off")
        self.request_state_push()

    async def set_speech_enabled(self, enabled):
        enabled = bool(enabled)
        with self._thread_lock:
            self.state["speech_on"] = enabled
        if enabled:
            self.tts.enable()
        else:
            self.tts.disable()
        await self.broadcast_local_state_update()
        self.push_toast("Speech on" if enabled else "Speech off")
        self.request_state_push()

    async def set_ai_enabled(self, enabled):
        enabled = bool(enabled)
        with self._thread_lock:
            self.state["ai_on"] = enabled
        if not enabled:
            self.worker.reset_translation_state()
        await self.broadcast_local_state_update()
        self.push_toast("AI on" if enabled else "AI off")
        self.request_state_push()

    async def broadcast_local_state_update(self):
        room_code, role, camera_on, speech_on, ai_on = self.get_connection_snapshot()
        if not room_code:
            self.request_state_push()
            return
        await self.queue_payload(
            build_state_update_message(
                speech_on,
                camera_enabled=camera_on,
                ai_enabled=ai_on,
                role=role,
            )
        )
        self.request_state_push()

    def send_translation_message(self, sentence, *, gesture, intent):
        if not sentence or not self.get_room_code():
            return

        payload = build_communication_message(
            sentence,
            self.is_speech_enabled(),
            gesture=gesture,
            intent=intent or "",
            message_id=uuid.uuid4().hex,
            sample_id="web-gesture",
        )
        self.queue_payload_from_thread(payload)
        self.add_message("sent", sentence, source="translation")

    def send_video_frame(self, jpeg_bytes):
        if not jpeg_bytes or not self.get_room_code():
            return
        payload = {
            "type": "video_frame",
            "image": base64.b64encode(jpeg_bytes).decode("ascii"),
            "role": self.get_role(),
        }
        self.queue_payload_from_thread(payload)

    def update_local_frame(self, jpeg_bytes):
        if jpeg_bytes is None:
            was_available = False
            with self._thread_lock:
                was_available = bool(self.state["local_stream_available"])
                self.state["local_stream_available"] = False
            self.local_frames.clear()
            if was_available:
                self.request_state_push()
            return

        self.local_frames.set_frame(jpeg_bytes)
        became_available = False
        with self._thread_lock:
            if not self.state["local_stream_available"]:
                self.state["local_stream_available"] = True
                became_available = True
        if became_available:
            self.request_state_push()

        current_time = now_timestamp()
        if (
            (current_time - self._last_local_frame_sent_at)
            >= DEFAULT_VIDEO_FRAME_INTERVAL_SECONDS
        ):
            self._last_local_frame_sent_at = current_time
            self.queue_payload_from_thread(
                self._build_ui_payload(
                    {
                        "type": "local_frame",
                        "image": base64.b64encode(jpeg_bytes).decode("ascii"),
                    }
                )
            )

    def update_translation(self, gesture=None, intent=None, sentence=None, placeholder=None):
        with self._thread_lock:
            translation = self.state["translation"]
            translation["gesture"] = gesture
            translation["intent"] = intent
            translation["sentence"] = sentence
            translation["placeholder"] = placeholder
        self.request_state_push()

    def clear_remote_video(self):
        was_available = False
        with self._thread_lock:
            was_available = bool(self.state["remote_stream_available"])
            self.state["remote_stream_available"] = False
        self.remote_frames.clear()
        if was_available:
            self.request_state_push()

    async def remote_frame_watchdog_loop(self):
        while not self.stop_event.is_set():
            await asyncio.sleep(0.5)
            if self.remote_frame_last_at is None:
                continue
            if (
                now_timestamp() - self.remote_frame_last_at
                <= DEFAULT_REMOTE_FRAME_TIMEOUT_SECONDS
            ):
                continue
            self.remote_frame_last_at = None
            self.clear_remote_video()

    def add_message(self, direction, text, *, source):
        cleaned = str(text or "").strip()
        if not cleaned:
            return
        with self._thread_lock:
            self.messages.append(
                SnapshotMessage(
                    direction=direction,
                    text=cleaned,
                    time=now_time_label(),
                    source=source,
                )
            )
            if len(self.messages) > CHAT_HISTORY_LIMIT:
                self.messages = self.messages[-CHAT_HISTORY_LIMIT:]
        self.request_state_push()

    def push_toast(self, message, *, level="info"):
        payload = {
            "type": "toast",
            "message": str(message),
            "level": str(level),
        }
        if self.loop.is_closed():
            return

        async def _broadcast():
            await self.broadcast_json(payload)

        try:
            self.loop.call_soon_threadsafe(lambda: asyncio.create_task(_broadcast()))
        except RuntimeError:
            pass

    def request_state_push(self):
        if self.loop.is_closed():
            return

        def _schedule():
            if self._push_scheduled:
                return
            self._push_scheduled = True
            asyncio.create_task(self._flush_state())

        try:
            self.loop.call_soon_threadsafe(_schedule)
        except RuntimeError:
            pass

    async def _flush_state(self):
        self._push_scheduled = False
        await self.broadcast_json(
            {
                "type": "snapshot",
                "state": self.build_snapshot(),
            }
        )

    async def send_snapshot(self, websocket):
        await websocket.send(
            json.dumps(
                {
                    "type": "snapshot",
                    "state": self.build_snapshot(),
                }
            )
        )

    async def broadcast_json(self, payload):
        await self.queue_payload(payload)

    def build_snapshot(self):
        with self._thread_lock:
            translation = dict(self.state["translation"])
            remote_status = dict(self.state["remote_status"])
            messages = [
                {
                    "direction": message.direction,
                    "text": message.text,
                    "time": message.time,
                    "source": message.source,
                }
                for message in self.messages
            ]
            room_code = self.state["room_code"]
            pending_room_code = self.state["pending_room_code"]
            role = self.state["role"]
            server_connected = self.state["server_connected"]
            camera_on = self.state["camera_on"]
            speech_on = self.state["speech_on"]
            ai_on = self.state["ai_on"]
            local_stream_available = self.state["local_stream_available"]
            remote_stream_available = self.state["remote_stream_available"]
            remote_connected = self.state["remote_connected"]
            remote_role = self.state["remote_role"]

        status_text = "Not connected"
        if server_connected and room_code:
            status_text = f"Connected · {room_code}"
            status_text = f"Connected - {room_code}"
        elif pending_room_code:
            status_text = f"Joining - {pending_room_code}"
        elif server_connected:
            status_text = "Connected"
        elif room_code:
            status_text = f"Disconnected · {room_code}"

            status_text = f"Disconnected - {room_code}"

        return {
            "stage": "room" if room_code or pending_room_code else "lobby",
            "status": {
                "connected": bool(server_connected and room_code),
                "server_connected": bool(server_connected),
                "text": status_text,
            },
            "room_code": room_code or pending_room_code,
            "role": role,
            "camera_on": camera_on,
            "speech_on": speech_on,
            "ai_on": ai_on,
            "local_stream_available": local_stream_available,
            "remote_stream_available": remote_stream_available,
            "remote_connected": remote_connected,
            "remote_role": remote_role,
            "remote_status": remote_status,
            "translation": translation,
            "messages": messages,
        }

    def _apply_room_join(self, room_code, *, participants):
        normalized = normalize_room_code(room_code)
        with self._thread_lock:
            self.state["room_code"] = normalized
            self.state["pending_room_code"] = None
        self._apply_participants(participants)
        self.request_state_push()

    def _apply_participants(self, participants):
        remote_connected = False
        remote_role = "receiver" if self.get_role() == "signer" else "signer"
        if isinstance(participants, list):
            for participant in participants:
                if not isinstance(participant, dict):
                    continue
                role = normalize_role(participant.get("role"))
                if role == self.get_role():
                    continue
                remote_connected = True
                remote_role = role
                break

        with self._thread_lock:
            self.state["remote_connected"] = remote_connected
            self.state["remote_role"] = remote_role
        if not remote_connected:
            self.clear_remote_video()

    def _update_remote_participant(self, participant, *, connected):
        remote_role = normalize_role(participant.get("role"))
        with self._thread_lock:
            self.state["remote_connected"] = bool(connected)
            self.state["remote_role"] = remote_role
        if not connected:
            self.clear_remote_video()

    def _apply_remote_state_update(self, state_update):
        with self._thread_lock:
            self.state["remote_status"] = {
                "speech_enabled": bool(state_update.get("speech_enabled", False)),
                "camera_enabled": bool(state_update.get("camera_enabled", False)),
                "ai_enabled": bool(state_update.get("ai_enabled", False)),
            }
            remote_role = state_update.get("role")
            if remote_role:
                self.state["remote_role"] = normalize_role(remote_role)
            if not self.state["remote_status"]["camera_enabled"]:
                self.state["remote_stream_available"] = False
        if not self.state["remote_status"]["camera_enabled"]:
            self.clear_remote_video()
        self.request_state_push()

    def _set_remote_connected(self, connected):
        with self._thread_lock:
            self.state["remote_connected"] = bool(connected)
            if not connected:
                self.state["remote_status"] = {
                    "speech_enabled": False,
                    "camera_enabled": False,
                    "ai_enabled": False,
                }
                self.state["remote_stream_available"] = False
        self.clear_remote_video()

    def _clear_room_state(self, *, reset_messages):
        next_remote_role = "receiver" if self.get_role() == "signer" else "signer"
        with self._thread_lock:
            self.state["room_code"] = None
            self.state["pending_room_code"] = None
            self.state["remote_connected"] = False
            self.state["remote_stream_available"] = False
            self.state["remote_role"] = next_remote_role
            self.state["remote_status"] = {
                "speech_enabled": False,
                "camera_enabled": False,
                "ai_enabled": False,
            }
            if reset_messages:
                self.messages = []
        self.clear_remote_video()
        self.worker.reset_translation_state()

    def _set_server_connected(self, connected):
        with self._thread_lock:
            self.state["server_connected"] = bool(connected)
        self.request_state_push()

    def _maybe_speak(self, text):
        if self.is_speech_enabled():
            self.tts.speak(text)

    def set_role(self, role):
        normalized = normalize_role(role)
        with self._thread_lock:
            self.state["role"] = normalized
            self.state["remote_role"] = "receiver" if normalized == "signer" else "signer"
        self.request_state_push()

    def get_role(self):
        with self._thread_lock:
            return self.state["role"]

    def get_role_label(self):
        return "Signer" if self.get_role() == "signer" else "Receiver"

    def get_room_code(self):
        with self._thread_lock:
            return self.state["room_code"]

    def get_resume_room_code(self):
        with self._thread_lock:
            return self.state["room_code"] or self.state["pending_room_code"]

    def is_camera_enabled(self):
        with self._thread_lock:
            return bool(self.state["camera_on"])

    def is_speech_enabled(self):
        with self._thread_lock:
            return bool(self.state["speech_on"])

    def is_ai_enabled(self):
        with self._thread_lock:
            return bool(self.state["ai_on"])

    def get_connection_snapshot(self):
        with self._thread_lock:
            return (
                self.state["room_code"],
                self.state["role"],
                bool(self.state["camera_on"]),
                bool(self.state["speech_on"]),
                bool(self.state["ai_on"]),
            )


class CameraPipelineWorker:
    def __init__(self, session):
        self.session = session
        self.stop_flag = threading.Event()
        self.thread = None
        self._builder = SmartSentenceBuilder(
            timeout_seconds=float(config_get("sentence_builder.timeout_seconds", 2.0))
        )
        self._thread_lock = threading.Lock()
        self._reset_translation_state = False

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self.run, name="DualTalkWebCamera", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def reset_translation_state(self):
        with self._thread_lock:
            self._reset_translation_state = True

    def _consume_reset(self):
        with self._thread_lock:
            should_reset = self._reset_translation_state
            self._reset_translation_state = False
            return should_reset

    def run(self):
        model = load_model()
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        capture = None
        last_camera_retry_at = 0.0
        last_video_frame_at = 0.0
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
        hand_loss_started_at = None

        try:
            with mp_hands.Hands(
                max_num_hands=int(config_get("detection.max_num_hands", 2)),
                min_detection_confidence=float(
                    config_get("detection.detection_confidence", 0.7)
                ),
                min_tracking_confidence=float(
                    config_get("detection.tracking_confidence", 0.7)
                ),
            ) as hands:
                while not self.stop_flag.is_set():
                    room_code, role, camera_on, speech_on, ai_on = (
                        self.session.get_connection_snapshot()
                    )
                    visible_room_code = self.session.get_resume_room_code()
                    is_signer = role == "signer"

                    if self._consume_reset():
                        self._builder.clear()
                        last_processed_gesture = None
                        last_sent_signature = None
                        hand_loss_started_at = None

                    if not camera_on:
                        if capture is not None:
                            capture.release()
                            capture = None
                        self.session.update_local_frame(None)
                        self._publish_placeholder(role, ai_on, visible_room_code)
                        time.sleep(DEFAULT_CAMERA_IDLE_SLEEP_SECONDS)
                        continue

                    if capture is None or not capture.isOpened():
                        now = now_timestamp()
                        if (
                            capture is None
                            and (now - last_camera_retry_at) >= DEFAULT_CAMERA_RETRY_DELAY_SECONDS
                        ):
                            last_camera_retry_at = now
                            capture = open_camera(int(config_get("camera.index", 0)))
                        if capture is None:
                            self.session.update_local_frame(None)
                            self.session.update_translation(
                                gesture=None,
                                intent=None,
                                sentence=None,
                                placeholder="Waiting for camera...",
                            )
                            time.sleep(DEFAULT_CAMERA_IDLE_SLEEP_SECONDS)
                            continue

                    success, frame = capture.read()
                    if not success:
                        capture.release()
                        capture = None
                        self.session.update_local_frame(None)
                        self.session.update_translation(
                            gesture=None,
                            intent=None,
                            sentence=None,
                            placeholder="Camera unavailable.",
                        )
                        time.sleep(DEFAULT_CAMERA_IDLE_SLEEP_SECONDS)
                        continue

                    frame_started_at = time.perf_counter()
                    frame = cv2.flip(frame, 1)
                    render_frame = frame.copy()

                    detected_gesture = None
                    detected_zone = None
                    detected_confidence = 0.0
                    display_gesture = "READY"

                    if is_signer and ai_on:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = hands.process(rgb_frame)
                        if result.multi_hand_landmarks:
                            for hand_index, hand_landmarks in enumerate(
                                result.multi_hand_landmarks
                            ):
                                if hand_landmarks is None or len(hand_landmarks.landmark) < 21:
                                    continue

                                mp_draw.draw_landmarks(
                                    render_frame,
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
                                candidate_display = get_display_gesture(
                                    hand_gesture,
                                    confidence,
                                )
                                if (
                                    candidate_display != "UNSURE"
                                    and hand_gesture != "UNKNOWN"
                                    and confidence >= detected_confidence
                                ):
                                    detected_gesture = hand_gesture
                                    detected_zone = hand_zone
                                    detected_confidence = confidence
                                    display_gesture = candidate_display
                                elif detected_gesture is None:
                                    detected_gesture = candidate_display
                                    detected_zone = hand_zone
                                    detected_confidence = confidence
                                    display_gesture = candidate_display

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

                    if not is_signer:
                        self.session.update_translation(
                            gesture=None,
                            intent=None,
                            sentence=None,
                            placeholder="Receiver mode - waiting for incoming messages.",
                        )
                    elif not ai_on:
                        self.session.update_translation(
                            gesture=None,
                            intent=None,
                            sentence=None,
                            placeholder="AI recognition paused.",
                        )
                        self._builder.clear()
                        last_processed_gesture = None
                        last_sent_signature = None
                    elif detected_gesture is None:
                        if hand_loss_started_at is None:
                            hand_loss_started_at = frame_started_at
                        if (
                            hand_loss_started_at is not None
                            and (frame_started_at - hand_loss_started_at)
                            >= HAND_RESET_GRACE_SECONDS
                        ):
                            self._builder.clear()
                            last_processed_gesture = None
                            last_sent_signature = None
                        self.session.update_translation(
                            gesture="READY",
                            intent=None,
                            sentence=None,
                            placeholder="Gesture recognition active - start signing...",
                        )
                    else:
                        hand_loss_started_at = None
                        if detected_gesture in {"UNSURE", "UNKNOWN"}:
                            self.session.update_translation(
                                gesture=display_gesture,
                                intent=None,
                                sentence=None,
                                placeholder="Hold a stable gesture.",
                            )
                        elif detected_gesture != last_processed_gesture:
                            last_processed_gesture = detected_gesture
                            trace = self._builder.build_sentence_trace(
                                detected_gesture,
                                placement=detected_zone,
                            )
                            intent = trace["intent"] or EMPTY_VALUE
                            sentence = trace["sentence"]
                            self.session.update_translation(
                                gesture=display_gesture,
                                intent=intent if intent != EMPTY_VALUE else None,
                                sentence=sentence,
                                placeholder=None if sentence else "Interpreting gesture...",
                            )
                            if sentence:
                                signature = (
                                    detected_gesture,
                                    trace["zone"],
                                    intent,
                                    sentence,
                                )
                                if signature != last_sent_signature:
                                    last_sent_signature = signature
                                    self.session.send_translation_message(
                                        sentence,
                                        gesture=detected_gesture,
                                        intent=intent,
                                    )
                                    if speech_on:
                                        self.session.tts.speak(sentence)
                        else:
                            self.session.update_translation(
                                gesture=display_gesture,
                                intent=None,
                                sentence=None,
                                placeholder="Waiting for the next gesture...",
                            )

                    jpeg_bytes = encode_jpeg(render_frame)
                    self.session.update_local_frame(jpeg_bytes)
                    if (
                        room_code
                        and jpeg_bytes is not None
                        and (now_timestamp() - last_video_frame_at)
                        >= DEFAULT_VIDEO_FRAME_INTERVAL_SECONDS
                    ):
                        last_video_frame_at = now_timestamp()
                        self.session.send_video_frame(jpeg_bytes)

                    time.sleep(0.01)
        finally:
            if capture is not None:
                capture.release()
            self.session.update_local_frame(None)

    def _publish_placeholder(self, role, ai_on, room_code):
        if role != "signer":
            placeholder = "Receiver mode - camera is off."
        elif not room_code:
            placeholder = "Join a room to start signing."
        elif not ai_on:
            placeholder = "AI recognition paused."
        else:
            placeholder = "Camera off."
        self.session.update_translation(
            gesture=None,
            intent=None,
            sentence=None,
            placeholder=placeholder,
        )


class DualTalkHTTPRequestHandler(BaseHTTPRequestHandler):
    session = None
    api_port = DEFAULT_API_PORT
    server_url = None

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        if route in {"/", "/index.html"}:
            self.serve_file(INDEX_HTML_PATH, content_type="text/html; charset=utf-8")
            return
        if route == "/app-config.json":
            self.serve_json(
                {
                    "serverUrl": self.server_url,
                    "sessionId": self.session.ui_session_id,
                    "streamLocalUrl": "/stream/local",
                    "streamRemoteUrl": "/stream/remote",
                }
            )
            return
        if route == "/stream/local":
            self.serve_mjpeg_stream(self.session.local_frames)
            return
        if route == "/stream/remote":
            self.serve_mjpeg_stream(self.session.remote_frames)
            return
        if route.startswith("/static/"):
            relative_path = route.removeprefix("/static/")
            safe_path = (STATIC_DIR / relative_path).resolve()
            if STATIC_DIR.resolve() not in safe_path.parents and safe_path != STATIC_DIR.resolve():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not safe_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type, _ = mimetypes.guess_type(str(safe_path))
            self.serve_file(safe_path, content_type=content_type)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def serve_file(self, path, *, content_type=None):
        try:
            body = Path(path).read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_mjpeg_stream(self, frame_store):
        self.send_response(HTTPStatus.OK)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame",
        )
        self.end_headers()

        last_sequence = -1
        try:
            while True:
                sequence, frame_bytes = frame_store.wait_for_frame(last_sequence, timeout=1.0)
                if sequence == last_sequence:
                    continue
                last_sequence = sequence
                if frame_bytes is None:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame_bytes)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format, *args):
        return


class DualTalkWebApp:
    def __init__(self, host, port, api_host, api_port, server_url, crypto=None):
        self.host = host
        self.port = port
        self.api_host = api_host
        self.api_port = api_port
        self.server_url = server_url
        self.crypto = crypto
        self.http_server = None
        self.http_thread = None
        self.session = None

    async def run(self):
        loop = asyncio.get_running_loop()
        self.session = DualTalkSession(loop, self.server_url, crypto=self.crypto)
        await self.session.start()
        self.start_http_server()
        print(f"DualTalk UI available at http://{self.host}:{self.port}")
        print(f"Browser UI serving local streams; communication server: {self.server_url}")
        try:
            await asyncio.Future()
        finally:
            await self.stop()

    async def stop(self):
        # No local WebSocket server to close; communication handled by external server
        if self.http_server is not None:
            self.http_server.shutdown()
            self.http_server.server_close()
            self.http_server = None
        if self.http_thread is not None and self.http_thread.is_alive():
            self.http_thread.join(timeout=2.0)
        if self.session is not None:
            await self.session.stop()
            self.session = None

    def start_http_server(self):
        DualTalkHTTPRequestHandler.session = self.session
        DualTalkHTTPRequestHandler.api_port = self.api_port
        DualTalkHTTPRequestHandler.server_url = self.server_url
        self.http_server = ThreadingHTTPServer((self.host, self.port), DualTalkHTTPRequestHandler)
        self.http_thread = threading.Thread(
            target=self.http_server.serve_forever,
            name="DualTalkHTTP",
            daemon=True,
        )
        self.http_thread.start()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the DualTalk browser UI.")
    parser.add_argument(
        "--host",
        default=DEFAULT_HTTP_HOST,
        help="HTTP host to bind for the browser UI.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_HTTP_PORT,
        help="HTTP port to bind for the browser UI.",
    )
    parser.add_argument(
        "--api-host",
        default=DEFAULT_API_HOST,
        help="WebSocket API host to bind for the browser UI.",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=DEFAULT_API_PORT,
        help="WebSocket API port to bind for the browser UI.",
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER_URL,
        help="DualTalk communication server URL.",
    )
    parser.add_argument(
        "--encrypt",
        action="store_true",
        default=bool(config_get("communication.encrypted", False)),
        help="Encrypt browser bridge messages to the communication server.",
    )
    parser.add_argument(
        "--key",
        default=config_get("communication.encryption_key"),
        help="Shared Fernet key, optionally prefixed with DUALTALK_KEY=",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    crypto = resolve_crypto(
        args.encrypt,
        key_str=args.key,
        generate_if_missing=args.encrypt and not args.key,
    )
    app = DualTalkWebApp(
        host=args.host,
        port=args.port,
        api_host=args.api_host,
        api_port=args.api_port,
        server_url=args.server,
        crypto=crypto,
    )
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("DualTalk web UI stopped.")


if __name__ == "__main__":
    main()

import argparse
import asyncio
import json
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

try:
    import websockets
except ImportError as exc:
    raise SystemExit("Missing dependency: install it with `pip install websockets`.") from exc

from src.config import get as config_get
from src.security import resolve_crypto


DEFAULT_SERVER_URL = str(config_get("server.url", "ws://127.0.0.1:8765"))
RECONNECT_DELAY_SECONDS = float(
    config_get("communication.reconnect_delay_seconds", 2.0)
)
KEEPALIVE_INTERVAL_SECONDS = float(
    config_get("communication.keepalive_interval_seconds", 8.0)
)
KEEPALIVE_TIMEOUT_SECONDS = float(
    config_get("communication.keepalive_timeout_seconds", 8.0)
)
WINDOW_NAME = "DualTalk - Receiver"
FRAME_WIDTH = 860
FRAME_HEIGHT = 620
MAX_HISTORY = 12
HISTORY_LINE_HEIGHT = 22


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


def build_delivery_ack_message(message_id):
    return {
        "type": "delivery_ack",
        "message_id": str(message_id),
    }


def normalize_speech_enabled(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "on", "yes"}:
            return True
        if normalized in {"false", "0", "off", "no"}:
            return False
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def parse_protocol_message(raw_message):
    if raw_message is None:
        return None

    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8", errors="ignore")

    timestamp = time.time()
    try:
        message = json.loads(raw_message)
    except json.JSONDecodeError:
        text = str(raw_message).strip()
        if not text:
            return None
        return {
            "type": "communication",
            "text": text,
            "timestamp": timestamp,
            "speech_enabled": None,
        }

    if isinstance(message, str):
        text = message.strip()
        if not text:
            return None
        return {
            "type": "communication",
            "text": text,
            "timestamp": timestamp,
            "speech_enabled": None,
        }

    if not isinstance(message, dict):
        text = str(message).strip()
        if not text:
            return None
        return {
            "type": "communication",
            "text": text,
            "timestamp": timestamp,
            "speech_enabled": None,
        }

    message_type = str(message.get("type", "communication")).strip().lower()
    speech_enabled = normalize_speech_enabled(message.get("speech_enabled"))

    if message_type == "state_update":
        return {
            "type": "state_update",
            "speech_enabled": speech_enabled,
            "timestamp": timestamp,
        }
    if message_type == "delivery_ack":
        message_id = str(message.get("message_id", "")).strip()
        if not message_id:
            return None
        return {
            "type": "delivery_ack",
            "message_id": message_id,
            "timestamp": timestamp,
        }

    text_value = message.get("sentence")
    if text_value is None:
        text_value = message.get("text")
    if text_value is None:
        text_value = message.get("gesture")

    text = str(text_value).strip() if text_value is not None else ""
    if not text and speech_enabled is None:
        return None

    return {
        "type": "communication",
        "hand": "Remote",
        "gesture": text,
        "text": text,
        "confidence": 1.0,
        "timestamp": float(message.get("timestamp", timestamp)),
        "speech_enabled": speech_enabled,
        "message_id": str(message.get("message_id", "")).strip(),
    }


def parse_message(raw_message):
    parsed_message = parse_protocol_message(raw_message)
    if not parsed_message or not parsed_message.get("text"):
        return None

    return {
        "type": parsed_message.get("type", "communication"),
        "hand": str(parsed_message.get("hand", "Remote")),
        "gesture": str(parsed_message.get("gesture", parsed_message["text"])),
        "text": str(parsed_message["text"]).strip(),
        "confidence": float(parsed_message.get("confidence", 1.0)),
        "timestamp": float(parsed_message.get("timestamp", time.time())),
        "speech_enabled": parsed_message.get("speech_enabled"),
        "message_id": str(parsed_message.get("message_id", "")).strip(),
    }


def update_latest_message_state(state, parsed_message):
    if not parsed_message or not parsed_message.get("text"):
        return

    entry = {
        "text": str(parsed_message.get("text", "")).strip(),
        "timestamp": float(parsed_message.get("timestamp", time.time())),
    }
    state["latest"] = entry
    state["history"].append(entry)


def update_remote_speech_state(state, speech_enabled, *, force_log=False):
    if speech_enabled is None:
        return

    previous_value = state.get("remote_speech_enabled")
    state["remote_speech_enabled"] = speech_enabled
    if force_log or previous_value != speech_enabled:
        print("STATE RECEIVED")
        print("SPEECH: ON" if speech_enabled else "SPEECH: OFF")


def apply_protocol_message(state, parsed_message):
    if not parsed_message:
        return

    update_remote_speech_state(
        state,
        parsed_message.get("speech_enabled"),
        force_log=parsed_message.get("type") == "state_update",
    )
    if parsed_message.get("type") == "communication":
        update_latest_message_state(state, parsed_message)


def wrap_text(text, max_width, font, scale, thickness, max_lines=None):
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
            if max_lines is not None and len(lines) >= max_lines - 1:
                break

    remaining_words = words[len(" ".join(lines + [current_line]).split()):]
    if remaining_words:
        current_line = f"{current_line}..."
    lines.append(current_line)
    return lines[:max_lines] if max_lines is not None else lines


def fit_text_scale(text, max_width, font, base_scale, thickness):
    scale = base_scale
    while scale > 0.6:
        text_width, _ = cv2.getTextSize(text, font, scale, thickness)[0]
        if text_width <= max_width:
            return scale
        scale -= 0.1
    return scale


def draw_centered_text(frame, text, y_position, font, scale, color, thickness):
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    x_position = max((frame.shape[1] - text_size[0]) // 2, 30)
    cv2.putText(
        frame,
        text,
        (x_position, y_position),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


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


def draw_panel(frame, top_left, bottom_right, *, fill_color, border_color):
    cv2.rectangle(frame, top_left, bottom_right, fill_color, -1)
    cv2.rectangle(frame, top_left, bottom_right, border_color, 1)


def format_received_time(timestamp):
    if not timestamp:
        return "--:--:--"
    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")


def render_frame(
    latest_message,
    connection_status,
    *,
    history=None,
    remote_speech_enabled=None,
    title="DualTalk Receiver",
    message_label="Current message:",
    footer_text="Press q to exit",
    status_lines=None,
):
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    frame[:] = (20, 24, 30)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame,
        title,
        (24, 42),
        font,
        0.82,
        (180, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        connection_status,
        (24, 72),
        font,
        0.58,
        (120, 190, 120),
        2,
        cv2.LINE_AA,
    )
    speech_text = "Speech: ON" if remote_speech_enabled else "Speech: OFF"
    cv2.putText(
        frame,
        speech_text,
        (FRAME_WIDTH - 170, 42),
        font,
        0.58,
        (180, 220, 255),
        2,
        cv2.LINE_AA,
    )

    current_panel_top = 92
    current_panel_bottom = 300
    history_panel_top = 322
    history_panel_bottom = FRAME_HEIGHT - 56
    draw_panel(
        frame,
        (20, current_panel_top),
        (FRAME_WIDTH - 20, current_panel_bottom),
        fill_color=(27, 32, 40),
        border_color=(60, 72, 88),
    )
    draw_panel(
        frame,
        (20, history_panel_top),
        (FRAME_WIDTH - 20, history_panel_bottom),
        fill_color=(23, 28, 35),
        border_color=(52, 62, 76),
    )

    cv2.putText(
        frame,
        message_label,
        (36, current_panel_top + 28),
        font,
        0.66,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    timestamp_text = "Timestamp: --:--:--"
    if latest_message and latest_message.get("text"):
        text = str(latest_message.get("text", "")).strip()
        timestamp_text = f"Timestamp: {format_received_time(latest_message.get('timestamp'))}"
        text_lines = wrap_text(
            text,
            FRAME_WIDTH - 90,
            font,
            1.0,
            2,
            max_lines=4,
        )
        draw_text_lines(
            frame,
            text_lines,
            36,
            current_panel_top + 72,
            font,
            1.0,
            (0, 255, 145),
            2,
            34,
        )
    else:
        draw_text_lines(
            frame,
            ["Waiting for message..."],
            36,
            current_panel_top + 110,
            font,
            1.0,
            (0, 165, 255),
            2,
            34,
        )

    cv2.putText(
        frame,
        timestamp_text,
        (36, current_panel_bottom - 18),
        font,
        0.58,
        (190, 190, 190),
        2,
        cv2.LINE_AA,
    )

    history_title_y = history_panel_top + 28
    cv2.putText(
        frame,
        "History",
        (36, history_title_y),
        font,
        0.66,
        (180, 220, 255),
        2,
        cv2.LINE_AA,
    )

    history_y = history_title_y + 30
    if status_lines:
        for line in status_lines:
            cv2.putText(
                frame,
                line,
                (36, history_y),
                font,
                0.5,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            history_y += 22
        history_y += 8

    if history is not None:
        max_width = FRAME_WIDTH - 92
        for entry in list(history)[-8:][::-1]:
            timestamp = format_received_time(entry.get("timestamp"))
            line = f"{timestamp}  {str(entry.get('text', '')).strip()}"
            while cv2.getTextSize(line, font, 0.5, 1)[0][0] > max_width and len(line) > 3:
                line = f"{line[:-4]}..."
            cv2.putText(
                frame,
                line,
                (36, history_y),
                font,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            history_y += HISTORY_LINE_HEIGHT

    cv2.putText(
        frame,
        footer_text,
        (24, FRAME_HEIGHT - 20),
        font,
        0.56,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return frame


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


async def receive_gestures(server_url, state, status, tts=None, crypto=None):
    _ = tts
    status["text"] = f"Connecting to {server_url}"
    async with websockets.connect(
        server_url,
        compression=None,
        ping_interval=20,
        ping_timeout=20,
        max_queue=1,
    ) as websocket:
        status["text"] = "Connected"
        keepalive_task = asyncio.create_task(keep_websocket_alive(websocket, "Receiver"))
        try:
            async for raw_message in websocket:
                message = decode_message(raw_message, crypto)
                parsed_message = parse_protocol_message(message)
                if not parsed_message:
                    continue
                apply_protocol_message(state, parsed_message)
                if parsed_message.get("type") != "communication":
                    continue
                message_id = str(parsed_message.get("message_id", "")).strip()
                if not message_id:
                    continue
                try:
                    await websocket.send(
                        encode_message(
                            build_delivery_ack_message(message_id),
                            crypto=crypto,
                        )
                    )
                except Exception as exc:
                    raise OSError(f"Failed to send delivery ack: {exc}") from exc
        finally:
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            try:
                await websocket.close()
            except Exception:
                pass


async def keep_receiving(server_url, state, status, stop_event, tts=None, crypto=None):
    while not stop_event.is_set():
        try:
            await receive_gestures(server_url, state, status, tts=tts, crypto=crypto)
            status["text"] = "Disconnected. Reconnecting..."
        except (OSError, websockets.exceptions.ConnectionClosed) as exc:
            status["text"] = f"Disconnected: {exc}"
        except Exception as exc:
            status["text"] = f"Disconnected: {exc}"

        if stop_event.is_set():
            break

        await asyncio.sleep(RECONNECT_DELAY_SECONDS)


async def display_loop(server_url, tts=None, crypto=None):
    state = {
        "latest": None,
        "history": deque(maxlen=MAX_HISTORY),
        "remote_speech_enabled": False,
    }
    status = {"text": f"Connecting to {server_url}"}
    stop_event = asyncio.Event()
    receiver_task = asyncio.create_task(
        keep_receiving(server_url, state, status, stop_event, tts=tts, crypto=crypto)
    )

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)
        while not stop_event.is_set():
            frame = render_frame(
                state["latest"],
                status["text"],
                history=state["history"],
                remote_speech_enabled=state["remote_speech_enabled"],
            )
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

            await asyncio.sleep(0.01)
    finally:
        stop_event.set()
        receiver_task.cancel()
        try:
            await receiver_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Receive gesture text from a server.")
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER_URL,
        help="WebSocket server URL, for example ws://192.168.1.20:8765",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable automatic text-to-speech for new messages.",
    )
    parser.add_argument(
        "--encrypt",
        action="store_true",
        default=bool(config_get("communication.encrypted", False)),
        help="Decrypt incoming messages with the shared Fernet key.",
    )
    parser.add_argument(
        "--key",
        default=config_get("communication.encryption_key"),
        help="Shared Fernet key, optionally prefixed with DUALTALK_KEY=",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    crypto = resolve_crypto(args.encrypt, key_str=args.key, generate_if_missing=False)
    try:
        asyncio.run(display_loop(args.server, tts=None, crypto=crypto))
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("Receiver stopped.")


if __name__ == "__main__":
    main()

import argparse
import asyncio
import base64
import json
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import cv2
import numpy as np
try:
    import mediapipe as mp
    print("DEBUG: Successfully executed 'import mediapipe as mp'", flush=True)
    print(f"DEBUG: mp.__file__: {getattr(mp, '__file__', 'No __file__')}", flush=True)
    print(f"DEBUG: mp.__version__: {getattr(mp, '__version__', 'No __version__')}", flush=True)
    print(f"DEBUG: hasattr(mp, 'solutions'): {hasattr(mp, 'solutions')}", flush=True)
    print(f"DEBUG: dir(mp): {dir(mp)}", flush=True)
    print(f"DEBUG: sys.path: {sys.path}", flush=True)
    
    try:
        from mediapipe.python import solutions
        print("DEBUG: Successfully executed 'from mediapipe.python import solutions'", flush=True)
    except Exception as e:
        print(f"DEBUG: Failed to import mediapipe.python.solutions: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
except Exception as e:
    print(f"DEBUG: Failed to import mediapipe: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()

try:
    import websockets
except ImportError as exc:
    raise SystemExit("Missing dependency: install it with `pip install websockets`.") from exc

# Add project root to python path if not already there
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.inference.run_realtime import load_model, get_display_gesture
from src.communication.sender import (
    predict_gesture_for_hand,
    clear_stale_hand_state,
    get_hand_label_and_score
)
from src.config import get as config_get
from src.processing.conversation_context import ConversationContext
from src.processing.placement_engine import detect_zone
from src.processing.prompt_builder import NLPProcessor, PromptBuilder
from src.processing.smart_sentence_builder import SmartSentenceBuilder
from src.processing.adaptive_engine import AdaptiveEngine

# Global configurations
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
DEFAULT_ROOM_PREFIX = "DT"
DEFAULT_ROOM_DIGITS = 4
WEBSOCKET_PATH = "/ws"

# Initialize MediaPipe solutions and Model globally
global_model = load_model()

try:
    print("DEBUG: Attempting mp.solutions.hands", flush=True)
    mp_hands = mp.solutions.hands
    print("DEBUG: Successfully got mp.solutions.hands", flush=True)
except Exception as e:
    print(f"DEBUG: Failed on mp.solutions.hands: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
    raise e

mp_draw = mp.solutions.drawing_utils

# Create thread pool executor for processing frame images
executor = ThreadPoolExecutor(max_workers=4)

@dataclass
class ClientSession:
    websocket: object
    client_id: str
    room_code: str | None = None
    role: str = "signer"
    display_name: str | None = None
    
    # State flags
    camera_on: bool = True
    speech_on: bool = False
    ai_on: bool = True
    processing_frame: bool = False
    
    # Hand-tracking & gesture recognition states
    hands_processor: object = None
    sentence_builder: object = None
    prediction_buffers: dict = None
    stable_predictions: dict = None
    wave_buffers: dict = None
    no_x_buffers: dict = None
    no_y_buffers: dict = None
    no_latches: dict = None
    no_stable_counts: dict = None
    hand_last_seen_at: dict = None
    last_processed_gesture: str = None
    last_sent_signature: tuple = None
    hand_loss_started_at: float = None

    def __hash__(self):
        return hash(self.client_id)

    def __eq__(self, other):
        if not isinstance(other, ClientSession):
            return False
        return self.client_id == other.client_id

    def init_ml_states(self):
        self.sentence_builder = SmartSentenceBuilder(timeout_seconds=2.0)
        self.prediction_buffers = {}
        self.stable_predictions = {}
        self.wave_buffers = {}
        self.no_x_buffers = {}
        self.no_y_buffers = {}
        self.no_latches = {}
        self.no_stable_counts = {}
        self.hand_last_seen_at = {}
        self.last_processed_gesture = None
        self.last_sent_signature = None
        self.hand_loss_started_at = None
        if self.hands_processor is None:
            self.hands_processor = mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            
    def reset_ml_states(self):
        if self.sentence_builder:
            self.sentence_builder.clear()
        self.prediction_buffers = {}
        self.stable_predictions = {}
        self.wave_buffers = {}
        self.no_x_buffers = {}
        self.no_y_buffers = {}
        self.no_latches = {}
        self.no_stable_counts = {}
        self.hand_last_seen_at = {}
        self.last_processed_gesture = None
        self.last_sent_signature = None
        self.hand_loss_started_at = None
        
    def close(self):
        if self.hands_processor:
            try:
                self.hands_processor.close()
            except Exception:
                pass
            self.hands_processor = None


class RoomState:
    def __init__(self, room_code, profile_name=None):
        self.room_code = room_code
        self.members = set()  # set of ClientSessions
        self.messages = []    # list of message dictionaries
        self.translation = {
            "gesture": None,
            "intent": None,
            "sentence": None,
            "suggestions": [],
            "placeholder": "Gesture recognition active - start signing..."
        }
        self.context = ConversationContext(room_code=room_code)
        self.prompt_builder = PromptBuilder(
            profile_name=profile_name or config_get("gestures.profile", "hospital")
        )
        self.nlp_processor = NLPProcessor(self.prompt_builder)
        self.adaptive = AdaptiveEngine(profile_name or config_get("gestures.profile", "general"), room_code)


class ASGIWebSocketAdapter:
    def __init__(self, websocket):
        self._websocket = websocket

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await self._websocket.receive_text()
        except Exception as exc:
            if exc.__class__.__name__ == "WebSocketDisconnect":
                raise StopAsyncIteration from exc
            raise

    async def send(self, message):
        if isinstance(message, bytes):
            await self._websocket.send_bytes(message)
            return
        await self._websocket.send_text(message)

    async def close(self, code=1000, reason=""):
        await self._websocket.close(code=code, reason=reason)


def process_frame_sync(session, base64_image):
    try:
        image_bytes = base64.b64decode(base64_image)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None, None
    except Exception as e:
        print(f"Error decoding frame: {e}")
        return None, None
        
    gesture_detected = None
    intent_detected = None
    sentence_built = None
    display_gesture = "READY"
    
    annotated_frame = frame.copy()
    
    if session.ai_on and session.hands_processor:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = session.hands_processor.process(rgb_frame)
        
        detected_gesture = None
        detected_zone = None
        detected_confidence = 0.0
        
        if result.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(result.multi_hand_landmarks):
                if hand_landmarks is None or len(hand_landmarks.landmark) < 21:
                    continue
                    
                mp_draw.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
                
                hand_label, hand_score = get_hand_label_and_score(result, hand_index)
                if hand_score < 0.5:
                    continue
                    
                session.hand_last_seen_at[hand_label] = time.time()
                hand_zone = detect_zone(hand_landmarks.landmark)
                hand_gesture, confidence = predict_gesture_for_hand(
                    global_model,
                    hand_landmarks.landmark,
                    hand_label,
                    session.prediction_buffers,
                    session.stable_predictions,
                    session.wave_buffers,
                    session.no_x_buffers,
                    session.no_y_buffers,
                    session.no_latches,
                    session.no_stable_counts
                )
                
                candidate_display = get_display_gesture(hand_gesture, confidence)
                if candidate_display != "UNSURE" and hand_gesture != "UNKNOWN" and confidence >= detected_confidence:
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
                session.prediction_buffers,
                session.stable_predictions,
                session.wave_buffers,
                session.no_x_buffers,
                session.no_y_buffers,
                session.no_latches,
                session.no_stable_counts,
                session.hand_last_seen_at,
                time.time()
            )
            
        if detected_gesture is not None:
            session.hand_loss_started_at = None
            if detected_gesture not in {"UNSURE", "UNKNOWN"}:
                if detected_gesture != session.last_processed_gesture:
                    session.last_processed_gesture = detected_gesture
                    trace = session.sentence_builder.build_sentence_trace(
                        detected_gesture,
                        placement=detected_zone
                    )
                    intent_detected = trace["intent"] or "-"
                    sentence_built = trace["sentence"]
                    
                    if sentence_built:
                        signature = (detected_gesture, trace["zone"], intent_detected, sentence_built)
                        if signature != session.last_sent_signature:
                            session.last_sent_signature = signature
                        else:
                            sentence_built = None
        else:
            if session.hand_loss_started_at is None:
                session.hand_loss_started_at = time.time()
            if time.time() - session.hand_loss_started_at >= 1.5:
                session.sentence_builder.clear()
                session.last_processed_gesture = None
                session.last_sent_signature = None
                
        if detected_gesture:
            gesture_detected = display_gesture
            
    success, encoded_local = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
    local_base64 = base64.b64encode(encoded_local.tobytes()).decode("ascii") if success else None
    
    return local_base64, (gesture_detected, intent_detected, sentence_built)


class GestureServer:
    def __init__(self):
        self.clients: dict[object, ClientSession] = {}
        self.rooms: dict[str, RoomState] = {}
        self._client_sequence = 0

    async def register(self, websocket):
        self._client_sequence += 1
        session = ClientSession(
            websocket=websocket,
            client_id=f"client-{self._client_sequence}",
        )
        self.clients[websocket] = session
        print(f"Client connected. Total clients: {len(self.clients)}")
        return session

    async def unregister(self, websocket):
        session = self.clients.pop(websocket, None)
        if session is None:
            return
            
        await self._remove_from_room(session)
        session.close()
        print(f"Client disconnected. Total clients: {len(self.clients)}")

    async def handle_client(self, websocket, path=None):
        session = await self.register(websocket)
        try:
            # Send initial lobby state
            await self.send_lobby_snapshot(session)
            
            async for message in websocket:
                # First try processing as a control message to be backwards compatible with tests
                if await self._handle_control_message(session, message):
                    continue
                    
                payload = self._parse_json_message(message)
                if not isinstance(payload, dict):
                    continue
                    
                message_type = str(payload.get("type", "")).strip().lower()
                
                if message_type == "toggle_camera":
                    session.camera_on = not session.camera_on
                    if not session.camera_on:
                        session.reset_ml_states()
                    await self.broadcast_room_snapshot_by_session(session)
                elif message_type == "toggle_speech":
                    session.speech_on = not session.speech_on
                    await self.broadcast_room_snapshot_by_session(session)
                elif message_type == "toggle_ai":
                    session.ai_on = not session.ai_on
                    if not session.ai_on:
                        session.reset_ml_states()
                    await self.broadcast_room_snapshot_by_session(session)
                elif message_type == "send_chat":
                    await self._handle_send_chat(session, payload.get("text"))
                elif message_type == "video_frame":
                    asyncio.create_task(self.process_client_frame(session, payload.get("image")))
        except Exception as exc:
            print("Client error:", exc)
        finally:
            await self.unregister(websocket)

    async def _handle_control_message(self, session, raw_message):
        payload = self._parse_json_message(raw_message)
        if not isinstance(payload, dict):
            return False
            
        message_type = str(payload.get("type", "")).strip().lower()
        if message_type not in {"create_room", "join_room", "leave_room"}:
            return False
            
        if message_type == "create_room":
            await self._create_room(session, payload.get("role"))
        elif message_type == "join_room":
            await self._join_room(session, payload.get("room_code"), payload.get("role"))
        elif message_type == "leave_room":
            await self._leave_room(session)
        return True

    async def _create_room(self, session, role):
        await self._remove_from_room(session)
        
        room_code = self._generate_room_code()
        room = RoomState(
            room_code,
            profile_name=config_get("gestures.profile", "hospital"),
        )
        self.rooms[room_code] = room
        
        session.room_code = room_code
        session.role = self._normalize_role(role)
        if session.role == "signer":
            session.init_ml_states()
            
        room.members.add(session)
        
        await session.websocket.send(json.dumps({
            "type": "toast",
            "message": f"Room created: {room_code}",
            "level": "info"
        }))
        
        await self.broadcast_room_snapshot(room)

        # Original message response for tests/compatibility (sent last)
        await session.websocket.send(json.dumps({
            "type": "room_created",
            "room_code": room_code,
            "client_id": session.client_id,
            "role": session.role,
            "participants": self._list_room_participants(room_code)
        }))

    async def _join_room(self, session, room_code, role):
        room_code = self._normalize_room_code(room_code)
        if not room_code or room_code not in self.rooms:
            await session.websocket.send(json.dumps({
                "type": "room_error",
                "message": "Room not found.",
                "room_code": room_code
            }))
            return
            
        room = self.rooms[room_code]
        if len(room.members) >= 2:
            await session.websocket.send(json.dumps({
                "type": "room_error",
                "message": "Room is full.",
                "room_code": room_code
            }))
            return
            
        await self._remove_from_room(session)
        
        session.room_code = room_code
        session.role = self._normalize_role(role)
        if session.role == "signer":
            session.init_ml_states()
            
        room.members.add(session)
        
        await session.websocket.send(json.dumps({
            "type": "toast",
            "message": f"Joined room {room_code}",
            "level": "info"
        }))

        # Original message response for tests/compatibility (sent last)
        await session.websocket.send(json.dumps({
            "type": "room_joined",
            "room_code": room_code,
            "client_id": session.client_id,
            "role": session.role,
            "participants": self._list_room_participants(room_code)
        }))
        
        # Notify peers using original format
        await self._broadcast_room_event(
            room_code,
            {
                "type": "peer_joined",
                "room_code": room_code,
                "participant": self._participant_summary(session),
                "participants": self._list_room_participants(room_code),
            },
            exclude=session
        )
        
        # Also notify via toast
        for member in room.members:
            if member != session:
                await member.websocket.send(json.dumps({
                    "type": "toast",
                    "message": "Another user joined the room.",
                    "level": "info"
                }))
                
        await self.broadcast_room_snapshot(room)

    async def _leave_room(self, session):
        room_code = session.room_code
        await self._remove_from_room(session)
        
        await self.send_lobby_snapshot(session)
        await session.websocket.send(json.dumps({
            "type": "room_left",
            "client_id": session.client_id,
        }))

    async def _remove_from_room(self, session):
        room_code = session.room_code
        if not room_code:
            return
            
        room = self.rooms.get(room_code)
        if room:
            room.members.discard(session)
            
            # Notify remaining members via toast & snapshot
            for member in list(room.members):
                await member.websocket.send(json.dumps({
                    "type": "toast",
                    "message": "The other user left the room.",
                    "level": "info"
                }))
                await self.broadcast_room_snapshot(room)
                
            # Notify remaining members using original format (sent last)
            participant_summary = self._participant_summary(session)
            await self._broadcast_room_event(
                room_code,
                {
                    "type": "peer_left",
                    "room_code": room_code,
                    "participant": participant_summary,
                    "participants": self._list_room_participants(room_code),
                },
                exclude=session
            )
            
            if not room.members:
                room.context.clear()
                self.rooms.pop(room_code, None)
                
        session.room_code = None
        session.reset_ml_states()

    async def _forward_message(self, session, message):
        recipients = self._resolve_forward_targets(session)
        for client in recipients:
            await client.send(message)

    def _resolve_forward_targets(self, session):
        if session.room_code:
            room_members = self.rooms.get(session.room_code)
            members_set = room_members.members if room_members else set()
            return [member.websocket for member in members_set if member != session]
        return [
            client.websocket
            for client in self.clients.values()
            if client != session and client.room_code is None
        ]

    async def _broadcast_room_event(self, room_code, payload, *, exclude=None):
        room = self.rooms.get(room_code)
        if not room:
            return
        for member in list(room.members):
            if member == exclude:
                continue
            try:
                await member.websocket.send(json.dumps(payload))
            except Exception as exc:
                print(f"Error broadcasting event: {exc}")

    async def _handle_send_chat(self, session, text):
        room_code = session.room_code
        if not room_code:
            return
            
        room = self.rooms.get(room_code)
        if not room:
            return
            
        cleaned = str(text or "").strip()
        if not cleaned:
            return
            
        msg = {
            "sender_id": session.client_id,
            "text": cleaned,
            "time": datetime.now().strftime("%H:%M"),
            "source": "chat"
        }
        room.messages.append(msg)
        
        # Speak notification to other client if speech enabled
        for member in room.members:
            if member != session and member.speech_on:
                await member.websocket.send(json.dumps({
                    "type": "speak",
                    "text": cleaned
                }))
                
        await self.broadcast_room_snapshot(room)

    async def process_client_frame(self, session, image_base64):
        if not session.room_code or not image_base64:
            return
            
        if session.processing_frame:
            return
            
        session.processing_frame = True
        try:
            loop = asyncio.get_running_loop()
            local_base64, gesture_info = await loop.run_in_executor(
                executor,
                process_frame_sync,
                session,
                image_base64
            )
            
            if local_base64 is None:
                return
                
            # Send local annotated frame back to signer
            await session.websocket.send(json.dumps({
                "type": "local_frame",
                "image": local_base64
            }))
            
            # Send remote frame to receiver in same room
            room = self.rooms.get(session.room_code)
            if room:
                for member in room.members:
                    if member != session:
                        await member.websocket.send(json.dumps({
                            "type": "remote_frame",
                            "image": local_base64,
                            "role": session.role
                        }))
                        
            # Handle translation update if AI was processing and returned something
            if gesture_info:
                detected_gesture, intent_detected, sentence_built = gesture_info
                normalized_intent = (
                    intent_detected if intent_detected and intent_detected != "-" else None
                )
                natural_sentence = None
                if room and sentence_built:
                    natural_sentence = room.nlp_processor.process(
                        normalized_intent,
                        sentence_built,
                        room.context,
                    )
                    if not natural_sentence:
                        natural_sentence = sentence_built
                    room.context.add_entry(normalized_intent, natural_sentence)
                    room.adaptive.record(natural_sentence)
                    suggestions = room.adaptive.suggestions(
                        gesture=detected_gesture,
                        intent=normalized_intent,
                        sentence=natural_sentence,
                    )

                if room:
                    if sentence_built:
                        msg = {
                            "sender_id": session.client_id,
                            "text": natural_sentence,
                            "time": datetime.now().strftime("%H:%M"),
                            "source": "translation"
                        }
                        room.messages.append(msg)
                        
                        for member in room.members:
                            if member != session and member.speech_on:
                                await member.websocket.send(json.dumps({
                                    "type": "speak",
                                    "text": natural_sentence
                                }))
                                
                        room.translation = {
                            "gesture": detected_gesture,
                            "intent": normalized_intent,
                            "sentence": natural_sentence,
                            "suggestions": suggestions,
                            "placeholder": None
                        }
                        await self.broadcast_room_snapshot(room)
                    elif detected_gesture:
                        room.translation["gesture"] = detected_gesture
                        room.translation["intent"] = normalized_intent
                        room.translation["placeholder"] = None
                        await self.broadcast_room_snapshot(room)
        except Exception as e:
            print(f"Error processing frame: {e}")
        finally:
            session.processing_frame = False

    async def broadcast_room_snapshot_by_session(self, session):
        if not session.room_code:
            await self.send_lobby_snapshot(session)
            return
        room = self.rooms.get(session.room_code)
        if room:
            await self.broadcast_room_snapshot(room)

    async def broadcast_room_snapshot(self, room):
        for session in list(room.members):
            try:
                snapshot = self.build_snapshot_for_session(session, room)
                await session.websocket.send(json.dumps({
                    "type": "snapshot",
                    "state": snapshot
                }))
            except Exception as e:
                print(f"Error broadcasting snapshot: {e}")

    async def send_lobby_snapshot(self, session):
        snapshot = {
            "stage": "lobby",
            "status": {
                "connected": False,
                "server_connected": True,
                "text": "Connected"
            },
            "room_code": None,
            "role": session.role,
            "camera_on": session.camera_on,
            "speech_on": session.speech_on,
            "ai_on": session.ai_on,
            "local_stream_available": False,
            "remote_stream_available": False,
            "remote_connected": False,
            "remote_role": "receiver" if session.role == "signer" else "signer",
            "remote_status": {
                "speech_enabled": False,
                "camera_enabled": False,
                "ai_enabled": False
            },
            "translation": {
                "gesture": None,
                "intent": None,
                "sentence": None,
                "suggestions": [],
                "placeholder": "Join a room to start signing."
            },
            "messages": []
        }
        await session.websocket.send(json.dumps({
            "type": "snapshot",
            "state": snapshot
        }))

    def build_snapshot_for_session(self, session, room):
        peer = None
        for member in room.members:
            if member != session:
                peer = member
                break
                
        status_text = f"Connected - {room.room_code}"
        
        formatted_messages = []
        for msg in room.messages:
            direction = "system"
            if "sender_id" in msg:
                direction = "sent" if msg["sender_id"] == session.client_id else "received"
            formatted_messages.append({
                "direction": direction,
                "text": msg["text"],
                "time": msg["time"],
                "source": msg["source"]
            })
            
        return {
            "stage": "room",
            "status": {
                "connected": True,
                "server_connected": True,
                "text": status_text
            },
            "room_code": room.room_code,
            "role": session.role,
            "camera_on": session.camera_on,
            "speech_on": session.speech_on,
            "ai_on": session.ai_on,
            "local_stream_available": session.camera_on,
            "remote_stream_available": peer.camera_on if peer else False,
            "remote_connected": peer is not None,
            "remote_role": peer.role if peer else ("receiver" if session.role == "signer" else "signer"),
            "remote_status": {
                "speech_enabled": peer.speech_on if peer else False,
                "camera_enabled": peer.camera_on if peer else False,
                "ai_enabled": peer.ai_on if peer else False
            },
            "translation": room.translation,
            "messages": formatted_messages
        }

    def _list_room_participants(self, room_code):
        room = self.rooms.get(room_code)
        if not room:
            return []
        return [self._participant_summary(member) for member in list(room.members)]

    @staticmethod
    def _participant_summary(session):
        return {
            "client_id": session.client_id,
            "role": session.role,
            "display_name": session.display_name,
        }

    def _generate_room_code(self):
        while True:
            suffix = random.randint(10 ** (DEFAULT_ROOM_DIGITS - 1), (10**DEFAULT_ROOM_DIGITS) - 1)
            room_code = f"{DEFAULT_ROOM_PREFIX}-{suffix}"
            if room_code not in self.rooms:
                return room_code

    @staticmethod
    def _parse_json_message(message):
        if isinstance(message, bytes):
            try:
                message = message.decode("utf-8")
            except UnicodeDecodeError:
                return None
        if not isinstance(message, str):
            return None
        try:
            return json.loads(message)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _normalize_room_code(room_code):
        normalized = str(room_code or "").strip().upper()
        return normalized or None

    @staticmethod
    def _normalize_role(role):
        normalized = str(role or "").strip().lower()
        if normalized in {"signer", "receiver"}:
            return normalized
        return "signer"


def _origin_allowed(origin):
    if os.environ.get("ENV") != "production" or not origin:
        return True

    parsed = urlparse(origin)
    hostname = parsed.hostname
    if not hostname:
        return False

    allowed_origin = os.environ.get("FRONTEND_URL")
    if allowed_origin:
        allowed_host = urlparse(allowed_origin).hostname
        return hostname == allowed_host

    return hostname in ("localhost", "127.0.0.1") or hostname.endswith(".vercel.app") or hostname == "vercel.app"


def create_app():
    try:
        from fastapi import FastAPI, WebSocket, status
        from fastapi.responses import PlainTextResponse, Response
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install it with `pip install fastapi uvicorn`."
        ) from exc

    server = GestureServer()
    app = FastAPI(docs_url=None, redoc_url=None)

    @app.get("/", include_in_schema=False)
    async def root():
        return PlainTextResponse("ok")

    @app.head("/", include_in_schema=False)
    async def root_head():
        return Response(status_code=200)

    @app.get("/health", include_in_schema=False)
    async def health():
        return PlainTextResponse("ok")

    @app.head("/health", include_in_schema=False)
    async def health_head():
        return Response(status_code=200)

    @app.websocket(WEBSOCKET_PATH)
    async def websocket_endpoint(websocket: WebSocket):
        if not _origin_allowed(websocket.headers.get("origin")):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Forbidden")
            return

        await websocket.accept()
        await server.handle_client(ASGIWebSocketAdapter(websocket), WEBSOCKET_PATH)

    return app

def parse_args():
    parser = argparse.ArgumentParser(description="Run the DualTalk gesture text server.")
    parser.add_argument("--host", default=os.environ.get("HOST", DEFAULT_HOST), help="Host/IP to bind.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", DEFAULT_PORT)), help="Port to bind.")
    return parser.parse_args()


def main():
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install it with `pip install uvicorn`."
        ) from exc

    args = parse_args()
    port = int(os.environ.get("WS_PORT", args.port))
    app = create_app()
    print(f"DualTalk server listening on http://{args.host}:{port} with WebSocket endpoint {WEBSOCKET_PATH}")
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        ws_ping_interval=20.0,
        ws_ping_timeout=20.0,
    )


if __name__ == "__main__":
    main()

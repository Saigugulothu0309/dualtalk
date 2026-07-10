import argparse
import asyncio
import time

import cv2

try:
    import websockets
except ImportError as exc:
    raise SystemExit("Missing dependency: install it with `pip install websockets`.") from exc

from src.audio import SpeechToText, TTS
from src.communication.receiver import decode_message, parse_message, render_frame
from src.config import get as config_get
from src.processing.translator import SUPPORTED_LANGUAGES, Translator
from src.security import resolve_crypto


DEFAULT_SERVER_URL = str(config_get("server.url", "ws://127.0.0.1:8765"))
RECONNECT_DELAY_SECONDS = float(
    config_get("communication.reconnect_delay_seconds", 2.0)
)
WINDOW_NAME = "DualTalk - Normal User"


def resolve_language_settings(language_value: str | None, stt_override: str | None = None):
    translation_target = str(config_get("language.default", "en"))
    stt_language = str(config_get("stt.language", "en-US"))

    value = str(language_value or "").strip()
    if value:
        if value in SUPPORTED_LANGUAGES:
            translation_target = SUPPORTED_LANGUAGES[value]
        elif value in SUPPORTED_LANGUAGES.values():
            translation_target = value
        elif "-" in value:
            stt_language = value
            translation_target = value.split("-", 1)[0]
        else:
            translation_target = value

    if stt_override:
        stt_language = stt_override

    return translation_target, stt_language


def is_speech_available(stt: SpeechToText) -> bool:
    return not (stt.request_error_detected or stt.last_error is not None)


def build_status_lines(state, use_speech, stt, tts):
    if state["type_mode"]:
        prompt = state["typed_buffer"] or "Type your message and press Enter"
        listening_line = f"Type mode: {prompt}"
    elif use_speech and is_speech_available(stt):
        listening_line = "Listening for your speech..."
    else:
        listening_line = "Speech unavailable - press T to type"

    last_sent = state["last_sent"] or ""
    tts_state = "ON" if tts.enabled else "OFF"
    return [
        listening_line,
        f'Last sent: "{last_sent}"',
        f"TTS: {tts_state}",
    ]


def handle_type_keypress(key, state, outgoing_queue):
    if key == 13:
        text = state["typed_buffer"].strip()
        if text:
            outgoing_queue.put_nowait(text)
        state["typed_buffer"] = ""
        return

    if key == 8:
        state["typed_buffer"] = state["typed_buffer"][:-1]
        return

    if 32 <= key <= 126:
        state["typed_buffer"] += chr(key)


async def normal_user_loop(
    server_url: str,
    use_speech: bool = True,
    language_value: str | None = None,
    stt_language_override: str | None = None,
    tts_enabled: bool = True,
    crypto=None,
):
    translation_target, stt_language = resolve_language_settings(
        language_value,
        stt_language_override,
    )
    translator = Translator(translation_target)
    tts = TTS(
        rate=int(config_get("tts.rate", 150)),
        volume=float(config_get("tts.volume", 1.0)),
    )
    if not tts_enabled or not bool(config_get("tts.enabled", True)):
        tts.disable()

    stt = SpeechToText(
        language=stt_language,
        timeout=float(config_get("stt.timeout", 5)),
        phrase_time_limit=float(config_get("stt.phrase_time_limit", 10)),
    )

    state = {
        "received": None,
        "last_sent": "",
        "typed_buffer": "",
        "type_mode": not use_speech,
    }
    stop_event = asyncio.Event()
    outgoing_queue: asyncio.Queue[str] = asyncio.Queue()

    print(f"Connecting to {server_url}")
    async with websockets.connect(
        server_url,
        compression=None,
        ping_interval=20,
        ping_timeout=20,
        max_queue=1,
    ) as websocket:
        print("Connected to normal user session.")
        loop = asyncio.get_running_loop()

        if use_speech:
            def _on_speech(text: str):
                if stop_event.is_set() or state["type_mode"]:
                    return
                cleaned = text.strip()
                if cleaned:
                    loop.call_soon_threadsafe(outgoing_queue.put_nowait, cleaned)

            stt.listen_continuous(_on_speech)

        async def receive_task():
            last_spoken = None
            try:
                async for raw_message in websocket:
                    print("RECEIVED:", raw_message)
                    text = decode_message(raw_message, crypto)
                    parsed = parse_message(text)
                    if not parsed or not parsed.get("text"):
                        continue

                    translated = translator.translate(parsed["text"])
                    cleaned = str(translated).strip()
                    state["received"] = (
                        {"text": cleaned, "timestamp": time.time()} if cleaned else None
                    )
                    if state["received"] and cleaned != last_spoken:
                        tts.speak(cleaned)
                        last_spoken = cleaned
            finally:
                state["received"] = None

        async def input_task():
            while not stop_event.is_set():
                text = await outgoing_queue.get()
                cleaned = text.strip()
                if not cleaned or cleaned == state["last_sent"]:
                    continue
                print("SENDING:", cleaned)
                payload = crypto.encrypt(cleaned) if crypto else cleaned
                await websocket.send(payload)
                state["last_sent"] = cleaned

        receiver = asyncio.create_task(receive_task())
        sender = asyncio.create_task(input_task())

        try:
            while not stop_event.is_set():
                if use_speech and not state["type_mode"] and not is_speech_available(stt):
                    state["type_mode"] = True

                if receiver.done():
                    task_exception = receiver.exception()
                    if task_exception is not None:
                        raise task_exception
                    raise OSError("Normal user receive task ended.")

                if sender.done():
                    task_exception = sender.exception()
                    if task_exception is not None:
                        raise task_exception
                    raise OSError("Normal user input task ended.")

                frame = render_frame(
                    state["received"],
                    f"Connected to {server_url}",
                    title="DualTalk - Normal User",
                    message_label="Received from sign user:",
                    footer_text="[Q: quit]  [M: mute]  [T: type mode]",
                    status_lines=build_status_lines(state, use_speech, stt, tts),
                )
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_event.set()
                    break
                if key == ord("m"):
                    tts.toggle()
                elif key == ord("t"):
                    if use_speech:
                        state["type_mode"] = not state["type_mode"]
                    else:
                        state["type_mode"] = True
                elif state["type_mode"] and key != 255:
                    handle_type_keypress(key, state, outgoing_queue)

                await asyncio.sleep(0.01)
        finally:
            stop_event.set()
            stt.stop()
            receiver.cancel()
            sender.cancel()
            for task in (receiver, sender):
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
            cv2.destroyAllWindows()


async def run_with_reconnect(
    server_url: str,
    use_speech: bool,
    language_value: str | None,
    stt_language_override: str | None,
    tts_enabled: bool,
    crypto=None,
):
    while True:
        try:
            await normal_user_loop(
                server_url,
                use_speech=use_speech,
                language_value=language_value,
                stt_language_override=stt_language_override,
                tts_enabled=tts_enabled,
                crypto=crypto,
            )
            break
        except (OSError, websockets.exceptions.ConnectionClosed) as exc:
            print(f"Normal user disconnected: {exc}")
            print(f"Retrying in {RECONNECT_DELAY_SECONDS:.0f}s...")
            await asyncio.sleep(RECONNECT_DELAY_SECONDS)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the DualTalk normal user client.")
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER_URL,
        help="WebSocket server URL, for example ws://192.168.1.20:8765",
    )
    parser.add_argument(
        "--speech",
        action="store_true",
        help="Use microphone speech recognition for input.",
    )
    parser.add_argument(
        "--type",
        action="store_true",
        help="Start in keyboard typing mode instead of speech mode.",
    )
    parser.add_argument(
        "--language",
        default=str(config_get("stt.language", "en-US")),
        help="Language name/code for translation or STT locale such as en-US.",
    )
    parser.add_argument(
        "--stt-language",
        default=None,
        help="Optional STT locale override, for example en-US or hi-IN.",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable automatic text-to-speech playback.",
    )
    parser.add_argument(
        "--encrypt",
        action="store_true",
        default=bool(config_get("communication.encrypted", False)),
        help="Encrypt outgoing messages and decrypt incoming ones with a shared key.",
    )
    parser.add_argument(
        "--key",
        default=config_get("communication.encryption_key"),
        help="Shared Fernet key, optionally prefixed with DUALTALK_KEY=",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_speech = True
    if args.type:
        use_speech = False
    elif args.speech:
        use_speech = True

    crypto = resolve_crypto(args.encrypt, key_str=args.key, generate_if_missing=False)
    try:
        asyncio.run(
            run_with_reconnect(
                args.server,
                use_speech=use_speech,
                language_value=args.language,
                stt_language_override=args.stt_language,
                tts_enabled=not args.no_tts,
                crypto=crypto,
            )
        )
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("Normal user stopped.")


if __name__ == "__main__":
    main()

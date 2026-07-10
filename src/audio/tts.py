import queue
import threading

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class TTS:
    """
    Offline text-to-speech backed by a worker thread so callers never block UI loops.
    """

    def __init__(
        self,
        rate: int = 150,
        volume: float = 1.0,
        enabled: bool = True,
        engine_factory=None,
    ):
        self._rate = rate
        self._volume = volume
        self._engine_factory = engine_factory or (
            pyttsx3.init if pyttsx3 is not None else None
        )
        self._enabled = bool(enabled and self._engine_factory is not None)
        self._available = self._engine_factory is not None
        self._engine = None
        self._lock = threading.Lock()
        self._worker_lock = threading.Lock()
        self._tts_busy = False
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = None

        if self._available:
            self._start_worker()

    def _start_worker(self):
        with self._worker_lock:
            if not self._available or self._stop_event.is_set():
                return
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._worker_thread = threading.Thread(
                target=self._worker,
                name="DualTalkTTS",
                daemon=True,
            )
            self._worker_thread.start()

    def _ensure_worker(self):
        if not self._available or self._stop_event.is_set():
            return
        worker_thread = self._worker_thread
        if worker_thread is None or not worker_thread.is_alive():
            self._start_worker()

    def _worker(self):
        self._set_busy(False)
        print("TTS READY")
        try:
            while not self._stop_event.is_set():
                try:
                    text = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if text is None:
                    break

                if not self.enabled:
                    continue

                self._set_busy(True)
                print("TTS START")
                engine = None
                try:
                    engine = self._create_engine()
                    if engine is None:
                        continue
                    engine.say(text)
                    engine.runAndWait()
                except Exception as exc:
                    print(f"TTS ERROR: {exc}")
                    self._recover_from_failure()
                finally:
                    self._dispose_engine(engine)
                    self._set_busy(False)
                    print("TTS COMPLETE")
                    print("TTS READY")
        finally:
            self._set_busy(False)

    def _create_engine(self):
        try:
            engine = self._engine_factory()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            with self._lock:
                self._engine = engine
            return engine
        except Exception as exc:
            print(f"TTS ERROR: {exc}")
            with self._lock:
                self._engine = None
            return None

    def _dispose_engine(self, engine):
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
        with self._lock:
            if self._engine is engine:
                self._engine = None

    def _recover_from_failure(self):
        recovery_engine = self._create_engine()
        self._dispose_engine(recovery_engine)

    def _set_busy(self, value: bool):
        with self._lock:
            self._tts_busy = bool(value)

    def speak(self, text: str):
        cleaned = str(text or "").strip()
        if not cleaned or not self.enabled:
            return False

        self._ensure_worker()
        if self._worker_thread is None or not self._worker_thread.is_alive():
            return False

        try:
            self._queue.put_nowait(cleaned)
            return True
        except queue.Full:
            return False

    def set_rate(self, rate: int):
        self._rate = rate
        with self._lock:
            engine = self._engine
        if engine is not None:
            try:
                engine.setProperty("rate", rate)
            except Exception:
                pass

    def set_volume(self, volume: float):
        self._volume = volume
        with self._lock:
            engine = self._engine
        if engine is not None:
            try:
                engine.setProperty("volume", volume)
            except Exception:
                pass

    def enable(self):
        if self._available:
            self._enabled = True
            self._ensure_worker()
        return self._enabled

    def disable(self):
        self._enabled = False
        self._clear_pending()
        return self._enabled

    def toggle(self) -> bool:
        if self.enabled:
            return self.disable()
        return self.enable()

    def _clear_pending(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    @property
    def enabled(self) -> bool:
        return self._available and self._enabled

    @property
    def available(self) -> bool:
        return self._available

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._tts_busy

    def shutdown(self):
        self._enabled = False
        self._stop_event.set()
        self._clear_pending()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._worker_thread.join(timeout=1.0)
        self._set_busy(False)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

from __future__ import annotations

import csv
import ctypes
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None


DEFAULT_CSV_PATH = Path(__file__).resolve().parents[2] / "logs" / "metrics.csv"
CSV_HEADERS = [
    "timestamp",
    "gesture",
    "intent",
    "sentence",
    "fps",
    "detection_ms",
    "intent_ms",
    "sentence_ms",
    "network_ms",
    "total_ms",
    "cpu_percent",
    "ram_percent",
]
FPS_WINDOW_SIZE = 60
SYSTEM_SAMPLE_INTERVAL_SECONDS = 0.5


@dataclass
class MetricsRecord:
    sample_id: str
    timestamp: str
    gesture: str = ""
    intent: str = ""
    sentence: str = ""
    fps: float = 0.0
    detection_ms: float | None = None
    intent_ms: float | None = None
    sentence_ms: float | None = None
    network_ms: float | None = None
    total_ms: float | None = None
    cpu_percent: float | None = None
    ram_percent: float | None = None
    frame_started_at: float = field(default=0.0, repr=False)
    detection_completed_at: float | None = field(default=None, repr=False)
    intent_started_at: float | None = field(default=None, repr=False)
    intent_completed_at: float | None = field(default=None, repr=False)
    sentence_started_at: float | None = field(default=None, repr=False)
    sentence_completed_at: float | None = field(default=None, repr=False)
    send_started_at: float | None = field(default=None, repr=False)


@dataclass
class LiveMetricsSnapshot:
    fps: float = 0.0
    latency_ms: float = 0.0
    detection_ms: float = 0.0
    intent_ms: float = 0.0
    sentence_ms: float = 0.0
    network_ms: float = 0.0
    total_ms: float = 0.0
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    gesture: str = ""
    intent: str = ""
    sentence: str = ""


class _WindowsSystemSampler:
    class FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", ctypes.c_uint32),
            ("dwHighDateTime", ctypes.c_uint32),
        ]

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32),
            ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    def __init__(self):
        self._last_cpu_times = self._get_cpu_times()

    def sample(self):
        return self._sample_cpu_percent(), self._sample_ram_percent()

    def _sample_cpu_percent(self):
        current_times = self._get_cpu_times()
        if current_times is None:
            return 0.0
        if self._last_cpu_times is None:
            self._last_cpu_times = current_times
            return 0.0

        idle_now, kernel_now, user_now = current_times
        idle_before, kernel_before, user_before = self._last_cpu_times
        idle_delta = idle_now - idle_before
        kernel_delta = kernel_now - kernel_before
        user_delta = user_now - user_before
        total_delta = kernel_delta + user_delta

        self._last_cpu_times = current_times
        if total_delta <= 0:
            return 0.0

        busy_delta = total_delta - idle_delta
        return max(0.0, min(100.0, (busy_delta * 100.0) / total_delta))

    def _sample_ram_percent(self):
        memory_status = self.MEMORYSTATUSEX()
        memory_status.dwLength = ctypes.sizeof(self.MEMORYSTATUSEX)
        success = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
        if not success:
            return 0.0
        return float(memory_status.dwMemoryLoad)

    def _get_cpu_times(self):
        idle_time = self.FILETIME()
        kernel_time = self.FILETIME()
        user_time = self.FILETIME()
        success = ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        )
        if not success:
            return None

        return (
            self._filetime_to_int(idle_time),
            self._filetime_to_int(kernel_time),
            self._filetime_to_int(user_time),
        )

    @staticmethod
    def _filetime_to_int(filetime):
        return (filetime.dwHighDateTime << 32) | filetime.dwLowDateTime


class _FallbackSystemSampler:
    def sample(self):
        return 0.0, 0.0


def _create_system_sampler():
    if psutil is not None:
        psutil.cpu_percent(interval=None)
        return psutil

    if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "kernel32"):
        return _WindowsSystemSampler()

    return _FallbackSystemSampler()


class MetricsEngine:
    def __init__(
        self,
        csv_path: str | Path = DEFAULT_CSV_PATH,
        fps_window_size: int = FPS_WINDOW_SIZE,
        print_metrics: bool = False,
    ):
        self.csv_path = Path(csv_path)
        self.fps_window_size = max(2, int(fps_window_size))
        self.print_metrics = print_metrics
        self._lock = threading.RLock()
        self._frame_times = deque(maxlen=self.fps_window_size)
        self._records: dict[str, MetricsRecord] = {}
        self._message_to_sample: dict[str, str] = {}
        self._current_sample_id: str | None = None
        self._system_sampler = _create_system_sampler()
        self._csv_file = None
        self._csv_writer = None
        self._latest_snapshot = LiveMetricsSnapshot()
        self._last_system_sample_at = 0.0
        self._ensure_csv_writer()

    def tick_frame(self, frame_started_at: float | None = None):
        with self._lock:
            timestamp = frame_started_at if frame_started_at is not None else time.perf_counter()
            self._frame_times.append(timestamp)
            self._latest_snapshot.fps = self._compute_fps()
            self._refresh_system_usage()
            return self._latest_snapshot.fps

    def start_sample(
        self,
        gesture: str | None = None,
        sample_id: str | None = None,
        frame_started_at: float | None = None,
    ):
        with self._lock:
            started_at = (
                frame_started_at if frame_started_at is not None else time.perf_counter()
            )
            wall_clock = datetime.now().astimezone().isoformat(timespec="milliseconds")
            resolved_sample_id = sample_id or self._new_id(prefix="frame")
            record = MetricsRecord(
                sample_id=resolved_sample_id,
                timestamp=wall_clock,
                gesture=gesture or "",
                fps=self._latest_snapshot.fps or self._compute_fps(),
                frame_started_at=started_at,
            )
            self._records[resolved_sample_id] = record
            self._current_sample_id = resolved_sample_id
            if gesture:
                self._latest_snapshot.gesture = gesture
            return resolved_sample_id

    def start_frame(
        self,
        gesture: str | None = None,
        sample_id: str | None = None,
        frame_started_at: float | None = None,
    ):
        return self.start_sample(
            gesture=gesture,
            sample_id=sample_id,
            frame_started_at=frame_started_at,
        )

    def mark_detection(
        self,
        gesture: str | None,
        sample_id: str | None = None,
        detected_at: float | None = None,
    ):
        with self._lock:
            record = self._get_record(sample_id)
            timestamp = detected_at if detected_at is not None else time.perf_counter()
            record.gesture = gesture or record.gesture
            record.detection_completed_at = timestamp
            record.detection_ms = self._ms_since(record.frame_started_at, timestamp)
            self._latest_snapshot.gesture = record.gesture
            self._latest_snapshot.detection_ms = record.detection_ms or 0.0
            self._latest_snapshot.total_ms = record.detection_ms or 0.0
            self._latest_snapshot.latency_ms = self._latest_snapshot.total_ms
            return record.detection_ms

    def mark_intent(
        self,
        intent: str | None = None,
        sample_id: str | None = None,
        intent_started_at: float | None = None,
        intent_completed_at: float | None = None,
    ):
        with self._lock:
            record = self._get_record(sample_id)
            resolved_started_at = (
                intent_started_at
                if intent_started_at is not None
                else record.detection_completed_at or record.frame_started_at
            )
            resolved_completed_at = (
                intent_completed_at
                if intent_completed_at is not None
                else time.perf_counter()
            )
            record.intent = str(intent or "").strip()
            record.intent_started_at = resolved_started_at
            record.intent_completed_at = resolved_completed_at
            record.intent_ms = self._ms_since(
                resolved_started_at,
                resolved_completed_at,
            )
            record.total_ms = self._ms_since(record.frame_started_at, resolved_completed_at)
            self._latest_snapshot.intent = record.intent
            self._latest_snapshot.intent_ms = record.intent_ms or 0.0
            self._latest_snapshot.total_ms = record.total_ms or 0.0
            self._latest_snapshot.latency_ms = self._latest_snapshot.total_ms
            return record.intent_ms

    def mark_sentence(
        self,
        sentence: str | None = None,
        sample_id: str | None = None,
        sentence_started_at: float | None = None,
        sentence_completed_at: float | None = None,
    ):
        with self._lock:
            record = self._get_record(sample_id)
            resolved_started_at = (
                sentence_started_at
                if sentence_started_at is not None
                else record.intent_completed_at
                or record.detection_completed_at
                or record.frame_started_at
            )
            resolved_completed_at = (
                sentence_completed_at
                if sentence_completed_at is not None
                else time.perf_counter()
            )
            record.sentence = str(sentence or "").strip()
            record.sentence_started_at = resolved_started_at
            record.sentence_completed_at = resolved_completed_at
            record.sentence_ms = self._ms_since(
                resolved_started_at,
                resolved_completed_at,
            )
            record.total_ms = self._ms_since(record.frame_started_at, resolved_completed_at)
            self._latest_snapshot.sentence = record.sentence
            self._latest_snapshot.sentence_ms = record.sentence_ms or 0.0
            self._latest_snapshot.total_ms = record.total_ms or 0.0
            self._latest_snapshot.latency_ms = self._latest_snapshot.total_ms
            return record.sentence_ms

    def mark_send(
        self,
        sample_id: str | None = None,
        message_id: str | None = None,
        sent_at: float | None = None,
    ):
        with self._lock:
            record = self._get_record(sample_id)
            timestamp = sent_at if sent_at is not None else time.perf_counter()
            resolved_message_id = message_id or self._new_id(prefix="msg")
            record.send_started_at = timestamp
            self._message_to_sample[resolved_message_id] = record.sample_id
            record.total_ms = self._ms_since(record.frame_started_at, timestamp)
            self._latest_snapshot.total_ms = record.total_ms or 0.0
            self._latest_snapshot.latency_ms = self._latest_snapshot.total_ms
            return resolved_message_id

    def mark_delivery(
        self,
        message_id: str,
        delivered_at: float | None = None,
    ):
        with self._lock:
            sample_id = self._message_to_sample.pop(message_id, None)
            if sample_id is None:
                return None

            record = self._records.get(sample_id)
            if record is None:
                return None

            timestamp = delivered_at if delivered_at is not None else time.perf_counter()
            network_base = (
                record.send_started_at
                or record.sentence_completed_at
                or record.intent_completed_at
                or record.detection_completed_at
                or record.frame_started_at
            )
            record.network_ms = self._ms_since(network_base, timestamp)
            record.total_ms = self._ms_since(record.frame_started_at, timestamp)
            return self._finalize_record(sample_id)

    def mark_receiver_update(
        self,
        message_id: str,
        receiver_updated_at: float | None = None,
    ):
        return self.mark_delivery(
            message_id=message_id,
            delivered_at=receiver_updated_at,
        )

    def discard_sample(self, sample_id: str | None = None):
        with self._lock:
            record = self._get_record(sample_id)
            if self._current_sample_id == record.sample_id:
                self._current_sample_id = None
            stale_message_ids = [
                message_id
                for message_id, mapped_sample_id in self._message_to_sample.items()
                if mapped_sample_id == record.sample_id
            ]
            for message_id in stale_message_ids:
                self._message_to_sample.pop(message_id, None)
            self._records.pop(record.sample_id, None)

    def finalize_local(self, sample_id: str | None = None):
        with self._lock:
            record = self._get_record(sample_id)
            if record.total_ms is None:
                final_base = (
                    record.send_started_at
                    or record.sentence_completed_at
                    or record.intent_completed_at
                    or record.detection_completed_at
                    or record.frame_started_at
                )
                record.total_ms = self._ms_since(record.frame_started_at, final_base)
            return self._finalize_record(record.sample_id)

    def flush_pending_without_network(self):
        with self._lock:
            sample_ids = list(self._records)
        committed = []
        for sample_id in sample_ids:
            committed.append(self.finalize_local(sample_id=sample_id))
        return committed

    def get_live_snapshot(self):
        with self._lock:
            self._refresh_system_usage(force=False)
            return LiveMetricsSnapshot(**self._latest_snapshot.__dict__)

    def close(self):
        with self._lock:
            if self._csv_file is not None:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None

    def _finalize_record(self, sample_id: str):
        record = self._records.pop(sample_id)
        if self._current_sample_id == sample_id:
            self._current_sample_id = None

        stale_message_ids = [
            message_id
            for message_id, mapped_sample_id in self._message_to_sample.items()
            if mapped_sample_id == sample_id
        ]
        for message_id in stale_message_ids:
            self._message_to_sample.pop(message_id, None)

        self._refresh_system_usage(force=True)
        record.cpu_percent = self._latest_snapshot.cpu_percent
        record.ram_percent = self._latest_snapshot.ram_percent
        self._latest_snapshot.gesture = record.gesture
        self._latest_snapshot.intent = record.intent
        self._latest_snapshot.sentence = record.sentence
        self._latest_snapshot.detection_ms = record.detection_ms or 0.0
        self._latest_snapshot.intent_ms = record.intent_ms or 0.0
        self._latest_snapshot.sentence_ms = record.sentence_ms or 0.0
        self._latest_snapshot.network_ms = record.network_ms or 0.0
        self._latest_snapshot.total_ms = record.total_ms or 0.0
        self._latest_snapshot.latency_ms = self._latest_snapshot.total_ms
        self._write_csv_row(record)
        if self.print_metrics:
            self._print_record(record)
        return record

    def _refresh_system_usage(self, force: bool = False):
        now = time.perf_counter()
        if not force and (now - self._last_system_sample_at) < SYSTEM_SAMPLE_INTERVAL_SECONDS:
            return

        cpu_percent, ram_percent = self._sample_system_usage()
        self._latest_snapshot.cpu_percent = cpu_percent
        self._latest_snapshot.ram_percent = ram_percent
        self._last_system_sample_at = now

    def _sample_system_usage(self):
        if psutil is not None and self._system_sampler is psutil:
            return (
                float(psutil.cpu_percent(interval=None)),
                float(psutil.virtual_memory().percent),
            )

        return self._system_sampler.sample()

    def _compute_fps(self):
        if len(self._frame_times) < 2:
            return 0.0

        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed <= 0:
            return 0.0

        return (len(self._frame_times) - 1) / elapsed

    def _write_csv_row(self, record: MetricsRecord):
        if self._csv_writer is None:
            self._ensure_csv_writer()

        row = {
            "timestamp": record.timestamp,
            "gesture": record.gesture,
            "intent": record.intent,
            "sentence": record.sentence,
            "fps": self._format_csv_number(record.fps),
            "detection_ms": self._format_csv_number(record.detection_ms),
            "intent_ms": self._format_csv_number(record.intent_ms),
            "sentence_ms": self._format_csv_number(record.sentence_ms),
            "network_ms": self._format_csv_number(record.network_ms),
            "total_ms": self._format_csv_number(record.total_ms),
            "cpu_percent": self._format_csv_number(record.cpu_percent),
            "ram_percent": self._format_csv_number(record.ram_percent),
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def _ensure_csv_writer(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self._csv_file = self.csv_path.open("a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_HEADERS)
        if not file_exists:
            self._csv_writer.writeheader()
            self._csv_file.flush()

    def _get_record(self, sample_id: str | None):
        resolved_sample_id = sample_id or self._current_sample_id
        if resolved_sample_id is None or resolved_sample_id not in self._records:
            raise KeyError("No active metrics sample found. Call start_sample() first.")
        return self._records[resolved_sample_id]

    @staticmethod
    def _new_id(prefix: str):
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _ms_since(start_time: float, end_time: float):
        return max(0.0, (end_time - start_time) * 1000.0)

    @staticmethod
    def _format_csv_number(value: float | None):
        if value is None:
            return ""
        return f"{value:.3f}"

    @staticmethod
    def _format_print_number(value: float | None, suffix: str = ""):
        if value is None:
            return "N/A"
        return f"{value:.2f}{suffix}"

    def _print_record(self, record: MetricsRecord):
        print(f"FPS: {self._format_print_number(record.fps)}")
        print(
            f"DETECTION LATENCY: "
            f"{self._format_print_number(record.detection_ms, ' ms')}"
        )
        print(f"INTENT LATENCY: {self._format_print_number(record.intent_ms, ' ms')}")
        print(
            f"SENTENCE LATENCY: "
            f"{self._format_print_number(record.sentence_ms, ' ms')}"
        )
        print(
            f"NETWORK LATENCY: "
            f"{self._format_print_number(record.network_ms, ' ms')}"
        )
        print(f"TOTAL LATENCY: {self._format_print_number(record.total_ms, ' ms')}")
        print(f"CPU: {self._format_print_number(record.cpu_percent, ' %')}")
        print(f"RAM: {self._format_print_number(record.ram_percent, ' %')}")


_global_engine: MetricsEngine | None = None


def get_global_engine(csv_path: str | Path = DEFAULT_CSV_PATH):
    global _global_engine
    if _global_engine is None:
        _global_engine = MetricsEngine(csv_path=csv_path)
    return _global_engine


def tick_frame(frame_started_at: float | None = None):
    return get_global_engine().tick_frame(frame_started_at=frame_started_at)


def start_sample(
    gesture: str | None = None,
    sample_id: str | None = None,
    frame_started_at: float | None = None,
):
    return get_global_engine().start_sample(
        gesture=gesture,
        sample_id=sample_id,
        frame_started_at=frame_started_at,
    )


def start_frame(
    gesture: str | None = None,
    sample_id: str | None = None,
    frame_started_at: float | None = None,
):
    return get_global_engine().start_frame(
        gesture=gesture,
        sample_id=sample_id,
        frame_started_at=frame_started_at,
    )


def mark_detection(
    gesture: str | None,
    sample_id: str | None = None,
    detected_at: float | None = None,
):
    return get_global_engine().mark_detection(
        gesture=gesture,
        sample_id=sample_id,
        detected_at=detected_at,
    )


def mark_intent(
    intent: str | None = None,
    sample_id: str | None = None,
    intent_started_at: float | None = None,
    intent_completed_at: float | None = None,
):
    return get_global_engine().mark_intent(
        intent=intent,
        sample_id=sample_id,
        intent_started_at=intent_started_at,
        intent_completed_at=intent_completed_at,
    )


def mark_sentence(
    sentence: str | None = None,
    sample_id: str | None = None,
    sentence_started_at: float | None = None,
    sentence_completed_at: float | None = None,
):
    return get_global_engine().mark_sentence(
        sentence=sentence,
        sample_id=sample_id,
        sentence_started_at=sentence_started_at,
        sentence_completed_at=sentence_completed_at,
    )


def mark_send(
    sample_id: str | None = None,
    message_id: str | None = None,
    sent_at: float | None = None,
):
    return get_global_engine().mark_send(
        sample_id=sample_id,
        message_id=message_id,
        sent_at=sent_at,
    )


def mark_delivery(message_id: str, delivered_at: float | None = None):
    return get_global_engine().mark_delivery(
        message_id=message_id,
        delivered_at=delivered_at,
    )


def mark_receiver_update(message_id: str, receiver_updated_at: float | None = None):
    return get_global_engine().mark_receiver_update(
        message_id=message_id,
        receiver_updated_at=receiver_updated_at,
    )


def discard_sample(sample_id: str | None = None):
    return get_global_engine().discard_sample(sample_id=sample_id)


def finalize_local(sample_id: str | None = None):
    return get_global_engine().finalize_local(sample_id=sample_id)


def flush_pending_without_network():
    return get_global_engine().flush_pending_without_network()


def get_live_snapshot():
    return get_global_engine().get_live_snapshot()


def close():
    return get_global_engine().close()

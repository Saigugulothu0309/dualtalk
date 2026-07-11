import os
from copy import deepcopy

try:
    import yaml
except ImportError:
    raise ImportError(
        "Missing dependency: PyYAML. Install using 'pip install pyyaml'"
    )


_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "config.yaml",
)

_DEFAULT_CONFIG = {
    "server": {
        "host": "0.0.0.0",
        "port": 8765,
        "url": "ws://127.0.0.1:8765/ws",
    },
    "camera": {
        "index": 0,
        "width": 1280,
        "height": 720,
        "fps": 30,
    },
    "detection": {
        "max_num_hands": 2,
        "detection_confidence": 0.7,
        "tracking_confidence": 0.7,
        "confidence_threshold": 0.75,
        "stable_frame_count": 5,
        "wave_buffer_size": 12,
    },
    "sentence_builder": {
        "timeout_seconds": 2.0,
    },
    "gestures": {
        "profile": "hospital",
        "profiles_dir": os.path.join("config", "gestures"),
    },
    "communication": {
        "send_debounce_seconds": 0.2,
        "reconnect_delay_seconds": 2.0,
        "encrypted": False,
        "encryption_key": None,
    },
    "tts": {
        "rate": 150,
        "volume": 1.0,
        "enabled": True,
    },
    "stt": {
        "language": "en-US",
        "timeout": 5,
        "phrase_time_limit": 10,
    },
    "language": {
        "default": "en",
    },
    "ui": {
        "offline_width": 1280,
        "offline_height": 720,
    },
}


def _merge_dicts(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str = _DEFAULT_CONFIG_PATH) -> dict:
    config = deepcopy(_DEFAULT_CONFIG)
    if not os.path.exists(path):
        return config

    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            loaded = yaml.safe_load(file_obj) or {}
    except OSError:
        return config
    except yaml.YAMLError as exc:
        print(f"Warning: failed to parse config file '{path}': {exc}")
        return config

    if not isinstance(loaded, dict):
        return config

    return _merge_dicts(config, loaded)


_config: dict | None = None


def get(key_path: str, default=None):
    global _config
    if _config is None:
        _config = load_config()

    value = _config
    for part in key_path.split("."):
        if not isinstance(value, dict):
            return default
        value = value.get(part, default)

    return default if value is None else value

"""Persistent configuration management for NoteTaker."""

import json
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG_PATH = os.path.join(_SCRIPT_DIR, ".notetaker_config.json")

DEFAULTS = {
    "hf_token": "",
    "model": "medium",
    "model_cache_dir": "",  # Custom cache dir for models; blank = default ~/.cache
    "ollama_url": "http://localhost:11434",
    "ollama_model": "llama3.2:3b",
    "default_export_dir": os.path.join(_SCRIPT_DIR, "exports"),
    "audio_device_loopback": "",
    "audio_device_mic": "",
}


class ConfigManager:
    def __init__(self, config_path: str = None):
        self._path = config_path or _DEFAULT_CONFIG_PATH
        self._data: dict = {}
        self.load()

    def load(self) -> dict:
        try:
            with open(self._path, "r") as f:
                self._data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = {}
        return self._data

    def save(self) -> None:
        try:
            with open(self._path, "w") as f:
                json.dump(self._data, f, indent=2)
        except OSError:
            pass

    def get(self, key: str, default=None):
        if default is None:
            default = DEFAULTS.get(key)
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value
        self.save()

    def set_many(self, updates: dict) -> None:
        self._data.update(updates)
        self.save()

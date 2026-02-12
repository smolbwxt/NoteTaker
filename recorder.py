"""Dual audio recording: WASAPI loopback (system audio) + microphone."""
from __future__ import annotations

import wave
import threading
import struct
import os
from typing import Tuple, List, Optional

import numpy as np


class DualAudioRecorder:
    """Records system audio via WASAPI loopback and microphone simultaneously."""

    CHUNK = 1024
    TARGET_RATE = 16000  # WhisperX expects 16 kHz

    def __init__(self):
        self._pa = None
        self._loopback_stream = None
        self._mic_stream = None
        self._loopback_frames: list[bytes] = []
        self._mic_frames: list[bytes] = []
        self._is_recording = False
        self._lock = threading.Lock()
        # Device info filled during recording
        self._loopback_rate: int = 0
        self._loopback_channels: int = 0
        self._mic_rate: int = 0
        self._mic_channels: int = 0

    # ------------------------------------------------------------------
    # Device enumeration
    # ------------------------------------------------------------------

    @staticmethod
    def get_audio_devices() -> Tuple[List[dict], List[dict]]:
        """Return (loopback_devices, mic_devices).

        Each entry: {"index": int, "name": str, "sample_rate": int, "channels": int}
        """
        import pyaudiowpatch as pyaudio

        pa = pyaudio.PyAudio()
        loopback_devices: list[dict] = []
        mic_devices: list[dict] = []

        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                entry = {
                    "index": i,
                    "name": info.get("name", f"Device {i}"),
                    "sample_rate": int(info.get("defaultSampleRate", 44100)),
                    "channels": int(info.get("maxInputChannels", 0)),
                }
                if info.get("isLoopbackDevice", False):
                    loopback_devices.append(entry)
                elif entry["channels"] > 0:
                    mic_devices.append(entry)
        finally:
            pa.terminate()

        return loopback_devices, mic_devices

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def start_recording(
        self,
        loopback_device_index: Optional[int] = None,
        mic_device_index: Optional[int] = None,
    ) -> None:
        if self._is_recording:
            raise RuntimeError("Already recording")

        import pyaudiowpatch as pyaudio

        self._pa = pyaudio.PyAudio()
        self._loopback_frames = []
        self._mic_frames = []

        # -- Resolve loopback device --
        if loopback_device_index is not None:
            lb_info = self._pa.get_device_info_by_index(loopback_device_index)
        else:
            try:
                wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_speakers = self._pa.get_device_info_by_index(
                    wasapi_info["defaultOutputDevice"]
                )
                lb_info = self._pa.get_device_info_by_index(
                    default_speakers["index"]
                )
                # Try to find corresponding loopback device
                for i in range(self._pa.get_device_count()):
                    d = self._pa.get_device_info_by_index(i)
                    if d.get("isLoopbackDevice") and d.get("name", "").startswith(
                        default_speakers.get("name", "???")[:30]
                    ):
                        lb_info = d
                        break
            except Exception:
                raise ValueError(
                    "No WASAPI loopback device found.\n\n"
                    "On Windows, open Sound Settings → Recording and enable 'Stereo Mix', "
                    "or install a virtual audio cable."
                )

        self._loopback_rate = int(lb_info["defaultSampleRate"])
        self._loopback_channels = int(lb_info["maxInputChannels"]) or 2

        self._loopback_stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._loopback_channels,
            rate=self._loopback_rate,
            input=True,
            input_device_index=lb_info["index"],
            frames_per_buffer=self.CHUNK,
            stream_callback=self._loopback_callback,
        )

        # -- Resolve mic device --
        if mic_device_index is not None:
            mic_info = self._pa.get_device_info_by_index(mic_device_index)
        else:
            mic_info = self._pa.get_default_input_device_info()

        self._mic_rate = int(mic_info["defaultSampleRate"])
        self._mic_channels = int(mic_info["maxInputChannels"]) or 1

        self._mic_stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._mic_channels,
            rate=self._mic_rate,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=self.CHUNK,
            stream_callback=self._mic_callback,
        )

        self._is_recording = True
        self._loopback_stream.start_stream()
        self._mic_stream.start_stream()

    def stop_recording(self) -> None:
        if not self._is_recording:
            return

        self._is_recording = False

        for stream in (self._loopback_stream, self._mic_stream):
            if stream and stream.is_active():
                stream.stop_stream()
            if stream:
                stream.close()

        self._loopback_stream = None
        self._mic_stream = None

        if self._pa:
            self._pa.terminate()
            self._pa = None

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _loopback_callback(self, in_data, frame_count, time_info, status):
        import pyaudiowpatch as pyaudio

        with self._lock:
            if self._is_recording:
                self._loopback_frames.append(in_data)
        return (None, pyaudio.paContinue)

    def _mic_callback(self, in_data, frame_count, time_info, status):
        import pyaudiowpatch as pyaudio

        with self._lock:
            if self._is_recording:
                self._mic_frames.append(in_data)
        return (None, pyaudio.paContinue)

    # ------------------------------------------------------------------
    # Mixing & saving
    # ------------------------------------------------------------------

    def save_mixed_wav(
        self,
        output_path: str,
        loopback_gain: float = 1.0,
        mic_gain: float = 1.0,
    ) -> str:
        """Mix loopback + mic audio and write a mono 16 kHz WAV (ideal for Whisper)."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        lb_audio = self._frames_to_mono(
            self._loopback_frames, self._loopback_channels, self._loopback_rate
        )
        mic_audio = self._frames_to_mono(
            self._mic_frames, self._mic_channels, self._mic_rate
        )

        # Pad shorter array
        max_len = max(len(lb_audio), len(mic_audio))
        if len(lb_audio) < max_len:
            lb_audio = np.pad(lb_audio, (0, max_len - len(lb_audio)))
        if len(mic_audio) < max_len:
            mic_audio = np.pad(mic_audio, (0, max_len - len(mic_audio)))

        mixed = np.clip(
            lb_audio.astype(np.float32) * loopback_gain
            + mic_audio.astype(np.float32) * mic_gain,
            -32768,
            32767,
        ).astype(np.int16)

        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.TARGET_RATE)
            wf.writeframes(mixed.tobytes())

        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _frames_to_mono(frames: list[bytes], channels: int, source_rate: int) -> np.ndarray:
        """Convert raw frames to mono int16 array resampled to TARGET_RATE."""
        if not frames:
            return np.array([], dtype=np.int16)

        raw = b"".join(frames)
        audio = np.frombuffer(raw, dtype=np.int16)

        # Down-mix to mono if stereo+
        if channels > 1:
            # Reshape and mean across channels
            trim = len(audio) - (len(audio) % channels)
            audio = audio[:trim].reshape(-1, channels).mean(axis=1).astype(np.int16)

        # Resample to TARGET_RATE with proper anti-aliasing
        if source_rate != DualAudioRecorder.TARGET_RATE:
            from math import gcd
            from scipy.signal import resample_poly

            g = gcd(source_rate, DualAudioRecorder.TARGET_RATE)
            up = DualAudioRecorder.TARGET_RATE // g
            down = source_rate // g
            audio = resample_poly(audio.astype(np.float64), up, down).astype(np.int16)

        return audio

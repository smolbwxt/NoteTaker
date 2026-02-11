"""WhisperX transcription with optional speaker diarization."""
from __future__ import annotations

import os
import time
from typing import Optional, Callable




class TranscriptionResult:
    """Container for transcription output."""

    def __init__(self, segments: list, language: str, elapsed_time: float,
                 speaker_embeddings: dict = None):
        self.segments = segments
        self.language = language
        self.elapsed_time = elapsed_time
        self.speaker_embeddings = speaker_embeddings or {}

    def format_as_text(self, include_timestamps: bool = True) -> str:
        lines: list[str] = []
        diarized = any(seg.get("speaker") for seg in self.segments)

        lines.append(
            f"Language: {self.language}  |  Time: {self.elapsed_time:.1f}s  |  "
            f"Speakers: {'detected' if diarized else 'off'}"
        )
        lines.append("=" * 60)
        lines.append("")

        # ---- Clean transcript ----
        lines.append("TRANSCRIPT")
        lines.append("-" * 40)
        lines.append("")

        if diarized:
            current_speaker = None
            for seg in self.segments:
                speaker = seg.get("speaker", "???")
                text = seg.get("text", "").strip()
                if speaker != current_speaker:
                    if current_speaker is not None:
                        lines.append("")
                    lines.append(f"[{speaker}]")
                    current_speaker = speaker
                lines.append(f"  {text}")
        else:
            full_text = " ".join(seg.get("text", "").strip() for seg in self.segments)
            lines.append(full_text)

        lines.append("")
        lines.append("")

        # ---- Timestamped segments ----
        if include_timestamps:
            lines.append("TIMESTAMPED SEGMENTS")
            lines.append("-" * 40)
            lines.append("")

            for seg in self.segments:
                t_start = _fmt_time(seg.get("start", 0))
                t_end = _fmt_time(seg.get("end", 0))
                text = seg.get("text", "").strip()
                speaker = seg.get("speaker", "")
                speaker_tag = f"  ({speaker})" if speaker else ""
                lines.append(f"[{t_start} -> {t_end}]{speaker_tag}  {text}")

        return "\n".join(lines)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class WhisperXTranscriber:
    """Transcribe audio files using WhisperX."""

    SUPPORTED_EXTENSIONS = (".mp3", ".mp4", ".wav", ".m4a", ".webm", ".flac", ".ogg")
    MODEL_OPTIONS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

    def __init__(
        self,
        model_name: str = "base",
        device: str = None,
        compute_type: str = None,
        download_root: str = None,
    ):
        self.model_name = model_name
        self.download_root = download_root
        # Auto-detect device/compute
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        if compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type

    def transcribe(
        self,
        audio_path: str,
        enable_diarization: bool = False,
        hf_token: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> TranscriptionResult:
        offline = os.environ.get("HF_HUB_OFFLINE") == "1"
        if enable_diarization and not offline and not (hf_token and hf_token.strip()):
            raise ValueError("Speaker diarization requires a Hugging Face token.")

        def _status(msg: str):
            if progress_callback:
                progress_callback(msg)

        import whisperx

        _status(f"Loading '{self.model_name}' model ({self.device})...")
        load_kwargs = {}
        if self.download_root:
            load_kwargs["download_root"] = self.download_root
        model = whisperx.load_model(
            self.model_name, self.device, compute_type=self.compute_type,
            **load_kwargs
        )

        _status("Loading audio...")
        audio = whisperx.load_audio(audio_path)

        _status("Transcribing...")
        start = time.time()
        batch_size = 16 if self.device == "cuda" else 4
        result = model.transcribe(audio, batch_size=batch_size)
        language = result.get("language", "en")

        # Align word-level timestamps
        _status("Aligning word timestamps...")
        model_a, metadata = whisperx.load_align_model(
            language_code=language, device=self.device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        # Free alignment model memory
        import gc, torch
        del model_a
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Diarization
        speaker_embeddings = {}
        if enable_diarization:
            _status("Running speaker diarization (this can take a while)...")
            from whisperx.diarize import DiarizationPipeline
            auth_token = hf_token.strip() if hf_token else None
            diarize_model = DiarizationPipeline(
                use_auth_token=auth_token, device=self.device
            )
            kwargs = {}
            if min_speakers is not None and min_speakers > 0:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None and max_speakers > 0:
                kwargs["max_speakers"] = max_speakers

            diarize_segments = diarize_model(audio, **kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)

            _status("Extracting speaker voice prints...")
            speaker_embeddings = self._extract_embeddings(
                audio, diarize_segments, self.device
            )

        elapsed = time.time() - start
        segments = result.get("segments", [])
        return TranscriptionResult(segments, language, elapsed, speaker_embeddings)

    @staticmethod
    def _extract_embeddings(audio_np, diarize_segments, device: str) -> dict:
        """Extract per-speaker voice embeddings from diarized audio."""
        try:
            import numpy as np
            import torch
            from pyannote.audio import Inference

            inference = Inference(
                "pyannote/wespeaker-voxceleb-resnet34-LM",
                use_auth_token=None,
                window="whole",
                device=torch.device(device),
            )
            sr = 16000  # whisperx uses 16kHz

            # Group time ranges by speaker from the diarization DataFrame
            speaker_times: dict[str, list] = {}
            for _, row in diarize_segments.iterrows():
                speaker = str(row["speaker"])
                start = float(row["start"])
                end = float(row["end"])
                speaker_times.setdefault(speaker, []).append((start, end))

            embeddings = {}
            for speaker, times in speaker_times.items():
                chunks = []
                for start, end in times:
                    s_idx = int(start * sr)
                    e_idx = int(end * sr)
                    if e_idx > s_idx and e_idx <= len(audio_np):
                        chunks.append(audio_np[s_idx:e_idx])
                if not chunks:
                    continue

                combined = np.concatenate(chunks)
                # Limit to 60s for performance
                if len(combined) > 60 * sr:
                    combined = combined[: 60 * sr]

                waveform = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)
                emb = inference({"waveform": waveform, "sample_rate": sr})
                embeddings[speaker] = emb.flatten().tolist()

            return embeddings
        except Exception:
            return {}

"""WhisperX transcription with optional speaker diarization."""

import time
from typing import Optional, Callable




class TranscriptionResult:
    """Container for transcription output."""

    def __init__(self, segments: list, language: str, elapsed_time: float):
        self.segments = segments
        self.language = language
        self.elapsed_time = elapsed_time

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
    ):
        self.model_name = model_name
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
        if enable_diarization and not (hf_token and hf_token.strip()):
            raise ValueError("Speaker diarization requires a Hugging Face token.")

        def _status(msg: str):
            if progress_callback:
                progress_callback(msg)

        import whisperx

        _status(f"Loading '{self.model_name}' model ({self.device})...")
        model = whisperx.load_model(
            self.model_name, self.device, compute_type=self.compute_type
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
        if enable_diarization:
            _status("Running speaker diarization (this can take a while)...")
            from whisperx.diarize import DiarizationPipeline
            diarize_model = DiarizationPipeline(
                use_auth_token=hf_token.strip(), device=self.device
            )
            kwargs = {}
            if min_speakers is not None and min_speakers > 0:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None and max_speakers > 0:
                kwargs["max_speakers"] = max_speakers

            diarize_segments = diarize_model(audio, **kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        elapsed = time.time() - start
        segments = result.get("segments", [])
        return TranscriptionResult(segments, language, elapsed)

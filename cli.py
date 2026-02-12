#!/usr/bin/env python3
from __future__ import annotations

"""Headless CLI for NoteTaker — run on HPC or any machine without a GUI.

Examples:
    # Basic transcription (auto-detects GPU):
    python cli.py meeting.wav

    # Faster: use a smaller model
    python cli.py meeting.wav --model small

    # With speaker diarization:
    python cli.py meeting.wav --diarize

    # Process a whole directory:
    python cli.py recordings/ --model medium --diarize

    # Custom output directory:
    python cli.py meeting.wav -o /scratch/results/

    # Force CPU even if GPU is available:
    python cli.py meeting.wav --device cpu
"""

import os
import sys
import glob
import time
import argparse

# Offline and torch compat — must be set before any imports
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_script_dir = os.path.dirname(os.path.abspath(__file__))

# Apply custom model cache dir if configured
from config import ConfigManager
_cfg = ConfigManager()
_cache = _cfg.get("model_cache_dir", "")
if _cache and os.path.isdir(_cache):
    os.environ.setdefault("HF_HOME", os.path.join(_cache, "huggingface"))
    os.environ.setdefault("TORCH_HOME", os.path.join(_cache, "torch"))


SUPPORTED = (".mp3", ".mp4", ".wav", ".m4a", ".webm", ".flac", ".ogg")


def find_audio_files(path: str) -> list[str]:
    """Return a list of audio files from a path (file or directory)."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        files = []
        for ext in SUPPORTED:
            files.extend(glob.glob(os.path.join(path, f"*{ext}")))
        files.sort()
        return files
    print(f"ERROR: '{path}' is not a file or directory", file=sys.stderr)
    sys.exit(1)


def process_file(
    audio_path: str,
    output_dir: str,
    model_name: str,
    device: str | None,
    compute_type: str | None,
    diarize: bool,
    hf_token: str | None,
    min_speakers: int | None,
    max_speakers: int | None,
    speaker_db,
    summarize: bool = False,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "mistral:7b",
) -> None:
    """Transcribe (and optionally diarize) a single audio file."""
    from transcriber import WhisperXTranscriber
    from summarizer import build_clipboard_prompt, OllamaSummarizer
    from exporter import DocxExporter

    basename = os.path.splitext(os.path.basename(audio_path))[0]
    print(f"\n{'=' * 60}")
    print(f"  {os.path.basename(audio_path)}")
    print(f"{'=' * 60}")

    def status(msg: str):
        print(f"  [{_elapsed()}] {msg}")

    download_root = None
    cache_dir = _cfg.get("model_cache_dir", "")
    if cache_dir:
        download_root = os.path.join(cache_dir, "huggingface", "hub")

    transcriber = WhisperXTranscriber(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        download_root=download_root,
    )

    result = transcriber.transcribe(
        audio_path=audio_path,
        enable_diarization=diarize,
        hf_token=hf_token,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        progress_callback=status,
    )

    # Speaker matching
    if result.speaker_embeddings and speaker_db:
        matched = 0
        for label, emb in result.speaker_embeddings.items():
            name = speaker_db.match(emb)
            if name:
                matched += 1
                for seg in result.segments:
                    if seg.get("speaker") == label:
                        seg["speaker"] = name
                speaker_db.add_or_update(name, emb)
        if matched:
            status(f"Matched {matched} speaker(s) from voice database")

    transcript_text = result.format_as_text()

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    txt_path = os.path.join(output_dir, f"{basename}_transcript.txt")
    DocxExporter.save_transcript_txt(transcript_text, txt_path)
    status(f"Transcript: {txt_path}")

    prompt_path = os.path.join(output_dir, f"{basename}_summary_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(build_clipboard_prompt(transcript_text))
    status(f"Summary prompt: {prompt_path}")

    # Summarize with Ollama if requested
    if summarize:
        summarizer = OllamaSummarizer(
            ollama_url=ollama_url,
            model=ollama_model,
        )
        if summarizer.check_available():
            status(f"Summarizing with {ollama_model}...")
            try:
                summary = summarizer.summarize(transcript_text, progress_callback=status)
                summary_path = os.path.join(output_dir, f"{basename}_summary.txt")
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(summary.raw_summary)
                status(f"Summary: {summary_path}")
            except Exception as e:
                status(f"Summarization failed (skipping): {e}")
        else:
            status(f"Ollama not available at {ollama_url} (skipping summarization)")

    # Print speaker list
    speakers = sorted(set(
        seg.get("speaker", "") for seg in result.segments if seg.get("speaker")
    ))
    if speakers:
        status(f"Speakers detected: {', '.join(speakers)}")

    print(f"  [{_elapsed()}] Done ({result.elapsed_time:.1f}s transcription time)")


_start_time = time.time()


def _elapsed() -> str:
    m, s = divmod(int(time.time() - _start_time), 60)
    return f"{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(
        description="NoteTaker CLI — headless transcription for HPC / batch processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input", help="Audio file or directory of audio files"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=os.path.join(_script_dir, "exports"),
        help="Output directory (default: ./exports/)",
    )
    parser.add_argument(
        "-m", "--model",
        default=_cfg.get("model", "medium"),
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: medium). Smaller = faster.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device: 'cuda' or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Compute type: float16, int8, etc. (default: auto)",
    )
    parser.add_argument(
        "--diarize", action="store_true",
        help="Enable speaker diarization",
    )
    parser.add_argument(
        "--hf-token", default=None,
        help="Hugging Face token for diarization (not needed in offline mode)",
    )
    parser.add_argument(
        "--min-speakers", type=int, default=None,
        help="Minimum number of speakers (diarization hint)",
    )
    parser.add_argument(
        "--max-speakers", type=int, default=None,
        help="Maximum number of speakers (diarization hint)",
    )
    parser.add_argument(
        "--no-speaker-db", action="store_true",
        help="Disable speaker voice database matching",
    )
    parser.add_argument(
        "--summarize", action="store_true",
        help="Summarize transcript with Ollama (skips if unavailable)",
    )
    parser.add_argument(
        "--ollama-url",
        default=_cfg.get("ollama_url", "http://localhost:11434"),
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--ollama-model",
        default=_cfg.get("ollama_model", "mistral:7b"),
        help="Ollama model for summarization (default: mistral:7b)",
    )

    args = parser.parse_args()

    files = find_audio_files(args.input)
    if not files:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    print(f"NoteTaker CLI")
    print(f"  Model:  {args.model}")
    print(f"  Device: {args.device or 'auto'}")
    print(f"  Files:  {len(files)}")
    print(f"  Output: {args.output_dir}")
    if args.diarize:
        print(f"  Diarization: enabled")
    if args.summarize:
        print(f"  Summarize: {args.ollama_model} @ {args.ollama_url}")

    # Load speaker DB
    speaker_db = None
    if not args.no_speaker_db:
        from speaker_db import SpeakerDB
        speaker_db = SpeakerDB(
            os.path.join(_script_dir, ".notetaker_speakers.json")
        )
        known = speaker_db.list_speakers()
        if known:
            print(f"  Known speakers: {', '.join(known)}")

    for audio_path in files:
        try:
            process_file(
                audio_path=audio_path,
                output_dir=args.output_dir,
                model_name=args.model,
                device=args.device,
                compute_type=args.compute_type,
                diarize=args.diarize,
                hf_token=args.hf_token,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                speaker_db=speaker_db,
                summarize=args.summarize,
                ollama_url=args.ollama_url,
                ollama_model=args.ollama_model,
            )
        except Exception as e:
            print(f"\n  ERROR processing {audio_path}: {e}", file=sys.stderr)

    print(f"\nAll done! [{_elapsed()}]")


if __name__ == "__main__":
    main()
